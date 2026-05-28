"""Inter-process IPC buffer primitives for phyai.

Two **transport substrates** that let unrelated OS processes share the
same bytes:

* :class:`CudaIpcBuffer` — a GPU buffer (cudaMalloc) accessible from
  multiple processes via cudaIpc. The producing process calls
  :meth:`CudaIpcBuffer.create`, ships the resulting :class:`CudaIpcHandle`
  to a peer through any out-of-band channel (zmq, pipe, argv, file),
  and the peer calls :meth:`CudaIpcBuffer.attach` on that handle to
  obtain a non-owning view in its own address space. Both sides have
  full read/write access; the asymmetry is only at lifetime.

* :class:`HostShmBuffer` — a host POSIX shared-memory block. Same shape:
  :meth:`create` produces a :class:`HostShmHandle`, peer :meth:`attach`-es.
  Optional :attr:`cuda_register=True` registers the mapping with the
  CUDA driver as pinned/portable memory for zero-copy host↔device DMA.

Roles
-----
The buffer is symmetric — both creator and attacher read and write the
same bytes. **Producer/consumer semantics, sync flags, queues, and
lifetime coordination are application-layer concerns layered on top.**

Lifetime contract
-----------------
* The creator must outlive every attacher. Coordinate teardown via your
  own out-of-band protocol (e.g., ack messages, file locks, exit codes).
* :meth:`close` is idempotent and safe to call from finalizers.
* Tensor / numpy views obtained from :meth:`tensor` / :meth:`as_tensor` /
  :meth:`as_numpy` keep a strong reference to the buffer instance; they
  may safely outlive the local variable, but **NOT** the explicit
  :meth:`close` call. After ``close()`` the storage is released and any
  surviving view dangles.
* No built-in sync. To layer one, allocate a small :class:`HostShmBuffer`
  alongside the data buffer, view it as a counter, and increment under
  ``fcntl.flock``.

Multi-host
----------
``cudaIpc`` and POSIX shm are **node-local**. For cross-host transfers,
layer something else (RDMA, gRPC, ...) on top.

Example: tokenizer ↔ model
--------------------------
Tokenizer process::

    buf = CudaIpcBuffer.create(nbytes=4 << 20)             # 4 MiB on cuda:0
    buf.tensor(dtype=torch.bfloat16, shape=(...)).copy_(features)
    socket.send_pyobj({"buf": buf.handle, "shape": (...)}) # zmq, dataclass is pickle-friendly

Model process::

    msg = socket.recv_pyobj()
    buf = CudaIpcBuffer.attach(msg["buf"])
    features = buf.tensor(dtype=torch.bfloat16, shape=msg["shape"])
    # ... use features ...
    buf.close()  # releases this process's mapping; tokenizer still owns the bytes
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing.resource_tracker as resource_tracker
import os
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Literal

import numpy as np
import torch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA runtime helpers
# ---------------------------------------------------------------------------


def _import_cuda_runtime():
    """Lazy import of :mod:`cuda.bindings.runtime`.

    cuda-python is a transitive dep of phyai (via
    :mod:`phyai.vgpu.backends.flashinfer`); deferring the import keeps
    CPU-only entry points cheap.
    """
    try:
        import cuda.bindings.runtime as cuda_rt  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "phyai.runtime.ipc_buffer requires cuda-python (`pip install cuda-python`)."
        ) from e
    return cuda_rt


def _check_cuda(err: Any, what: str) -> None:
    """Convert a non-zero ``cudaError_t`` into a Python exception."""
    if int(err) != 0:
        cuda_rt = _import_cuda_runtime()
        _, msg = cuda_rt.cudaGetErrorString(err)
        raise RuntimeError(
            f"{what} failed: {msg.decode() if isinstance(msg, bytes) else msg}"
        )


def _ipc_handle_nbytes() -> int:
    """Length of ``cudaIpcMemHandle_t.reserved`` on the live CUDA runtime.

    64 today, but the type is opaque — query it once instead of hardcoding
    so an ABI change doesn't silently corrupt our handles.
    """
    cuda_rt = _import_cuda_runtime()
    return len(cuda_rt.cudaIpcMemHandle_t().reserved)


def _addr_of_buffer(buf) -> int:
    """Host VA of a writable Python buffer (e.g. ``shared_memory.SharedMemory.buf``)."""
    return ctypes.addressof(ctypes.c_byte.from_buffer(buf))


# ---------------------------------------------------------------------------
# Tensor / numpy wrapping over a raw pointer
# ---------------------------------------------------------------------------


_TORCH_DTYPE_TO_TYPESTR: dict[torch.dtype, str] = {
    torch.uint8: "<u1",
    torch.int8: "<i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.float16: "<f2",
    torch.bfloat16: "<V2",  # numpy has no bf16; opaque 2-byte view, torch reinterprets
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.bool: "<u1",
    torch.complex64: "<c8",
    torch.complex128: "<c16",
}


_TORCH_DTYPE_TO_NUMPY: dict[torch.dtype, type] = {
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: np.bool_,
}


def _resolve_shape_for_dtype(
    nbytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], int]:
    """Pick a default shape when ``shape is None``; validate fit otherwise.

    Returns ``(shape, total_bytes)``. Total bytes is computed in Python
    int (arbitrary precision) so >2 GiB allocations don't trip int32
    overflow somewhere down the line.
    """
    elem_size = torch.empty(0, dtype=dtype).element_size()
    if shape is None:
        if nbytes % elem_size != 0:
            raise ValueError(
                f"nbytes={nbytes} is not divisible by elem_size({elem_size}) "
                f"of dtype={dtype}; pass an explicit shape."
            )
        shape = (nbytes // elem_size,)
    n_elem = 1
    for s in shape:
        n_elem *= int(s)
    total = n_elem * elem_size
    if total > nbytes:
        raise ValueError(
            f"shape={shape} dtype={dtype} requires {total} bytes "
            f"but buffer has only {nbytes}."
        )
    return tuple(int(s) for s in shape), total


class _CudaArrayInterfaceWrapper:
    """``__cuda_array_interface__`` v3 provider over a raw device pointer.

    Holds a strong reference to the owning :class:`CudaIpcBuffer` via
    ``_keepalive`` so that a tensor created via ``torch.as_tensor(self)``
    keeps the buffer alive until the tensor is dropped — even if the
    user lets the local ``CudaIpcBuffer`` variable go out of scope.

    Calling :meth:`CudaIpcBuffer.close` is the explicit invalidation
    point; surviving tensor views on a closed buffer dangle.
    """

    __slots__ = ("__cuda_array_interface__", "_keepalive")

    def __init__(
        self,
        ptr: int,
        shape: tuple[int, ...],
        typestr: str,
        *,
        keepalive: object,
    ) -> None:
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),  # False = read/write
            "shape": tuple(shape),
            "typestr": typestr,
            "strides": None,
            "version": 3,
        }
        self._keepalive = keepalive


def _wrap_cuda_ptr_as_tensor(
    ptr: int,
    nbytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None,
    device: torch.device,
    *,
    keepalive: object,
) -> torch.Tensor:
    """Wrap a raw CUDA pointer as a non-owning ``torch.Tensor`` view."""
    if dtype not in _TORCH_DTYPE_TO_TYPESTR:
        raise TypeError(
            f"Unsupported torch.dtype for IPC tensor view: {dtype!r}. "
            f"Add a mapping in phyai.runtime.ipc_buffer if you need this."
        )
    resolved_shape, _ = _resolve_shape_for_dtype(nbytes, dtype, shape)
    typestr = _TORCH_DTYPE_TO_TYPESTR[dtype]
    wrapper = _CudaArrayInterfaceWrapper(
        ptr, resolved_shape, typestr, keepalive=keepalive
    )
    with torch.cuda.device(device):
        return torch.as_tensor(wrapper)


# ---------------------------------------------------------------------------
# Handles — serializable, transport-agnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CudaIpcHandle:
    """Serializable handle to a :class:`CudaIpcBuffer`.

    Pickle / dataclass-friendly: ship through any transport — ZMQ
    ``send_pyobj``, ``subprocess.Popen`` argv as base64, on-disk file,
    UNIX pipe — whatever the application uses. The receiver passes it
    to :meth:`CudaIpcBuffer.attach` to map the buffer.
    """

    handle_bytes: bytes  # ``cudaIpcMemHandle_t.reserved`` payload
    nbytes: int
    device_index: int  # the producer's CUDA device


@dataclass(frozen=True)
class HostShmHandle:
    """Serializable handle to a :class:`HostShmBuffer`."""

    name: str
    nbytes: int


# ---------------------------------------------------------------------------
# CudaIpcBuffer
# ---------------------------------------------------------------------------


def _resolve_cuda_device(
    device: "torch.device | int | str | None",
) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, str):
        d = torch.device(device)
    elif isinstance(device, torch.device):
        d = device
    else:
        raise TypeError(f"Unsupported device argument: {device!r}")
    if d.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {d!r}")
    return torch.device(
        "cuda", d.index if d.index is not None else torch.cuda.current_device()
    )


class CudaIpcBuffer:
    """A GPU buffer accessible from multiple processes via cudaIpc.

    Construction
    ------------
    Use the classmethods, not ``__init__`` directly:

    * :meth:`create` — allocate a fresh buffer; ``close()`` => ``cudaFree``.
    * :meth:`attach` — map a buffer created in another process by handle;
      ``close()`` => ``cudaIpcCloseMemHandle`` (no free).

    Both modes yield equivalent read/write access in this process.
    """

    __slots__ = (
        "_nbytes",
        "_device",
        "_origin_device_index",
        "_local_ptr",
        "_handle_bytes",
        "_close_kind",
        "_closed",
    )

    def __init__(
        self,
        *,
        nbytes: int,
        device: torch.device,
        origin_device_index: int,
        local_ptr: int,
        handle_bytes: bytes,
        close_kind: Literal["free", "ipc_close"],
    ) -> None:
        self._nbytes = int(nbytes)
        self._device = device
        self._origin_device_index = int(origin_device_index)
        self._local_ptr = int(local_ptr)
        self._handle_bytes = handle_bytes
        self._close_kind = close_kind
        self._closed = False

    # ---------- constructors ---------- #

    @classmethod
    def create(
        cls,
        nbytes: int,
        *,
        device: "torch.device | int | str | None" = None,
    ) -> "CudaIpcBuffer":
        """Allocate a fresh GPU buffer in this process.

        The returned buffer's :attr:`handle` is the serializable token
        peer processes pass to :meth:`attach`.
        """
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        if not torch.cuda.is_available():
            raise RuntimeError("CudaIpcBuffer requires CUDA, but it is unavailable")

        cuda_rt = _import_cuda_runtime()
        dev = _resolve_cuda_device(device)
        with torch.cuda.device(dev):
            err, ptr = cuda_rt.cudaMalloc(int(nbytes))
            _check_cuda(err, "cudaMalloc")
            (err,) = cuda_rt.cudaMemset(ptr, 0, int(nbytes))
            _check_cuda(err, "cudaMemset")
            err, raw_handle = cuda_rt.cudaIpcGetMemHandle(ptr)
            _check_cuda(err, "cudaIpcGetMemHandle")
            handle_bytes = bytes(raw_handle.reserved)

        logger.debug(
            "CudaIpcBuffer.create: %d bytes on %s, handle_len=%d",
            nbytes,
            dev,
            len(handle_bytes),
        )
        return cls(
            nbytes=nbytes,
            device=dev,
            origin_device_index=dev.index,
            local_ptr=int(ptr),
            handle_bytes=handle_bytes,
            close_kind="free",
        )

    @classmethod
    def attach(
        cls,
        handle: CudaIpcHandle,
        *,
        device: "torch.device | int | str | None" = None,
    ) -> "CudaIpcBuffer":
        """Map a buffer created in another process.

        ``device`` defaults to ``handle.device_index``. Pass an explicit
        ``cuda:N`` to attach the buffer onto a different GPU on this
        node — the lazy-peer-access flag handles cross-card mapping
        (P2P-capable topology required, e.g. NVLink or P2P PCIe). The
        guard call wraps the open in ``with torch.cuda.device(target)``
        so the CUDAGuard sits on the consumer's device.
        """
        if not isinstance(handle, CudaIpcHandle):
            raise TypeError(f"expected CudaIpcHandle, got {type(handle).__name__}")
        if handle.nbytes <= 0:
            raise ValueError(f"handle.nbytes must be positive, got {handle.nbytes}")
        if not torch.cuda.is_available():
            raise RuntimeError("CudaIpcBuffer requires CUDA, but it is unavailable")

        cuda_rt = _import_cuda_runtime()
        expected_len = _ipc_handle_nbytes()
        if len(handle.handle_bytes) != expected_len:
            raise ValueError(
                f"handle.handle_bytes length {len(handle.handle_bytes)} "
                f"does not match cudaIpcMemHandle_t size {expected_len}"
            )

        if device is None:
            target_dev = torch.device("cuda", handle.device_index)
        else:
            target_dev = _resolve_cuda_device(device)

        peer_handle = cuda_rt.cudaIpcMemHandle_t()
        peer_handle.reserved = handle.handle_bytes

        with torch.cuda.device(target_dev):
            err, ptr = cuda_rt.cudaIpcOpenMemHandle(
                peer_handle, cuda_rt.cudaIpcMemLazyEnablePeerAccess
            )
            _check_cuda(err, "cudaIpcOpenMemHandle")

        logger.debug(
            "CudaIpcBuffer.attach: %d bytes on %s (origin device cuda:%d)",
            handle.nbytes,
            target_dev,
            handle.device_index,
        )
        return cls(
            nbytes=handle.nbytes,
            device=target_dev,
            origin_device_index=handle.device_index,
            local_ptr=int(ptr),
            handle_bytes=handle.handle_bytes,
            close_kind="ipc_close",
        )

    # ---------- accessors ---------- #

    @property
    def handle(self) -> CudaIpcHandle:
        """Serializable handle. Re-exportable from attachers too."""
        if self._closed:
            raise RuntimeError("CudaIpcBuffer is closed")
        return CudaIpcHandle(
            handle_bytes=self._handle_bytes,
            nbytes=self._nbytes,
            device_index=self._origin_device_index,
        )

    @property
    def is_creator(self) -> bool:
        return self._close_kind == "free"

    @property
    def local_ptr(self) -> int:
        return self._local_ptr

    @property
    def device(self) -> torch.device:
        """Device the buffer is mapped on in *this* process.

        For attachers using a cross-device redirect this is the local
        device, NOT the origin device that originally created the buffer.
        """
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    # ---------- views ---------- #

    def tensor(
        self,
        dtype: torch.dtype = torch.uint8,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Non-owning ``torch.Tensor`` view over the buffer.

        Defaults to a ``(nbytes,)`` ``uint8`` view; pass ``dtype`` /
        ``shape`` to reinterpret. The view holds a strong reference to
        ``self`` so the buffer stays alive while any tensor is live —
        but :meth:`close` is the *explicit* invalidation point and
        leaves any surviving view dangling.
        """
        if self._closed:
            raise RuntimeError("CudaIpcBuffer is closed")
        return _wrap_cuda_ptr_as_tensor(
            self._local_ptr,
            self._nbytes,
            dtype,
            shape,
            self._device,
            keepalive=self,
        )

    # ---------- lifecycle ---------- #

    def close(self) -> None:
        """Release this process's mapping.

        * Creator: ``cudaFree`` (releases the underlying allocation).
        * Attacher: ``cudaIpcCloseMemHandle`` (unmaps; underlying memory
          remains owned by the creator).

        Idempotent. Errors are logged at DEBUG and swallowed — by the
        time ``close`` runs the CUDA context may already be in shutdown.
        """
        if self._closed:
            return
        self._closed = True
        if self._local_ptr == 0:
            return

        try:
            cuda_rt = _import_cuda_runtime()
        except RuntimeError:
            return

        try:
            with torch.cuda.device(self._device):
                if self._close_kind == "free":
                    cuda_rt.cudaFree(self._local_ptr)
                elif self._close_kind == "ipc_close":
                    cuda_rt.cudaIpcCloseMemHandle(self._local_ptr)
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            logger.debug("CudaIpcBuffer.close suppressed: %s", e)
        finally:
            self._local_ptr = 0

    def __enter__(self) -> "CudaIpcBuffer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# HostShmBuffer
# ---------------------------------------------------------------------------


def _new_shm_name() -> str:
    # ``phyai_ipc_`` (10) + 12 hex chars = 22 chars, well under macOS's
    # 31-char shm-name limit. uuid4 hex carries enough entropy that pid
    # is unnecessary (and would mislead post-fork).
    return f"phyai_ipc_{uuid.uuid4().hex[:12]}"


class HostShmBuffer:
    """Host POSIX shared-memory buffer accessible from multiple processes.

    Construction
    ------------
    * :meth:`create` — create a fresh shm block.
    * :meth:`attach` — open an existing block by name from a peer's
      :class:`HostShmHandle`.

    Both modes yield equivalent read/write access in this process.

    Optional ``cuda_register=True`` registers the mapping with the CUDA
    driver (``cudaHostRegisterPortable``) so the bytes are pinned and
    accessible from CUDA streams. Each process opts in independently;
    both sides may register without conflict (it's per-process state).
    """

    __slots__ = (
        "_nbytes",
        "_name",
        "_shm",
        "_close_kind",
        "_cuda_register",
        "_cuda_registered",
        "_closed",
    )

    def __init__(
        self,
        *,
        nbytes: int,
        name: str,
        shm: shared_memory.SharedMemory,
        close_kind: Literal["unlink", "noop"],
        cuda_register: bool,
        cuda_registered: bool,
    ) -> None:
        self._nbytes = int(nbytes)
        self._name = name
        self._shm = shm
        self._close_kind = close_kind
        self._cuda_register = cuda_register
        self._cuda_registered = cuda_registered
        self._closed = False

    # ---------- constructors ---------- #

    @classmethod
    def create(
        cls,
        nbytes: int,
        *,
        name: str | None = None,
        cuda_register: bool = False,
    ) -> "HostShmBuffer":
        """Create a fresh POSIX shm block.

        ``name=None`` auto-generates a unique name. The
        :attr:`handle` value is the serializable token to send to peers.
        """
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        chosen = name or _new_shm_name()
        shm = shared_memory.SharedMemory(name=chosen, create=True, size=int(nbytes))
        # Linux POSIX shm zero-initialises but be explicit.
        shm.buf[: int(nbytes)] = b"\x00" * int(nbytes)

        cuda_registered = False
        if cuda_register:
            cuda_registered = _do_cuda_host_register(shm, int(nbytes))

        logger.debug(
            "HostShmBuffer.create: name=%s nbytes=%d cuda_register=%s",
            chosen,
            nbytes,
            cuda_registered,
        )
        return cls(
            nbytes=nbytes,
            name=chosen,
            shm=shm,
            close_kind="unlink",
            cuda_register=cuda_register,
            cuda_registered=cuda_registered,
        )

    @classmethod
    def attach(
        cls,
        handle: HostShmHandle,
        *,
        cuda_register: bool = False,
    ) -> "HostShmBuffer":
        """Open an existing shm block by name."""
        if not isinstance(handle, HostShmHandle):
            raise TypeError(f"expected HostShmHandle, got {type(handle).__name__}")
        if handle.nbytes <= 0:
            raise ValueError(f"handle.nbytes must be positive, got {handle.nbytes}")
        shm = shared_memory.SharedMemory(name=handle.name)
        if shm.size < handle.nbytes:
            shm.close()
            raise RuntimeError(
                f"HostShmBuffer.attach: shm name={handle.name!r} has size={shm.size} "
                f"< handle.nbytes={handle.nbytes}"
            )
        # We're an attacher, not the owner. The stdlib's resource_tracker
        # would otherwise queue this name for unlink at process exit and
        # race the creator's unlink. Public unregister API (Python 3.4+)
        # tells the tracker we don't own it.
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            logger.debug("resource_tracker.unregister failed: %s", e)

        cuda_registered = False
        if cuda_register:
            cuda_registered = _do_cuda_host_register(shm, int(handle.nbytes))

        logger.debug(
            "HostShmBuffer.attach: name=%s nbytes=%d cuda_register=%s",
            handle.name,
            handle.nbytes,
            cuda_registered,
        )
        return cls(
            nbytes=handle.nbytes,
            name=handle.name,
            shm=shm,
            close_kind="noop",
            cuda_register=cuda_register,
            cuda_registered=cuda_registered,
        )

    # ---------- accessors ---------- #

    @property
    def handle(self) -> HostShmHandle:
        if self._closed:
            raise RuntimeError("HostShmBuffer is closed")
        return HostShmHandle(name=self._name, nbytes=self._nbytes)

    @property
    def is_creator(self) -> bool:
        return self._close_kind == "unlink"

    @property
    def name(self) -> str:
        return self._name

    @property
    def nbytes(self) -> int:
        return self._nbytes

    # ---------- views ---------- #

    def as_numpy(
        self,
        dtype: "np.dtype | type | str" = np.uint8,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Numpy view over the shared memory."""
        if self._closed or self._shm is None:
            raise RuntimeError("HostShmBuffer is closed")
        np_dtype = np.dtype(dtype)
        if shape is None:
            if self._nbytes % np_dtype.itemsize != 0:
                raise ValueError(
                    f"nbytes={self._nbytes} not divisible by itemsize={np_dtype.itemsize} "
                    f"of dtype={np_dtype}; pass an explicit shape."
                )
            shape = (self._nbytes // np_dtype.itemsize,)
        n_elem = 1
        for s in shape:
            n_elem *= int(s)
        if n_elem * np_dtype.itemsize > self._nbytes:
            raise ValueError(
                f"shape={shape} dtype={np_dtype} requires "
                f"{n_elem * np_dtype.itemsize} bytes but buffer has only {self._nbytes}."
            )
        return np.ndarray(shape, dtype=np_dtype, buffer=self._shm.buf)

    def as_tensor(
        self,
        dtype: torch.dtype = torch.uint8,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Torch tensor view backed by the same bytes as :meth:`as_numpy`.

        ``bfloat16`` is exposed via an opaque ``uint16`` view and
        reinterpreted with ``Tensor.view`` (numpy has no bf16).
        """
        if dtype == torch.bfloat16:
            np_view = self.as_numpy(np.uint16, shape)
            return torch.from_numpy(np_view).view(torch.bfloat16)
        if dtype not in _TORCH_DTYPE_TO_NUMPY:
            raise TypeError(f"Unsupported dtype for HostShmBuffer.as_tensor: {dtype!r}")
        np_view = self.as_numpy(_TORCH_DTYPE_TO_NUMPY[dtype], shape)
        return torch.from_numpy(np_view)

    # ---------- lifecycle ---------- #

    def close(self) -> None:
        """Release this process's mapping.

        * Creator: ``cudaHostUnregister`` (if registered) → ``shm.close``
          → ``shm.unlink``.
        * Attacher: ``cudaHostUnregister`` (if registered) → ``shm.close``.

        ``cudaHostUnregister`` MUST run before ``shm.close`` so we don't
        unregister a stale (already-unmapped) host VA.
        """
        if self._closed:
            return
        self._closed = True
        if self._shm is None:
            return

        if self._cuda_registered:
            try:
                cuda_rt = _import_cuda_runtime()
                cuda_rt.cudaHostUnregister(_addr_of_buffer(self._shm.buf))
            except Exception as e:  # pragma: no cover  # noqa: BLE001
                logger.debug("cudaHostUnregister failed: %s", e)
            self._cuda_registered = False

        try:
            self._shm.close()
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            logger.debug("SharedMemory.close failed: %s", e)
        if self._close_kind == "unlink":
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:  # pragma: no cover  # noqa: BLE001
                logger.debug("SharedMemory.unlink failed: %s", e)
        self._shm = None

    def __enter__(self) -> "HostShmBuffer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover  # noqa: BLE001
            pass


def _do_cuda_host_register(shm: shared_memory.SharedMemory, nbytes: int) -> bool:
    """Round ``nbytes`` up to a page and call ``cudaHostRegisterPortable``.

    Returns ``True`` iff registration succeeded and we should
    ``cudaHostUnregister`` at close. Raises on hard failure.
    """
    cuda_rt = _import_cuda_runtime()
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    aligned = ((nbytes + page - 1) // page) * page
    if aligned > shm.size:
        # Caller passed an `nbytes` larger than the shm block was created
        # with; refuse to silently extend.
        raise ValueError(
            f"cuda_register requires page-aligned size {aligned} but shm has only {shm.size}"
        )
    addr = _addr_of_buffer(shm.buf)
    (err,) = cuda_rt.cudaHostRegister(addr, aligned, cuda_rt.cudaHostRegisterPortable)
    _check_cuda(err, "cudaHostRegister")
    return True


__all__ = [
    "CudaIpcBuffer",
    "CudaIpcHandle",
    "HostShmBuffer",
    "HostShmHandle",
]
