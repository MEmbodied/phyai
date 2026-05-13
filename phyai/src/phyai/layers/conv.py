"""Conv{1,2,3}d wrappers tagged for the phyai loader system.

Plain ``torch.nn.Conv{1,2,3}d`` would suffice for compute, but the rest of
phyai expects every parameter to carry a ``param.loader`` attribute so
checkpoint loading can dispatch generically. These classes are nothing
more than ``F.conv{1,2,3}d`` with a :class:`ReplicatedLoader` stamped onto
weight and bias — no tensor-parallel sharding, no kernel dispatch.

Constructor signatures mirror ``torch.nn.Conv{1,2,3}d`` so existing
configs drop in unchanged: ``int`` / tuple / ``"same"`` / ``"valid"``
padding, grouped/depthwise convs via ``groups``, and the four
``padding_mode`` variants (``"zeros"``, ``"reflect"``, ``"replicate"``,
``"circular"``).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.loaders import ReplicatedLoader

_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]
_size_3_t = Union[int, Tuple[int, int, int]]

_VALID_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")
_VALID_PADDING_STRINGS = ("same", "valid")


def _ntuple(n: int, x: int | Tuple[int, ...]) -> Tuple[int, ...]:
    """Coerce an int or n-tuple of ints into a tuple of length ``n``."""
    if isinstance(x, int):
        return tuple([x] * n)
    t = tuple(x)
    if len(t) != n:
        raise ValueError(f"expected a length-{n} sequence, got {t!r}")
    return t


class _ConvNd(nn.Module):
    """Shared state for Conv{1,2,3}d.

    Subclasses set :attr:`_ndim` and call :meth:`_conv` from their
    ``forward`` with the matching ``F.conv{1,2,3}d``. The weight has the
    canonical PyTorch layout
    ``(out_channels, in_channels // groups, *kernel_size)``, so HuggingFace
    / ``nn.Conv*`` checkpoints copy in straight through
    :class:`ReplicatedLoader`.
    """

    _ndim: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...] | str,
        dilation: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError(f"groups must be >= 1, got {groups}")
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels={in_channels} not divisible by groups={groups}"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels={out_channels} not divisible by groups={groups}"
            )
        if padding_mode not in _VALID_PADDING_MODES:
            raise ValueError(
                f"padding_mode={padding_mode!r} not in {_VALID_PADDING_MODES!r}"
            )
        if isinstance(padding, str):
            if padding not in _VALID_PADDING_STRINGS:
                raise ValueError(
                    f"padding={padding!r} not in {_VALID_PADDING_STRINGS!r}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is incompatible with strided convolutions"
                )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # F.pad takes pads in reverse axis order with each axis getting
        # (left, right). Only used when padding_mode != "zeros", but it's
        # also the only way to spell padding="same" out for non-zero modes,
        # so precompute it for strings too.
        self._reversed_padding_repeated_twice: tuple[int, ...] = (
            self._build_reversed_pad(padding, kernel_size, dilation)
        )

        weight_shape = (out_channels, in_channels // groups) + tuple(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(weight_shape, dtype=dtype),
            requires_grad=False,
        )
        self.weight.loader = ReplicatedLoader()  # type: ignore[attr-defined]
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=dtype),
                requires_grad=False,
            )
            self.bias.loader = ReplicatedLoader()  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _build_reversed_pad(
        padding: tuple[int, ...] | str,
        kernel_size: tuple[int, ...],
        dilation: tuple[int, ...],
    ) -> tuple[int, ...]:
        n = len(kernel_size)
        out = [0] * (2 * n)
        if isinstance(padding, str):
            if padding == "same":
                for d, k, i in zip(dilation, kernel_size, range(n - 1, -1, -1)):
                    total = d * (k - 1)
                    left = total // 2
                    out[2 * i] = left
                    out[2 * i + 1] = total - left
            return tuple(out)
        for i in range(n):
            out[2 * (n - 1 - i)] = padding[i]
            out[2 * (n - 1 - i) + 1] = padding[i]
        return tuple(out)

    def _conv(self, fn, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return fn(
                x,
                self.weight,
                self.bias,
                self.stride,
                0,
                self.dilation,
                self.groups,
            )
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != tuple([0] * self._ndim):
            s += f", padding={self.padding!r}"
        if self.dilation != tuple([1] * self._ndim):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class Conv1d(_ConvNd):
    """1-D convolution. Mirrors :class:`torch.nn.Conv1d` for inference."""

    _ndim = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t | str = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            _ntuple(1, kernel_size),
            _ntuple(1, stride),
            padding if isinstance(padding, str) else _ntuple(1, padding),
            _ntuple(1, dilation),
            groups,
            bias,
            padding_mode,
            dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(F.conv1d, x)


class Conv2d(_ConvNd):
    """2-D convolution. Mirrors :class:`torch.nn.Conv2d` for inference."""

    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            _ntuple(2, kernel_size),
            _ntuple(2, stride),
            padding if isinstance(padding, str) else _ntuple(2, padding),
            _ntuple(2, dilation),
            groups,
            bias,
            padding_mode,
            dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(F.conv2d, x)


class Conv3d(_ConvNd):
    """3-D convolution. Mirrors :class:`torch.nn.Conv3d` for inference."""

    _ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t | str = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            _ntuple(3, kernel_size),
            _ntuple(3, stride),
            padding if isinstance(padding, str) else _ntuple(3, padding),
            _ntuple(3, dilation),
            groups,
            bias,
            padding_mode,
            dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(F.conv3d, x)


__all__ = ["Conv1d", "Conv2d", "Conv3d"]
