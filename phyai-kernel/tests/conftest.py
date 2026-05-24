"""Shared fixtures for phyai-kernel tests.

phyai-kernel's tests construct ``phyai.layers.*`` modules (RMSNorm,
AdaRMSNorm, MaskedEmbedding, etc.) directly. Those layers internally
build :class:`ReplicatedLinear` / :class:`ColumnParallelLinear` /
similar, whose ``__init__`` consults the
:class:`~phyai.layers.linear.dispatch.KernelDispatcher` singleton — so
:func:`phyai.layers.linear.init` MUST be called before construction or
the dispatcher raises ``RuntimeError("init... not called yet")``.

We also need a registered :class:`Mesh` named ``"model"`` for layers
that resolve TP collectives (``ReplicatedLinear`` short-circuits at
ws=1 but still asks for the mesh by name). A degenerate single-rank
mesh covers every test in this package.

The autouse fixture below:

1. Registers a degenerate ``"model"`` mesh.
2. Initialises the linear dispatcher with
   ``register_flashinfer=False, validate=False`` — phyai-kernel tests
   exercise Triton kernels, not flashinfer, and skipping flashinfer
   keeps construction CPU/GPU-driver-friendly.
3. Tears both down on exit so test isolation holds.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import phyai.layers.linear as L
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


def _register_fake_mesh(name: str = "model") -> Mesh:
    tm = MagicMock()
    tm.mesh_dim_names = ()
    tm.size.side_effect = lambda axis=None: 1
    tm.get_local_rank.side_effect = lambda axis=None: 0
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name=name)
    register_mesh(mesh)
    return mesh


@pytest.fixture(autouse=True)
def _phyai_layers_init():
    """Bootstrap ``phyai.layers.linear`` + a degenerate mesh per test."""
    saved_meshes = dict(_meshes)
    _register_fake_mesh()
    L.init(register_flashinfer=False, validate=False)
    try:
        yield
    finally:
        _meshes.clear()
        _meshes.update(saved_meshes)
        L._reset_for_test()
