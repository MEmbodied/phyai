"""Shared mesh fixture for tests/weights/."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


def _make_fake_mesh(
    *, sizes: dict[str, int] | None = None, ranks: dict[str, int] | None = None
) -> Mesh:
    sizes = sizes or {"tp": 1}
    ranks = ranks or {}
    tm = MagicMock()
    tm.mesh_dim_names = tuple(sizes.keys())
    _names = tm.mesh_dim_names
    tm.size.side_effect = lambda axis: sizes.get(
        axis if isinstance(axis, str) else _names[axis], 1
    )
    tm.get_local_rank.side_effect = lambda axis: ranks.get(axis, 0)
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name="model")
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    saved = dict(_meshes)
    try:
        yield _make_fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
