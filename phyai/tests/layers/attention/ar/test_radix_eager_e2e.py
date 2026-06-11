"""CPU end-to-end: radix prefix reuse through ARAttention(eager) equals a
full recompute over the same logical sequence."""

from __future__ import annotations

import struct

import torch

from phyai_ext.radix_cache import PrefixCache

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    ARAttention,
    ARAttnCtx,
    Attention,
    AttnLayout,
    AttnMode,
)
from phyai.layers.attention.ar import EagerARBackend
from phyai.layers.attention.ar.radix import RadixAttentionPlanner, RadixSequence


def _atoms(tokens):
    return struct.pack(f"{len(tokens)}i", *tokens)


def _ctx_for(meta, pool):
    backend = EagerARBackend()
    plan = backend.init_forward_metadata(meta)
    return ARAttnCtx(
        backend=backend,
        plan=plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=meta.write_indices,
    )


def test_radix_eager_e2e_prefix_reuse_matches_full_recompute():
    torch.manual_seed(0)
    H, H_kv, D = 4, 2, 8
    N, P = 6, 3
    q = torch.randn(N, H, D)
    k = torch.randn(N, H_kv, D)
    v = torch.randn(N, H_kv, D)

    # Reference: full causal attention over the whole sequence; suffix rows.
    ref_layer = Attention(
        num_heads=H, head_dim=D, num_kv_heads=H_kv, causal=True, backend="eager"
    )
    ref_suffix = ref_layer(
        q, k, v, cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32)
    )[P:]

    cache = PrefixCache(atom_bytes=4, atoms_per_unit=1, device_total_units=64)
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=H_kv,
        head_dim=D,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    planner = RadixAttentionPlanner(cache, pool)
    layer = ARAttention(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H_kv,
        causal=True,
        backend="eager",
    )

    toks = [10, 11, 12, 20, 21, 22]
    # Seed the shared prefix [10,11,12] with K/V[0:P] at its slots.
    a = RadixSequence(_atoms(toks[:P]))
    layer(q[:P], k[:P], v[:P], _ctx_for(planner.plan([a]), pool))
    planner.commit([a])
    # Decoy to push the reuse suffix off the prefix's contiguous range.
    d = RadixSequence(_atoms([55, 66]))
    layer(
        torch.randn(2, H, D),
        torch.randn(2, H_kv, D),
        torch.randn(2, H_kv, D),
        _ctx_for(planner.plan([d]), pool),
    )
    planner.commit([d])
    # Reuse: query = suffix tokens only; KV = cached prefix + written suffix.
    c = RadixSequence(_atoms(toks))
    meta_c = planner.plan([c])
    assert c.prefix_len == P
    assert meta_c.paged_kv_indices.tolist() == [1, 2, 3, 6, 7, 8]  # non-contiguous
    out = layer(q[P:], k[P:], v[P:], _ctx_for(meta_c, pool))

    assert torch.allclose(out, ref_suffix, atol=1e-5)
