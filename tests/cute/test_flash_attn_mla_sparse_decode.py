"""Sparse-MLA (DSA) forward on the decode kernel: fewer than 128 query heads per KV head."""

import pytest
import torch

import test_flash_attn_mla_sparse_flat_kv as flat_kv
from test_flash_attn_mla_sparse_flat_kv import IS_SM100, SOFTMAX_SCALE, TILE_N, TOPK

from flash_attn.cute import flash_attn_varlen_func

NUM_KV_ROWS = 4096
HEADS = [8, 16, 32, 64]
# one total_q per (G, S) class the split rule produces at 148 SMs, plus multi-wave grids
TOTAL_Q = [1, 9, 18, 37, 74, 148, 222, 300]

pytestmark = pytest.mark.skipif(not IS_SM100, reason="sparse MLA forward is SM100 only")


def ragged_counts(total_q):
    """Ragged top-k lengths including exact-zero rows and both tile-boundary cases."""
    pattern = [TOPK, 0, 1, TILE_N, TILE_N + 1, TOPK - 1, 777, 0, 129, 1024]
    return (pattern * (total_q // len(pattern) + 1))[:total_q]


def run_decode(q, qv, k, v, cu_seqlens_q, idx, valid_len):
    # return_lse=False is what routes the interface at these head counts to the decode kernel.
    return flat_kv.run_kernel(
        q, qv, k, v, cu_seqlens_q, NUM_KV_ROWS, idx, valid_len, return_lse=False
    )[0]


def check(
    q_lens, counts, num_heads=16, split_views=True, seed=0,
    dtype=torch.bfloat16, use_valid_length=True, pad=-1,
):
    device = "cuda"
    cu_seqlens_q = flat_kv.cu_seqlens_from(q_lens, device)
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, dtype=dtype, seed=seed)
    q, qv = flat_kv.make_q(
        sum(q_lens), device, dtype=dtype, seed=seed + 1,
        split_views=split_views, num_heads=num_heads,
    )
    idx, valid_len = flat_kv.make_indices(
        counts, NUM_KV_ROWS, device, seed=seed + 2, pad=pad,
    )

    out = run_decode(
        q, qv, k, v, cu_seqlens_q, idx, valid_len if use_valid_length else None,
    )
    out_ref, _ = flat_kv.ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = flat_kv.ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)

    for m, n in enumerate(counts):
        if n == 0:
            assert (out[m] == 0).all(), f"row {m}: expected all-zero output for 0 valid indices"
    sel = torch.tensor([m for m, n in enumerate(counts) if n > 0], device=device)
    o, o_ref, o_pt = out[sel].float(), out_ref[sel].float(), out_pt[sel].float()
    pt_err = (o_pt - o_ref).abs().max().item()
    err = (o - o_ref).abs().max().item()
    assert err <= 2 * pt_err + 1e-3, f"out max diff {err} vs torch-bf16 {pt_err}"


@pytest.mark.parametrize("total_q", TOTAL_Q)
@pytest.mark.parametrize("num_heads", HEADS)
def test_decode_one_token_per_request(num_heads, total_q):
    """q_len == 1 per request, strided q/qv views: the shape vLLM's decode path calls with."""
    check([1] * total_q, ragged_counts(total_q), num_heads=num_heads, seed=10 * total_q)


@pytest.mark.parametrize("num_heads", HEADS)
def test_decode_contiguous_q(num_heads):
    check([1] * 18, ragged_counts(18), num_heads=num_heads, split_views=False, seed=31)


def test_decode_varlen_q_len_gt_1():
    """Several query tokens per request, so cu_seqlens_q has fewer rows than total_q."""
    q_lens = [1, 5, 2, 8, 1, 3, 16, 4]
    check(q_lens, ragged_counts(sum(q_lens)), seed=41)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [8, 16, 32])
def test_decode_without_valid_length(dtype, num_heads):
    check([1] * 9, ragged_counts(9), num_heads=num_heads, dtype=dtype, use_valid_length=False)


@pytest.mark.parametrize("pad", [-1, -2147483648, 1 << 20])
def test_decode_out_of_range_padding_sentinels(pad):
    """The gather forms a gmem pointer from the raw index, so any sentinel must be predicated."""
    check([1] * 9, ragged_counts(9), pad=pad, seed=51)


@pytest.mark.parametrize("kv_heads", [[1, 2, 1], [2, 1, 2]])
def test_decode_cache_reuse_and_kv_offsets(kv_heads):
    """Changing the dynamic KV-head count must update grid Y on a cache hit."""
    from flash_attn.cute.interface import _flash_attn_fwd

    _flash_attn_fwd.compile_cache.clear()
    device = "cuda"
    tokens, rows_per_req, head_ratio = 148, 128, 8
    counts = [128, 0, 1, 127] * 37
    idx, valid = flat_kv.make_indices(counts, rows_per_req, device, topk=128)
    cuq = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    cuk = cuq * rows_per_req
    for heads_k in kv_heads:
        q, qv = flat_kv.make_q(tokens, device, num_heads=heads_k * head_ratio)
        kv = torch.randn(tokens * rows_per_req, heads_k, 576, device=device, dtype=q.dtype)
        out = flash_attn_varlen_func(
            q, kv[..., 512:], kv[..., :512], qv=qv,
            cu_seqlens_q=cuq, cu_seqlens_k=cuk,
            max_seqlen_q=1, max_seqlen_k=rows_per_req,
            gather_kv_indices=idx, gather_kv_valid_length=valid,
            softmax_scale=SOFTMAX_SCALE,
        )[0]
        for row in [0, 1, 2, 3, 147]:
            for head_k in range(heads_k):
                head_slice = slice(head_k * head_ratio, (head_k + 1) * head_ratio)
                kv_row = kv[row * rows_per_req:(row + 1) * rows_per_req, head_k:head_k + 1]
                ref, _ = flat_kv.ref_sparse_mla(
                    q[row:row + 1, head_slice], qv[row:row + 1, head_slice],
                    kv_row, idx[row:row + 1], counts[row:row + 1],
                )
                torch.testing.assert_close(
                    out[row:row + 1, head_slice], ref, atol=0.015, rtol=0.015,
                )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("tokens,topk", [(1, 2048), (148, 256)])
@pytest.mark.parametrize("use_valid_length", [False, True])
def test_decode_leading_empty_block(dtype, num_heads, tokens, topk, use_valid_length):
    """An empty first block must not clamp subsequent negative score maxima to zero."""
    q = torch.zeros(tokens, num_heads, 64, device="cuda", dtype=dtype)
    qv = torch.full((tokens, num_heads, 512), -10, device="cuda", dtype=dtype)
    k = torch.zeros(1, 1, 64, device="cuda", dtype=dtype)
    v = torch.ones(1, 1, 512, device="cuda", dtype=dtype)
    idx = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
    idx[:, TILE_N:2 * TILE_N] = 0
    valid = torch.full((tokens,), 2 * TILE_N, device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(
        q, k, v, qv=qv,
        cu_seqlens_q=torch.arange(tokens + 1, device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.zeros(tokens + 1, device="cuda", dtype=torch.int32),
        seqused_k=torch.ones(tokens, device="cuda", dtype=torch.int32),
        max_seqlen_q=1, max_seqlen_k=1,
        gather_kv_indices=idx,
        gather_kv_valid_length=valid if use_valid_length else None,
        softmax_scale=SOFTMAX_SCALE,
    )[0]
    torch.testing.assert_close(out, torch.ones_like(out), atol=0.002, rtol=0)
