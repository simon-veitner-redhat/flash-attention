"""Sparse-MLA (DSA) forward on the decode kernel: fewer than 128 query heads per KV head."""

import functools
import math

import pytest
import torch

import test_flash_attn_mla_sparse_flat_kv as flat_kv
from test_flash_attn_mla_sparse_flat_kv import IS_SM100, SOFTMAX_SCALE, TILE_N, TOPK

from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute.flash_fwd_mla_decode_h64_sm100 import (
    MIN_BLK_PER_SPLIT,
    FlashAttentionMLADecodeH64Sm100,
)

NUM_KV_ROWS = 4096
HEADS = [8, 16, 32, 64]
# one total_q per (G, S) class the split rule produces at 148 SMs, plus multi-wave grids
TOTAL_Q = [1, 9, 18, 37, 74, 148, 222, 300]
# (num_heads, mla_decode_h64): every head count on the default kernel, plus 64 heads on the
# heads-on-M kernel
HEAD_ROUTES = [(h, False) for h in HEADS] + [(64, True)]
ROUTES = [(16, False), (64, True)]

pytestmark = pytest.mark.skipif(not IS_SM100, reason="sparse MLA forward is SM100 only")


@pytest.fixture(autouse=True)
def decode_route(request, monkeypatch):
    """Assert every _flash_attn_fwd call in this module routes to the decode kernel, and to the
    64-head heads-on-M kernel exactly when the test sets h64 (parameter or `h64` marker).

    A test marked `skip_route_check` checks its routes itself, or makes no forward call."""
    if request.node.get_closest_marker("skip_route_check"):
        yield None
        return
    import flash_attn.cute.interface as interface

    # per mla_decode_splits call: True iff it took the heads-on-M route
    calls, forwards = [], []
    orig_splits = interface.mla_decode_splits

    def spy_splits(*a, **k):
        calls.append(k.get("mla_decode_h64", False))
        return orig_splits(*a, **k)

    orig_fwd = interface._flash_attn_fwd

    # functools.wraps copies __dict__, so the wrapper keeps the real `.compile_cache`.
    @functools.wraps(orig_fwd)
    def counting_fwd(*a, **k):
        forwards.append(None)
        return orig_fwd(*a, **k)

    monkeypatch.setattr(interface, "mla_decode_splits", spy_splits)
    monkeypatch.setattr(interface, "_flash_attn_fwd", counting_fwd)
    yield calls
    assert calls and len(calls) == len(forwards), (
        f"{len(forwards) - len(calls)} of {len(forwards)} call(s) did not route to "
        "the decode kernel"
    )
    callspec = getattr(request.node, "callspec", None)
    want_h64 = request.node.get_closest_marker("h64") is not None or bool(
        callspec is not None and callspec.params.get("h64", False)
    )
    assert all(h64 == want_h64 for h64 in calls), (
        f"expected every call {'on' if want_h64 else 'off'} the heads-on-M route: {calls}"
    )


def ragged_counts(total_q):
    """Ragged top-k lengths including exact-zero rows and both tile-boundary cases."""
    pattern = [TOPK, 0, 1, TILE_N, TILE_N + 1, TOPK - 1, 777, 0, 129, 1024]
    return (pattern * (total_q // len(pattern) + 1))[:total_q]


H64_EDGES = [0, 1, 63, 64, 65, 127, 128, 129, 2047, 2048]


def h64_counts(total_q, topk):
    """Ragged counts at 64-row block edges, clipped to the top-k width."""
    pattern = [min(c, topk) for c in H64_EDGES + [777, 1024]]
    return (pattern * (total_q // len(pattern) + 1))[:total_q]


def route_counts(total_q, h64):
    return h64_counts(total_q, TOPK) if h64 else ragged_counts(total_q)


def run_decode(q, qv, k, v, cu_seqlens_q, idx, valid_len, return_lse=True, h64=False):
    return flat_kv.run_kernel(
        q, qv, k, v, cu_seqlens_q, NUM_KV_ROWS, idx, valid_len, return_lse=return_lse,
        mla_decode_h64=h64,
    )


def assert_matches_ref(out, lse, q, qv, kv, idx, counts):
    """out and lse against the fp32 reference; the 16-bit torch reference sets the error scale."""
    out_ref, lse_ref = flat_kv.ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = flat_kv.ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)
    flat_kv.assert_close(out, lse, out_ref, lse_ref, out_pt, counts)


def assert_matches_ref_per_kv_head(out, lse, q, qv, kv, idx, counts):
    """assert_matches_ref for each KV head of kv and its query heads."""
    ratio = q.shape[1] // kv.shape[1]
    for head_k in range(kv.shape[1]):
        hs = slice(head_k * ratio, (head_k + 1) * ratio)
        kv_head = kv[:, head_k:head_k + 1].contiguous()
        assert_matches_ref(out[:, hs], lse[:, hs], q[:, hs], qv[:, hs], kv_head, idx, counts)


def check(
    q_lens, counts, num_heads=16, split_views=True, seed=0,
    dtype=torch.bfloat16, use_valid_length=True, pad=-1, h64=False, topk=TOPK,
    qscale=1.0,
):
    device = "cuda"
    cu_seqlens_q = flat_kv.cu_seqlens_from(q_lens, device)
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, dtype=dtype, seed=seed)
    q, qv = flat_kv.make_q(
        sum(q_lens), device, dtype=dtype, seed=seed + 1,
        split_views=split_views, num_heads=num_heads,
    )
    if qscale != 1.0:
        q.mul_(qscale)
        qv.mul_(qscale)
    idx, valid_len = flat_kv.make_indices(
        counts, NUM_KV_ROWS, device, seed=seed + 2, pad=pad, topk=topk,
    )

    out, lse = run_decode(
        q, qv, k, v, cu_seqlens_q, idx, valid_len if use_valid_length else None, h64=h64,
    )
    assert_matches_ref(out, lse, q, qv, kv, idx, counts)


@pytest.mark.parametrize("total_q", TOTAL_Q)
@pytest.mark.parametrize("num_heads", HEADS)
def test_decode_one_token_per_request(num_heads, total_q):
    """q_len == 1 per request, strided q/qv views: the shape vLLM's decode path calls with."""
    check([1] * total_q, ragged_counts(total_q), num_heads=num_heads, seed=10 * total_q)


@pytest.mark.parametrize("num_heads,h64", HEAD_ROUTES)
def test_decode_contiguous_q(num_heads, h64):
    check(
        [1] * 18, route_counts(18, h64), num_heads=num_heads, split_views=False, seed=31, h64=h64,
    )


@pytest.mark.parametrize("num_heads,h64", ROUTES)
def test_decode_varlen_q_len_gt_1(num_heads, h64):
    """Several query tokens per request, so cu_seqlens_q has fewer rows than total_q."""
    q_lens = [1, 5, 2, 8, 1, 3, 16, 4]
    check(q_lens, route_counts(sum(q_lens), h64), num_heads=num_heads, seed=41, h64=h64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads,h64", HEAD_ROUTES)
def test_decode_without_valid_length(dtype, num_heads, h64):
    # without the flag, 9 tokens at 64 heads split the head group (h = 32) and 148 run the
    # transposed h = 64 kernel; on the heads-on-M route 9 tokens run 16 KV splits
    total_q = 148 if num_heads == 64 and not h64 else 9
    check(
        [1] * total_q, route_counts(total_q, h64), num_heads=num_heads, dtype=dtype,
        use_valid_length=False, h64=h64,
    )


@pytest.mark.parametrize("num_heads,h64", ROUTES)
@pytest.mark.parametrize("pad", [-1, -2147483648, 1 << 20])
def test_decode_out_of_range_padding_sentinels(pad, num_heads, h64):
    """The gather forms a gmem pointer from the raw index, so any sentinel must be predicated."""
    check([1] * 9, route_counts(9, h64), num_heads=num_heads, pad=pad, seed=51, h64=h64)


def lse_inputs(total_q, num_heads, seed):
    device = "cuda"
    counts = ragged_counts(total_q)
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, seed=seed)
    q, qv = flat_kv.make_q(total_q, device, seed=seed + 1, num_heads=num_heads)
    idx, valid_len = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=seed + 2)
    cu_seqlens_q = flat_kv.cu_seqlens_from([1] * total_q, device)
    return (q, qv, k, v, cu_seqlens_q, idx, valid_len), kv, counts


# total_q 9 is KV-split (S == 8), 148 is not; same seed as test_decode_one_token_per_request
@pytest.mark.parametrize("total_q", [9, 148])
@pytest.mark.parametrize("num_heads,h64", HEAD_ROUTES)
def test_decode_lse_optional_and_output_unchanged(num_heads, h64, total_q):
    """Asking for the LSE must not perturb O, and not asking must return no LSE."""
    args, _, _ = lse_inputs(total_q, num_heads, seed=10 * total_q)
    out_no_lse, lse_none = run_decode(*args, return_lse=False, h64=h64)
    out_lse, lse = run_decode(*args, return_lse=True, h64=h64)
    assert lse_none is None
    assert lse.shape == (total_q, num_heads) and lse.dtype == torch.float32
    assert torch.equal(out_no_lse, out_lse), "the LSE store perturbed the output"


@pytest.mark.parametrize("total_q", [9, 148])
@pytest.mark.parametrize("num_heads", [16, 64])
def test_decode_lse_is_natural_log(num_heads, total_q):
    """The decode LSE is in natural log, like the 128-head kernel's."""
    args, kv, counts = lse_inputs(total_q, num_heads, seed=80 + total_q)
    q, qv, _, _, _, idx, _ = args
    _, lse = run_decode(*args)
    _, lse_ref_e = flat_kv.ref_sparse_mla(q, qv, kv, idx, counts)
    sel = torch.tensor([m for m, n in enumerate(counts) if n > 0], device=lse.device)
    lse_ref_2 = lse_ref_e / math.log(2.0)

    err_e = (lse[sel] - lse_ref_e[sel]).abs().max().item()
    err_2 = (lse[sel] - lse_ref_2[sel]).abs().max().item()
    assert err_e <= 5e-3, f"lse is not natural log: max diff vs ln {err_e}, vs log2 {err_2}"
    assert err_2 > 1.0, "lse could not be distinguished from log2; test inputs are degenerate"


@pytest.mark.parametrize("total_q", [9, 148])
def test_decode_preallocated_lse(total_q):
    """A caller-supplied lse buffer is filled by the decode path (the `lse is None` gate)."""
    from flash_attn.cute.interface import _flash_attn_fwd

    (q, qv, k, v, cu_seqlens_q, idx, valid_len), _, _ = lse_inputs(total_q, 16, seed=100 + total_q)
    lse_buf = torch.full((total_q, 16), 1234.0, device=q.device, dtype=torch.float32)
    out, lse = _flash_attn_fwd(
        q, k, v, qv=qv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=torch.zeros(total_q + 1, device=q.device, dtype=torch.int32),
        seqused_k=torch.full((total_q,), NUM_KV_ROWS, device=q.device, dtype=torch.int32),
        max_seqlen_q=1, max_seqlen_k=NUM_KV_ROWS,
        gather_kv_indices=idx, gather_kv_valid_length=valid_len,
        softmax_scale=SOFTMAX_SCALE,
        lse=lse_buf,
    )[:2]
    assert lse is lse_buf, "the decode path replaced the caller-supplied lse buffer"
    assert (lse_buf != 1234.0).any(), "caller-supplied lse buffer was not written"
    out_ref, lse_ref = run_decode(q, qv, k, v, cu_seqlens_q, idx, valid_len)
    torch.testing.assert_close(lse_buf, lse_ref, atol=0, rtol=0)
    torch.testing.assert_close(out, out_ref, atol=0, rtol=0)


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
        # unsplit grids (one n-block): this is the LSE check of the 2-D store at head_base != 0
        out, lse = flash_attn_varlen_func(
            q, kv[..., 512:], kv[..., :512], qv=qv,
            cu_seqlens_q=cuq, cu_seqlens_k=cuk,
            max_seqlen_q=1, max_seqlen_k=rows_per_req,
            gather_kv_indices=idx, gather_kv_valid_length=valid,
            softmax_scale=SOFTMAX_SCALE,
            return_lse=True,
        )
        for row in [0, 1, 2, 3, 147]:
            for head_k in range(heads_k):
                head_slice = slice(head_k * head_ratio, (head_k + 1) * head_ratio)
                kv_row = kv[row * rows_per_req:(row + 1) * rows_per_req, head_k:head_k + 1]
                ref, lse_ref = flat_kv.ref_sparse_mla(
                    q[row:row + 1, head_slice], qv[row:row + 1, head_slice],
                    kv_row, idx[row:row + 1], counts[row:row + 1],
                )
                torch.testing.assert_close(
                    out[row:row + 1, head_slice], ref, atol=0.015, rtol=0.015,
                )
                # counts[1] == 0: -inf lse and zero output on the empty row.
                torch.testing.assert_close(
                    lse[row:row + 1, head_slice], lse_ref, atol=5e-3, rtol=0,
                )


@pytest.mark.parametrize("h64", [False, True])
@pytest.mark.parametrize("total_q", [18, 148])
def test_decode_two_kv_heads_64_ratio(total_q, h64):
    """64 query heads per KV head at head_kv = 1: the second head block of Q, O and the LSE."""
    device = "cuda"
    heads_k, ratio = 2, 64
    counts = route_counts(total_q, h64)
    idx, valid = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=3)
    q, qv = flat_kv.make_q(total_q, device, num_heads=heads_k * ratio, seed=4)
    gen = torch.Generator(device=device).manual_seed(5)
    kv = torch.randn(NUM_KV_ROWS, heads_k, 576, device=device, generator=gen).to(q.dtype)
    out, lse = run_decode(
        q, qv, kv[..., 512:], kv[..., :512], flat_kv.cu_seqlens_from([1] * total_q, device),
        idx, valid, h64=h64,
    )
    assert_matches_ref_per_kv_head(out, lse, q, qv, kv, idx, counts)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads,h64", HEAD_ROUTES)
# (37, 2048) splits the topk at 64 heads: split 0 opens on the empty block and the later
# splits see only masked blocks.  On the heads-on-M route 37 tokens run 4 KV splits and 1 token
# 16: at "second_block" with a valid length (4 blocks of 64) splits 0 and 1 are all masked, and at
# "last_row" every split but the last is.
@pytest.mark.parametrize("tokens,topk", [(1, 2048), (37, 2048), (148, 256)])
@pytest.mark.parametrize("use_valid_length", [False, True])
# the second 128-row block, or only the last row: every earlier block is masked
@pytest.mark.parametrize("valid_rows", ["second_block", "last_row"])
def test_decode_leading_empty_block(
    dtype, num_heads, h64, tokens, topk, use_valid_length, valid_rows
):
    """Leading empty blocks must not clamp subsequent negative score maxima to zero."""
    rows = slice(TILE_N, 2 * TILE_N) if valid_rows == "second_block" else slice(topk - 1, topk)
    q = torch.zeros(tokens, num_heads, 64, device="cuda", dtype=dtype)
    qv = torch.full((tokens, num_heads, 512), -10, device="cuda", dtype=dtype)
    k = torch.zeros(1, 1, 64, device="cuda", dtype=dtype)
    v = torch.ones(1, 1, 512, device="cuda", dtype=dtype)
    idx = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
    idx[:, rows] = 0
    valid = torch.full((tokens,), rows.stop, device="cuda", dtype=torch.int32)
    out, lse = flash_attn_varlen_func(
        q, k, v, qv=qv,
        cu_seqlens_q=torch.arange(tokens + 1, device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.zeros(tokens + 1, device="cuda", dtype=torch.int32),
        seqused_k=torch.ones(tokens, device="cuda", dtype=torch.int32),
        max_seqlen_q=1, max_seqlen_k=1,
        gather_kv_indices=idx,
        gather_kv_valid_length=valid if use_valid_length else None,
        softmax_scale=SOFTMAX_SCALE,
        return_lse=True,
        mla_decode_h64=h64,
    )
    torch.testing.assert_close(out, torch.ones_like(out), atol=0.002, rtol=0)
    s = -10 * 512 * SOFTMAX_SCALE
    expect = torch.full_like(lse, s + math.log(rows.stop - rows.start))
    torch.testing.assert_close(lse, expect, atol=5e-3 * abs(s), rtol=0)


@pytest.mark.parametrize("num_heads,h64", ROUTES)
@pytest.mark.parametrize("use_valid_length", [True, False])
@pytest.mark.parametrize("use_seqused", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_batched_kv_scattered_mask(dtype, use_seqused, use_valid_length, num_heads, h64):
    """Per-request KV offsets and lengths, real rows past seqlen_k, and invalid entries (-1,
    INT_MIN, seqlen_k and past it) scattered inside the valid length; empty requests and
    q_len > 1."""
    device = "cuda"
    gen = torch.Generator().manual_seed(0)
    q_lens = [1, 0, 3, 1, 1, 0, 2, 1, 5, 1, 1, 1]
    kv_lens = [int(torch.randint(100, 3000, (1,), generator=gen)) for _ in q_lens]
    # 500 real rows past each request's seqlen_k: a wrong seqlen_k mask attends them
    span = [n + 500 for n in kv_lens]
    offs = [0]
    for n in span:
        offs.append(offs[-1] + n)
    if not use_seqused:
        kv_lens = span  # cu_seqlens_k alone: seqlen_k is the span
    kv, k, v = flat_kv.make_flat_kv(offs[-1], device, dtype=dtype, seed=10)
    total_q = sum(q_lens)
    q, qv = flat_kv.make_q(total_q, device, dtype=dtype, seed=20, num_heads=num_heads)
    idx = torch.full((total_q, TOPK), -1, dtype=torch.int32)
    valid = torch.zeros(total_q, dtype=torch.int32)
    # the reference attends the valid entries as global rows, packed to the front
    idx_ref = torch.zeros(total_q, TOPK, dtype=torch.int32)
    counts = []
    m = 0
    for b, q_len in enumerate(q_lens):
        for _ in range(q_len):
            c = 0 if m % 7 == 3 else int(torch.randint(0, TOPK + 1, (1,), generator=gen))
            ent = torch.randint(0, kv_lens[b], (c,), generator=gen, dtype=torch.int32)
            bad = torch.rand(c, generator=gen) < 0.35
            kind = torch.randint(0, 4, (c,), generator=gen)
            bad_val = torch.where(
                kind == 0, -1,
                torch.where(kind == 1, -2147483648,
                            torch.where(kind == 2, kv_lens[b] + ent % 400, kv_lens[b])),
            ).to(torch.int32)
            ent = torch.where(bad, bad_val, ent)
            idx[m, :c] = ent
            valid[m] = c
            good = ent[(ent >= 0) & (ent < kv_lens[b])] + offs[b]
            idx_ref[m, :good.numel()] = good
            counts.append(good.numel())
            m += 1
    idx, valid, idx_ref = idx.to(device), valid.to(device), idx_ref.to(device)
    out, lse = flash_attn_varlen_func(
        q, k, v, qv=qv,
        cu_seqlens_q=flat_kv.cu_seqlens_from(q_lens, device),
        cu_seqlens_k=torch.tensor(offs, device=device, dtype=torch.int32),
        seqused_k=(
            torch.tensor(kv_lens, device=device, dtype=torch.int32) if use_seqused else None
        ),
        max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
        gather_kv_indices=idx,
        gather_kv_valid_length=valid if use_valid_length else None,
        softmax_scale=SOFTMAX_SCALE,
        return_lse=True,
        mla_decode_h64=h64,
    )
    assert_matches_ref(out, lse, q, qv, kv, idx_ref, counts)


# ---------------------------------------------------------------------------------------------
# cases only the heads-on-M 64-head kernel has (FlashAttentionMLADecodeH64Sm100, opt-in:
# mla_decode_h64): 64-row blocks, its KV splits, its lazy rescale, its index masks


@pytest.fixture
def force_h64_splits(monkeypatch):
    """force(S): the next decode calls take grid (1, S).  The call still goes through the
    decode_route fixture's spy, so the route check sees it."""
    import flash_attn.cute.interface as interface

    def force(splits):
        spy = interface.mla_decode_splits

        def forced_splits(*args, **kwargs):
            spy(*args, **kwargs)  # the route check still sees the call
            return 1, splits

        monkeypatch.setattr(interface, "mla_decode_splits", forced_splits)

    return force


@pytest.mark.h64
@pytest.mark.parametrize("topk", [128, 256, 512, 2048])
@pytest.mark.parametrize("total_q", [1, 9, 40, 148, 300])
def test_decode_h64_single_split(total_q, topk, force_h64_splits):
    """Valid counts at the 64-row block edges, one split."""
    force_h64_splits(1)
    check(
        [1] * total_q, h64_counts(total_q, topk), num_heads=64, h64=True, topk=topk,
        seed=7 * total_q + topk,
    )


@pytest.mark.h64
@pytest.mark.parametrize("splits", [None, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_h64_all_block_counts_deterministic(dtype, splits, force_h64_splits):
    """Every block count 1..32 with edge remainders; repeated launches are bitwise identical
    (automatic pick: unsplit; 2-4 splits: the in-kernel combine)."""
    if splits is not None:
        force_h64_splits(splits)
    device = "cuda"
    counts = [n * 64 - r for n in range(1, 33) for r in (0, 1, 31, 63)] + [0, 1, TOPK]
    total_q = len(counts)
    num_rows = 8192
    kv, k, v = flat_kv.make_flat_kv(num_rows, device, dtype=dtype, seed=3)
    q, qv = flat_kv.make_q(total_q, device, dtype=dtype, seed=4, num_heads=64)
    idx, valid = flat_kv.make_indices(counts, num_rows, device, seed=5)
    cu_seqlens_q = flat_kv.cu_seqlens_from([1] * total_q, device)
    args = (q, qv, k, v, cu_seqlens_q, num_rows, idx, valid)
    out, lse = flat_kv.run_kernel(*args, mla_decode_h64=True)
    assert_matches_ref(out, lse, q, qv, kv, idx, counts)
    for _ in range(20):
        out2, lse2 = flat_kv.run_kernel(*args, mla_decode_h64=True)
        assert torch.equal(out2, out) and torch.equal(lse2, lse), "non-deterministic output"


@pytest.mark.h64
def test_decode_h64_all_but_last_block_masked_by_index():
    """Every block except the last is masked by index only (valid length covers all of them)."""
    device = "cuda"
    total_q = 9
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, seed=60)
    q, qv = flat_kv.make_q(total_q, device, seed=61, num_heads=64)
    idx = torch.full((total_q, TOPK), -1, device=device, dtype=torch.int32)
    gen = torch.Generator(device=device).manual_seed(62)
    rows = torch.randperm(NUM_KV_ROWS, device=device, generator=gen)[:64]
    idx[:, TOPK - 64:] = rows.to(torch.int32)
    valid = torch.full((total_q,), TOPK, device=device, dtype=torch.int32)
    out, lse = run_decode(
        q, qv, k, v, flat_kv.cu_seqlens_from([1] * total_q, device), idx, valid, h64=True,
    )
    # the reference attends the leading valid rows: move the valid block to the front
    idx_ref = idx.roll(64, dims=1)
    counts = [64] * total_q
    assert_matches_ref(out, lse, q, qv, kv, idx_ref, counts)


@pytest.mark.h64
def test_decode_h64_peaky_logits():
    """q x 30: large logits, the lazy rescale path with scale factors far from 1."""
    check([1] * 40, h64_counts(40, TOPK), num_heads=64, h64=True, qscale=30.0, seed=70)


@pytest.mark.h64
@pytest.mark.parametrize("splits", [1, None])
@pytest.mark.parametrize("late_block", [1, 5, 31])
@pytest.mark.parametrize("delta", [5.9, 6.0, 6.1, 12.0, 60.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_h64_late_max_growth(dtype, delta, late_block, splits, force_h64_splits):
    """Explicit logits: the first block of every split at L0, one late block at L0 + delta (log2
    units), everything else near 0.  Around the lazy-rescale threshold (6) and far above it, O
    must be rescaled exactly when needed, in every split.  splits None takes the automatic pick
    (16 splits of 2 blocks at 3 tokens: the late block is block 1 of its split)."""
    device = "cuda"
    total_q, num_rows, l0 = 3, 8192, 3.0
    if splits is not None:
        force_h64_splits(splits)
    else:
        splits = 16  # the automatic pick at 3 tokens (test_decode_h64_split_picks)
    blocks_per_split = TOPK // 64 // splits
    assert late_block % blocks_per_split != 0, "the late block must not open its split"
    gen = torch.Generator(device=device).manual_seed(11)
    kv = torch.randn(num_rows, 1, 576, device=device, generator=gen) * 0.05
    q = torch.zeros(total_q, 64, 64, device=device)
    qv = torch.zeros(total_q, 64, 512, device=device)
    qv[..., 0] = 1.0
    perm = torch.randperm(num_rows, device=device, generator=gen)
    idx = perm[: total_q * TOPK].view(total_q, TOPK).int()
    # logit (log2 units) = kv[row, 0, 0] * SOFTMAX_SCALE * log2(e)
    to_kv = 1.0 / (SOFTMAX_SCALE * math.log2(math.e))
    for m in range(total_q):
        for b in range(0, TOPK // 64, blocks_per_split):
            kv[idx[m, b * 64:(b + 1) * 64].long(), 0, 0] = l0 * to_kv
        kv[idx[m, late_block * 64:(late_block + 1) * 64].long(), 0, 0] = (l0 + delta) * to_kv
    kv, q, qv = kv.to(dtype), q.to(dtype), qv.to(dtype)
    # the construction: the running max jumps by delta (to 16-bit rounding) at late_block
    logits = kv[idx.long(), 0, 0].float() * SOFTMAX_SCALE * math.log2(math.e)
    blk_max = logits.view(total_q, TOPK // 64, 64).amax(-1)
    jump = blk_max[:, late_block] - blk_max[:, :late_block].amax(-1)
    assert torch.allclose(jump, torch.full_like(jump, delta), atol=0.02 * delta), jump
    valid = torch.full((total_q,), TOPK, device=device, dtype=torch.int32)
    out, lse = flat_kv.run_kernel(
        q, qv, kv[..., 512:], kv[..., :512], flat_kv.cu_seqlens_from([1] * total_q, device),
        num_rows, idx, valid, mla_decode_h64=True,
    )
    counts = [TOPK] * total_q
    assert_matches_ref(out, lse, q, qv, kv, idx, counts)


def h64_split_counts(total_q, splits, topk=TOPK):
    """64-row edges, rows that fall in split 0 only, and rows that end exactly on a split edge."""
    edges = [64 * splits * k for k in (1, 2, 3)] + [64 * splits + 1, 64 * (splits + 1) - 1]
    pattern = [min(c, topk) for c in H64_EDGES + edges + [777]]
    return (pattern * (total_q // len(pattern) + 1))[:total_q]


@pytest.mark.h64
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("total_q", [1, 9, 37, 45])
@pytest.mark.parametrize("splits", [1, 2, 3, 4, 8, 16])
def test_decode_h64_forced_splits(splits, total_q, dtype, force_h64_splits):
    force_h64_splits(splits)
    counts = h64_split_counts(total_q, splits)
    if total_q == 1:
        counts = [TOPK - 1]
    check(
        [1] * total_q, counts, num_heads=64, h64=True, dtype=dtype, seed=3 * total_q + splits,
    )


@pytest.mark.h64
@pytest.mark.parametrize("use_valid_length", [True, False])
@pytest.mark.parametrize("splits", [2, 3, 4, 16])
def test_decode_h64_forced_splits_no_lse_views(splits, use_valid_length, force_h64_splits):
    """Split grids without the LSE (the partial LSE is still written for flash_fwd_combine, and
    kept in smem for the in-kernel combine), with contiguous q and without a valid length."""
    force_h64_splits(splits)
    total_q = 9
    counts = h64_split_counts(total_q, splits) if use_valid_length else [TOPK] * total_q
    device = "cuda"
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, seed=90)
    q, qv = flat_kv.make_q(total_q, device, seed=91, num_heads=64, split_views=False)
    idx, valid = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=92)
    cu_q = flat_kv.cu_seqlens_from([1] * total_q, device)
    valid = valid if use_valid_length else None
    out, lse_none = run_decode(q, qv, k, v, cu_q, idx, valid, return_lse=False, h64=True)
    out_lse, lse = run_decode(q, qv, k, v, cu_q, idx, valid, return_lse=True, h64=True)
    assert lse_none is None
    assert torch.equal(out, out_lse), "the LSE store perturbed the output"
    assert_matches_ref(out, lse, q, qv, kv, idx, counts)


@pytest.mark.h64
@pytest.mark.parametrize("real,padded", [(2, 4), (8, 16), (32, 40), (64, 74), (256, 272)])
def test_decode_h64_dp_padding(real, padded):
    """DP pads the per-rank batch with valid-0 rows; the automatic rule sees the padded count.
    Padded rows must come out exactly 0 / -inf after the combine."""
    counts = h64_counts(real, TOPK) + [0] * (padded - real)
    check([1] * padded, counts, num_heads=64, h64=True, seed=real)


@pytest.mark.h64
@pytest.mark.parametrize("total_q", [4, 40, 148])
def test_decode_h64_all_rows_empty(total_q):
    """Every row has valid length 0: 16 splits (4 tokens), the in-kernel combine (40 tokens: 3
    splits) and unsplit (148)."""
    check([1] * total_q, [0] * total_q, num_heads=64, h64=True, seed=total_q)


@pytest.mark.h64
@pytest.mark.parametrize("total_q", [16, 40, 272])
def test_decode_h64_random_valid_lengths(total_q):
    gen = torch.Generator().manual_seed(total_q)
    counts = torch.randint(0, TOPK + 1, (total_q,), generator=gen).tolist()
    check([1] * total_q, counts, num_heads=64, h64=True, seed=total_q + 1)


@pytest.mark.h64
@pytest.mark.parametrize("masked_split", [0, 1])
def test_decode_h64_split_masked_by_index(masked_split, force_h64_splits):
    """Two splits over the full top-k: one split's entries are all -1 (masked by index only,
    the valid length covers them), the other split is valid."""
    force_h64_splits(2)
    device = "cuda"
    total_q = 9
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, seed=95)
    q, qv = flat_kv.make_q(total_q, device, seed=96, num_heads=64)
    gen = torch.Generator(device=device).manual_seed(97)
    half = TOPK // 2
    idx = torch.full((total_q, TOPK), -1, device=device, dtype=torch.int32)
    live = slice(half, TOPK) if masked_split == 0 else slice(0, half)
    for m in range(total_q):
        idx[m, live] = torch.randperm(NUM_KV_ROWS, device=device, generator=gen)[:half].int()
    valid = torch.full((total_q,), TOPK, device=device, dtype=torch.int32)
    out, lse = run_decode(
        q, qv, k, v, flat_kv.cu_seqlens_from([1] * total_q, device), idx, valid, h64=True,
    )
    # the reference attends the leading rows: the live half only
    assert_matches_ref(out, lse, q, qv, kv, idx[:, live], [half] * total_q)


def h64_cluster_counts(splits):
    """Valid lengths that give every non-empty split count S_ne = 0..S of the in-kernel combine
    (per = max(ceil(n / S), MIN_BLK_PER_SPLIT) blocks, S_ne = ceil(n / per)), each block count
    once ending on a 64-row edge and once one row short of it."""
    counts, s_ne = [], set()
    for n in range(0, TOPK // 64 + 1):
        per = max(-(-n // splits), MIN_BLK_PER_SPLIT)
        s_ne.add(-(-n // per))
        counts += [64 * n] + ([64 * n - 1] if n else [])
    assert s_ne == set(range(splits + 1)), s_ne
    return counts


@pytest.mark.h64
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits", [2, 3, 4])
def test_decode_h64_cluster_all_split_counts(splits, dtype, force_h64_splits):
    """The in-kernel combine at every non-empty split count 0..S and every block count (on and
    one row short of a 64-row edge, which covers the rows that end on a split edge), with and
    without the LSE."""
    force_h64_splits(splits)
    counts = h64_cluster_counts(splits)
    total_q = len(counts)
    device = "cuda"
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, dtype=dtype, seed=110 + splits)
    q, qv = flat_kv.make_q(total_q, device, dtype=dtype, seed=111, num_heads=64)
    idx, valid = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=112)
    args = (q, qv, k, v, flat_kv.cu_seqlens_from([1] * total_q, device), idx, valid)
    out, lse = run_decode(*args, h64=True)
    assert_matches_ref(out, lse, q, qv, kv, idx, counts)
    out_no_lse, lse_none = run_decode(*args, return_lse=False, h64=True)
    assert lse_none is None
    assert torch.equal(out_no_lse, out), "the LSE store perturbed the output"


@pytest.mark.h64
@pytest.mark.parametrize("splits", [2, 3, 4])
def test_decode_h64_cluster_two_kv_heads_varlen(splits, force_h64_splits):
    """The in-kernel combine with 2 KV heads (grid y) and q_len > 1 (cu_seqlens_q shorter than
    total_q)."""
    force_h64_splits(splits)
    device = "cuda"
    heads_k, ratio = 2, 64
    q_lens = [1, 5, 2, 8, 1, 3, 16, 4]
    total_q = sum(q_lens)
    counts = h64_split_counts(total_q, splits)
    idx, valid = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=120)
    q, qv = flat_kv.make_q(total_q, device, num_heads=heads_k * ratio, seed=121)
    gen = torch.Generator(device=device).manual_seed(122)
    kv = torch.randn(NUM_KV_ROWS, heads_k, 576, device=device, generator=gen).to(q.dtype)
    out, lse = run_decode(
        q, qv, kv[..., 512:], kv[..., :512], flat_kv.cu_seqlens_from(q_lens, device),
        idx, valid, h64=True,
    )
    assert_matches_ref_per_kv_head(out, lse, q, qv, kv, idx, counts)


@pytest.mark.h64
@pytest.mark.parametrize("splits", [2, 3, 4])
def test_decode_h64_cluster_cuda_graph(splits, force_h64_splits):
    """One in-kernel-combine call captured in a CUDA graph and replayed twice with new q, indices
    and valid lengths: each replay equals an eager call (nothing persists across launches)."""
    force_h64_splits(splits)
    device = "cuda"
    total_q = 37
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, seed=130)
    cu_q = flat_kv.cu_seqlens_from([1] * total_q, device)

    def inputs(seed):
        q, qv = flat_kv.make_q(total_q, device, seed=seed, num_heads=64)
        gen = torch.Generator().manual_seed(seed)
        counts = torch.randint(0, TOPK + 1, (total_q,), generator=gen).tolist()
        counts[seed % total_q] = 0
        idx, valid = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=seed + 1)
        return q, qv, idx, valid

    cu_k = torch.zeros(total_q + 1, device=device, dtype=torch.int32)
    used_k = torch.full((total_q,), NUM_KV_ROWS, device=device, dtype=torch.int32)

    def call(q, qv, idx, valid):
        # run_decode's max_seqlen_q syncs the host, which a capture forbids
        return flash_attn_varlen_func(
            q, k, v, qv=qv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, seqused_k=used_k,
            max_seqlen_q=1, max_seqlen_k=NUM_KV_ROWS, gather_kv_indices=idx,
            gather_kv_valid_length=valid, softmax_scale=SOFTMAX_SCALE, return_lse=True,
            mla_decode_h64=True,
        )

    q_s, qv_s, idx_s, valid_s = inputs(131)
    call(q_s, qv_s, idx_s, valid_s)  # compile (and query the cluster slots) outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g, lse_g = call(q_s, qv_s, idx_s, valid_s)
    for seed in (140, 150):
        q, qv, idx, valid = inputs(seed)
        for dst, src in ((q_s, q), (qv_s, qv), (idx_s, idx), (valid_s, valid)):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        out_e, lse_e = call(q, qv, idx, valid)
        assert torch.equal(out_g, out_e) and torch.equal(lse_g, lse_e), seed


@pytest.mark.h64
@pytest.mark.parametrize("splits", [2, 3, 4, 8])
def test_decode_h64_nan_split_dropped(splits, force_h64_splits):
    """A NaN key makes its split's li NaN (an empty split, LSE -inf).  flash_fwd_combine (8
    splits) skips such a split; the in-kernel combine (2-4) must too: the output is that of the
    other splits' keys, not NaN."""
    force_h64_splits(splits)
    device = "cuda"
    n_blk = TOPK // 64
    per = -(-n_blk // splits)
    if splits <= 4:
        per = max(per, MIN_BLK_PER_SPLIT)
    kv, k, v = flat_kv.make_flat_kv(NUM_KV_ROWS, device, seed=160)
    q, qv = flat_kv.make_q(1, device, seed=161, num_heads=64)
    idx, valid = flat_kv.make_indices([TOPK], NUM_KV_ROWS, device, seed=162)
    lo, hi = 64 * per, 128 * per  # split 1
    kv[idx[0, lo].long(), 0, 512] = float("nan")
    out, lse = run_decode(q, qv, k, v, flat_kv.cu_seqlens_from([1], device), idx, valid, h64=True)
    kept = torch.cat([idx[:, :lo], idx[:, hi:]], dim=1)
    counts = [TOPK - (hi - lo)]
    assert_matches_ref(out, lse, q, qv, kv, kept, counts)


@pytest.mark.skip_route_check
@pytest.mark.parametrize("splits", [1, 2, 3, 4, 8, 16])
def test_decode_h64_launch_dims(splits):
    """The launch the kernel makes: grid (total_q * S, h_kv), and clusters of S CTAs along x
    exactly for the in-kernel combine, so cluster rank == blockIdx.x % S == the split index."""
    kernel = FlashAttentionMLADecodeH64Sm100(num_splits=splits)
    grid, cluster = kernel.launch_dims(45, 2)
    assert grid == (45 * splits, 2, 1)
    assert cluster == ((splits, 1, 1) if 2 <= splits <= 4 else None)


@pytest.mark.skip_route_check
def test_decode_h64_flag_selects_kernel_class(monkeypatch):
    """With the flag, 64 heads construct FlashAttentionMLADecodeH64Sm100; without it, the
    transposed FlashAttentionMLADecodeSm100."""
    import flash_attn.cute.interface as interface

    built = []

    def recording(cls):
        class Recording(cls):
            def __init__(self, *args, **kwargs):
                built.append(cls.__name__)
                super().__init__(*args, **kwargs)
        return Recording

    for name in ("FlashAttentionMLADecodeSm100", "FlashAttentionMLADecodeH64Sm100"):
        monkeypatch.setattr(interface, name, recording(getattr(interface, name)))
    interface._flash_attn_fwd.compile_cache.clear()
    args, _, _ = lse_inputs(148, 64, seed=5)
    run_decode(*args, h64=True)
    run_decode(*args, h64=False)
    assert built == ["FlashAttentionMLADecodeH64Sm100", "FlashAttentionMLADecodeSm100"], built


# mla_decode_splits(total_q, 1, h, 148, 16) without the flag, as the transposed kernel picks it
# (h = 64 included): (first total_q, (head groups, splits)) runs.
FROZEN_PICKS = {
    8: [(1, (1, 8)), (19, (1, 4)), (38, (1, 2)), (75, (1, 1)), (149, (1, 2)), (223, (1, 1))],
    16: [(1, (2, 8)), (10, (1, 8)), (19, (1, 4)), (38, (1, 2)), (75, (1, 1)), (149, (1, 2)),
         (223, (1, 1))],
    32: [(1, (4, 8)), (5, (2, 8)), (10, (1, 8)), (19, (1, 4)), (38, (1, 2)), (75, (1, 1)),
         (149, (1, 2)), (223, (1, 1))],
    64: [(1, (8, 8)), (3, (4, 8)), (5, (2, 8)), (10, (1, 8)), (19, (1, 4)), (38, (1, 2)),
         (75, (1, 1)), (149, (1, 2)), (223, (1, 1))],
}


@pytest.mark.skip_route_check
@pytest.mark.parametrize("h", [8, 16, 32, 64])
def test_decode_split_picks_frozen(h):
    from flash_attn.cute.interface import mla_decode_splits

    runs = FROZEN_PICKS[h]
    for total_q in range(1, 301):
        want = [r for start, r in runs if start <= total_q][-1]
        assert mla_decode_splits(total_q, 1, h, 148, 16) == want, (h, total_q)


# mla_decode_splits(B, 1, 64, 148, n_blocks, mla_decode_h64=True) with the B300 cluster slots
# (1 CTA per SM: 74 clusters of 2, 45 of 3, 33 of 4): 16 or 8 splits while the grid fits one
# wave (2 blocks of 64 rows per split), else 4, 3 or 2 while every cluster is resident (4 blocks
# per split), else 1.  (first B, splits) runs per top-k.
B300_SLOTS = {2: 74, 3: 45, 4: 33}
H64_PICKS = {
    2048: [(1, 16), (10, 8), (19, 4), (34, 3), (46, 2), (75, 1)],
    1024: [(1, 8), (19, 4), (34, 3), (46, 2), (75, 1)],
    512: [(1, 2), (75, 1)],
    256: [(1, 1)],
    128: [(1, 1)],
}


@pytest.mark.skip_route_check
def test_decode_h64_split_picks():
    from flash_attn.cute.interface import mla_decode_splits

    for topk, runs in H64_PICKS.items():
        for total_q in range(1, 301):
            want = (1, [s for start, s in runs if start <= total_q][-1])
            got = mla_decode_splits(
                total_q, 1, 64, 148, topk // 128, mla_decode_h64=True,
                cluster_slots=B300_SLOTS.get,
            )
            assert got == want, (topk, total_q, got)
    # the DP-padded batch sizes of the E2E matrix: 32:40 runs 3 splits (an unpadded 40 cannot be
    # told apart and 4-CTA clusters fit only 33 rows)
    assert mla_decode_splits(40, 1, 64, 148, 16, mla_decode_h64=True,
                             cluster_slots=B300_SLOTS.get) == (1, 3)
    # two KV heads count as two rows
    assert mla_decode_splits(10, 2, 64, 148, 16, mla_decode_h64=True,
                             cluster_slots=B300_SLOTS.get) == (1, 4)
    # no slot query (None, or a query that fails): 0.85 * num_SMs / S slots (31 / 41 / 62)
    for slots in (None, lambda s: None):
        picks = [mla_decode_splits(b, 1, 64, 148, 16, mla_decode_h64=True, cluster_slots=slots)
                 for b in (19, 31, 32, 41, 42, 62, 63)]
        assert picks == [(1, 4), (1, 4), (1, 3), (1, 3), (1, 2), (1, 2), (1, 1)], picks
    # a query above one wave of clusters is capped at num_SMs / S (37 / 49 / 74)
    picks = [mla_decode_splits(b, 1, 64, 148, 16, mla_decode_h64=True, cluster_slots=lambda s: 999)
             for b in (37, 38, 49, 50, 74, 75)]
    assert picks == [(1, 4), (1, 3), (1, 3), (1, 2), (1, 2), (1, 1)], picks


@pytest.mark.skip_route_check
def test_decode_h64_cluster_slots_gate(monkeypatch):
    """The slot query runs unless FLASH_ATTENTION_NUM_SMS overrides the SM count; then
    mla_decode_splits gets None and uses the estimate.  A local GPU of another arch never gets
    here: get_num_sms_for_selection raises first."""
    from flash_attn.cute.cute_dsl_utils import (
        _get_device_arch_and_num_sms,
        get_num_sms_for_selection,
    )
    from flash_attn.cute.interface import _h64_cluster_slots_fn

    arch = _get_device_arch_and_num_sms(0)[0]
    monkeypatch.delenv("FLASH_ATTENTION_NUM_SMS", raising=False)
    assert _h64_cluster_slots_fn(0)(4) is not None
    with pytest.raises(RuntimeError):
        get_num_sms_for_selection(0, arch + 1)
    monkeypatch.setenv("FLASH_ATTENTION_NUM_SMS", "148")
    assert _h64_cluster_slots_fn(0) is None


@pytest.mark.skip_route_check
def test_decode_h64_cluster_slots_query():
    """The device query gives a positive, non-increasing slot count per cluster size, within one
    wave of 1-CTA clusters (B300: 74 / 45 / 33)."""
    from flash_attn.cute.interface import _h64_cluster_slots

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    slots = [_h64_cluster_slots(0, s) for s in (2, 3, 4)]
    assert all(x is not None and x > 0 for x in slots), slots
    assert slots[0] >= slots[1] >= slots[2], slots
    assert all(x * s <= num_sms for x, s in zip(slots, (2, 3, 4))), (slots, num_sms)
