"""Sparse-MLA (DSA) forward on the decode kernel: fewer than 128 query heads per KV head."""

import functools
import math

import pytest
import torch

import test_flash_attn_mla_sparse_flat_kv as flat_kv
from test_flash_attn_mla_sparse_flat_kv import IS_SM100, SOFTMAX_SCALE, TILE_N, TOPK

from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute.interface import mla_decode_splits

NUM_KV_ROWS = 4096
HEADS = [8, 16, 32, 64]
# one total_q per (G, S) class the split rule produces at 148 SMs, plus multi-wave grids
TOTAL_Q = [1, 2, 4, 9, 18, 37, 74, 148, 222, 300]

pytestmark = pytest.mark.skipif(not IS_SM100, reason="sparse MLA forward is SM100 only")


@pytest.fixture(autouse=True)
def decode_route(monkeypatch):
    """Assert every _flash_attn_fwd call in this module routes to the decode kernel."""
    import flash_attn.cute.interface as interface

    calls, forwards = [], []
    orig_splits = interface.mla_decode_splits
    orig_fwd = interface._flash_attn_fwd

    # functools.wraps copies __dict__, so the wrapper keeps the real `.compile_cache`.
    @functools.wraps(orig_fwd)
    def counting_fwd(*a, **k):
        forwards.append(None)
        return orig_fwd(*a, **k)

    monkeypatch.setattr(
        interface, "mla_decode_splits", lambda *a, **k: (calls.append(a) or orig_splits(*a, **k))
    )
    monkeypatch.setattr(interface, "_flash_attn_fwd", counting_fwd)
    yield calls
    assert calls and len(calls) == len(forwards), (
        f"{len(forwards) - len(calls)} of {len(forwards)} call(s) did not route to "
        "FlashAttentionMLADecodeSm100"
    )


def ragged_counts(total_q):
    """Ragged top-k lengths including exact-zero rows and both tile-boundary cases."""
    pattern = [TOPK, 0, 1, TILE_N, TILE_N + 1, TOPK - 1, 777, 0, 129, 1024]
    return (pattern * (total_q // len(pattern) + 1))[:total_q]


def run_decode(q, qv, k, v, cu_seqlens_q, idx, valid_len, return_lse=True):
    return flat_kv.run_kernel(
        q, qv, k, v, cu_seqlens_q, NUM_KV_ROWS, idx, valid_len, return_lse=return_lse
    )


def to_fp8(fp8, q, qv, kv):
    """Kernel inputs (e4m3 q/qv/K/V if fp8), plus the 16-bit references they upcast to exactly."""
    if fp8:
        q, qv, kv = [t.to(torch.float8_e4m3fn) for t in (q, qv, kv)]
        return (q, qv, kv[..., 512:], kv[..., :512]), [t.to(torch.bfloat16) for t in (q, qv, kv)]
    return (q, qv, kv[..., 512:], kv[..., :512]), (q, qv, kv)


def check(
    q_lens, counts, num_heads=16, split_views=True, seed=0,
    dtype=torch.bfloat16, use_valid_length=True, pad=-1, fp8=False,
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
    (q_in, qv_in, k, v), (q, qv, kv) = to_fp8(fp8, q, qv, kv)

    out, lse = run_decode(
        q_in, qv_in, k, v, cu_seqlens_q, idx, valid_len if use_valid_length else None,
    )
    out_ref, lse_ref = flat_kv.ref_sparse_mla(q, qv, kv, idx, counts)
    out_pt, _ = flat_kv.ref_sparse_mla(q, qv, kv, idx, counts, upcast=False)

    flat_kv.assert_close(out, lse, out_ref, lse_ref, out_pt, counts)


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


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("pad", [-1, -2147483648, 1 << 20])
def test_decode_out_of_range_padding_sentinels(pad, fp8):
    """The gather forms a gmem pointer from the raw index, so any sentinel must be predicated."""
    check([1] * 9, ragged_counts(9), pad=pad, seed=51, fp8=fp8)


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
@pytest.mark.parametrize("num_heads", HEADS)
def test_decode_lse_optional_and_output_unchanged(num_heads, total_q):
    """Asking for the LSE must not perturb O, and not asking must return no LSE."""
    args, _, _ = lse_inputs(total_q, num_heads, seed=10 * total_q)
    out_no_lse, lse_none = run_decode(*args, return_lse=False)
    out_lse, lse = run_decode(*args, return_lse=True)
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


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("kv_heads", [[1, 2, 1], [2, 1, 2]])
def test_decode_cache_reuse_and_kv_offsets(kv_heads, fp8):
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
        (q_in, qv_in, k_in, v_in), (q, qv, kv) = to_fp8(fp8, q, qv, kv)
        # unsplit grids (one n-block): this is the LSE check of the 2-D store at head_base != 0
        out, lse = flash_attn_varlen_func(
            q_in, k_in, v_in, qv=qv_in,
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
                    out[row:row + 1, head_slice].to(ref.dtype), ref, atol=0.015, rtol=0.015,
                )
                # counts[1] == 0: -inf lse and zero output on the empty row.
                torch.testing.assert_close(
                    lse[row:row + 1, head_slice], lse_ref, atol=5e-3, rtol=0,
                )


@pytest.mark.parametrize(
    "dtype,fp8", [(torch.float16, False), (torch.bfloat16, False), (torch.bfloat16, True)]
)
@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("tokens,topk", [(1, 2048), (148, 256)])
@pytest.mark.parametrize("use_valid_length", [False, True])
def test_decode_leading_empty_block(dtype, fp8, num_heads, tokens, topk, use_valid_length):
    """An empty first block must not clamp subsequent negative score maxima to zero."""
    q = torch.zeros(tokens, num_heads, 64, device="cuda", dtype=dtype)
    qv = torch.full((tokens, num_heads, 512), -10, device="cuda", dtype=dtype)
    kv = torch.zeros(1, 1, 576, device="cuda", dtype=dtype)
    kv[..., :512] = 1
    (q, qv, k, v), _ = to_fp8(fp8, q, qv, kv)
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
    torch.testing.assert_close(out.float(), torch.ones_like(out, dtype=torch.float32), atol=0.002, rtol=0)


def fp8_flat_kv(num_rows, device, seed, poison_rows=4):
    """A per-tensor e4m3 flat cache (latent then rope, 576 B rows) after rows of NaN bytes."""
    _, k16, v16 = flat_kv.make_flat_kv(num_rows, device, seed=seed)
    store = torch.full((poison_rows + num_rows, 1, 576), 0x7F, device=device, dtype=torch.uint8)
    store[poison_rows:, :, :512] = v16.to(torch.float8_e4m3fn).view(torch.uint8)
    store[poison_rows:, :, 512:] = k16.to(torch.float8_e4m3fn).view(torch.uint8)
    kv8 = store[poison_rows:].view(torch.float8_e4m3fn)
    return kv8, kv8[..., 512:], kv8[..., :512]


def run_fp8_and_upcast(q, qv, kv8, idx, valid_len):
    """The decode kernel on e4m3 q/qv/K/V and the bf16 kernel on the exact upcast."""
    dtype16 = torch.bfloat16
    kv16 = kv8.to(dtype16)
    cu_seqlens_q = flat_kv.cu_seqlens_from([1] * q.shape[0], q.device)
    out8, lse8 = run_decode(q, qv, kv8[..., 512:], kv8[..., :512], cu_seqlens_q, idx, valid_len)
    out16, lse16 = run_decode(
        q.to(dtype16), qv.to(dtype16), kv16[..., 512:], kv16[..., :512], cu_seqlens_q, idx,
        valid_len,
    )
    assert out8.dtype == dtype16
    return out8, lse8, out16, lse16


def assert_fp8_q_close(out8, lse8, out16, lse16, counts):
    """fp8 output vs the 16-bit kernel: finite, empty rows zero, O relL2 and LSE bounds."""
    assert torch.isfinite(out8).all(), "NaN/inf in the fp8-MMA output"
    empty = torch.tensor([n == 0 for n in counts], device=out8.device)
    assert (out8[empty] == 0).all() and (lse8[empty] == -math.inf).all()
    rel = ((out8.float() - out16.float()).norm() / out16.float().norm()).item()
    assert rel < 4e-3, f"fp8-MMA output relL2 {rel:.2e} vs the 16-bit kernel"
    lse_err = (lse8[~empty] - lse16[~empty]).abs().max().item()
    assert lse_err < 1e-4, f"fp8-MMA LSE max diff {lse_err:.2e}"


@pytest.mark.parametrize("total_q", [9, 148])
@pytest.mark.parametrize("num_heads", HEADS)
def test_decode_fp8_vs_upcast_cache(num_heads, total_q):
    """e4m3 q/K/V against the bf16 kernel on the upcast inputs, with ragged/empty rows and
    -1 sentinels next to NaN bytes."""
    device = "cuda"
    counts = ragged_counts(total_q)
    kv8, _, _ = fp8_flat_kv(NUM_KV_ROWS, device, seed=60 + total_q)
    q, qv = flat_kv.make_q(
        total_q, device, dtype=torch.float8_e4m3fn, seed=61, num_heads=num_heads
    )
    idx, valid_len = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=62)
    out8, lse8, out16, lse16 = run_fp8_and_upcast(q, qv, kv8, idx, valid_len)
    assert_fp8_q_close(out8, lse8, out16, lse16, counts)


@pytest.mark.parametrize("n_blocks", [1, 2, 3])
@pytest.mark.parametrize(
    "total_q,topk,lead,splits", [(148, 3 * TILE_N, 0, 1), (1, 6 * TILE_N, 3, 2)],
    ids=["unsplit", "split"],
)
def test_decode_fp8_q_n_blocks(decode_route, total_q, topk, lead, splits, n_blocks):
    """e4m3 q with 1-3 blocks per CTA (S^T/P^T stage parity at the tail)."""
    device = "cuda"
    counts = [(lead + n_blocks) * TILE_N] * total_q
    kv8, _, _ = fp8_flat_kv(NUM_KV_ROWS, device, seed=63)
    q, qv = flat_kv.make_q(total_q, device, dtype=torch.float8_e4m3fn, seed=64, num_heads=16)
    idx, valid_len = flat_kv.make_indices(counts, NUM_KV_ROWS, device, seed=65, topk=topk)
    out8, lse8, out16, lse16 = run_fp8_and_upcast(q, qv, kv8, idx, valid_len)
    assert mla_decode_splits(*decode_route[0])[1] == splits
    assert_fp8_q_close(out8, lse8, out16, lse16, counts)


@pytest.mark.parametrize("num_heads", [8, 16, 32])
def test_decode_fp8_q_peaked_softmax(num_heads):
    """e4m3 q under a peaked softmax (logits 0, -1, -2, -3, the rest -20): every head's
    output is within 2^-8."""
    device = "cuda"
    total_q = NUM_KV_ROWS // TOPK  # disjoint rows, so each row's logit is per token
    kv8, _, _ = fp8_flat_kv(NUM_KV_ROWS, device, seed=66)
    idx = torch.randperm(NUM_KV_ROWS, device=device).view(total_q, TOPK).int()
    logits = torch.full((TOPK,), -20.0, device=device)
    logits[:4] = torch.tensor([0.0, -1.0, -2.0, -3.0])
    # score = SOFTMAX_SCALE * qv[0] * latent[0] with qv[0] = 2: latent[0] = 12 * logit (exact)
    kv8[idx.long(), 0, 0] = (logits / (2 * SOFTMAX_SCALE)).to(kv8.dtype)
    q = torch.zeros(total_q, num_heads, 64, device=device).to(torch.float8_e4m3fn)
    qv = torch.zeros(total_q, num_heads, 512, device=device)
    qv[..., 0] = 2
    out8, lse8, out16, lse16 = run_fp8_and_upcast(
        q, qv.to(torch.float8_e4m3fn), kv8, idx, torch.full((total_q,), TOPK, device=device).int()
    )
    err = (out8.float() - out16.float()).abs().amax(-1) / out16.float().abs().amax(-1)
    assert err.max() < 2.0**-8, f"per-head max error {err.max():.2e}"
    torch.testing.assert_close(lse8, lse16, atol=1e-4, rtol=0)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "q_lens",
    [[1] * 9, [2] * 9, [1, 2, 2, 1, 2, 1, 1, 2, 2], [1, 5, 2, 8, 1, 3, 16, 4]],
    ids=["q1", "q2", "q1q2", "ragged"],
)
def test_decode_batch_lookup_per_request_kv(q_lens, fp8):
    """Per-request KV regions (cu_seqlens_k + seqused_k) under varlen q: each token uses its
    own request and masks indices past that request's KV length."""
    device = "cuda"
    batch, total_q, topk, n_masked = len(q_lens), sum(q_lens), 256, 5
    kv_lens = [64 + 37 * r for r in range(batch)]
    cuq = flat_kv.cu_seqlens_from(q_lens, device)
    cuk = flat_kv.cu_seqlens_from(kv_lens, device)
    gen = torch.Generator(device=device).manual_seed(70)
    kv = torch.randn(sum(kv_lens), 1, 576, device=device, generator=gen).to(torch.bfloat16)
    q, qv = flat_kv.make_q(total_q, device, seed=71, num_heads=16)
    idx = torch.full((total_q, topk), -1, device=device, dtype=torch.int32)
    ref_idx = torch.full((total_q, topk), -1, device=device, dtype=torch.int32)
    counts, valid = [], []
    t = 0
    for r, n_q in enumerate(q_lens):
        for _ in range(n_q):
            n = [min(kv_lens[r], topk - n_masked), 1, 50][t % 3]
            perm = torch.randperm(kv_lens[r], device=device, generator=gen)[:n].int()
            idx[t, :n] = perm
            # in the valid window but past this request's KV length: masked by seqused_k
            idx[t, n:n + n_masked] = kv_lens[r] + torch.arange(n_masked, device=device).int()
            ref_idx[t, :n] = int(cuk[r]) + perm
            counts.append(n)
            valid.append(n + n_masked)
            t += 1
    (q_in, qv_in, k_in, v_in), (q, qv, kv) = to_fp8(fp8, q, qv, kv)
    out, lse = flash_attn_varlen_func(
        q_in, k_in, v_in, qv=qv_in,
        cu_seqlens_q=cuq, cu_seqlens_k=cuk,
        seqused_k=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
        gather_kv_indices=idx,
        gather_kv_valid_length=torch.tensor(valid, device=device, dtype=torch.int32),
        softmax_scale=SOFTMAX_SCALE,
        return_lse=True,
    )
    out_ref, lse_ref = flat_kv.ref_sparse_mla(q, qv, kv, ref_idx, counts)
    out_pt, _ = flat_kv.ref_sparse_mla(q, qv, kv, ref_idx, counts, upcast=False)
    flat_kv.assert_close(out, lse, out_ref, lse_ref, out_pt, counts)

