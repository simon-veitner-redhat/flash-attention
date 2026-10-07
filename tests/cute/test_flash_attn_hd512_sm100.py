"""hd512 GQA on SM100, served by the MLA kernel with a separate PV operand.

Shapes follow DiffusionGemma's global layers: 16 query heads, 2 KV heads,
d = dv = 512, a 256-token query block over a long paged prefix.
"""
import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd

DEV = "cuda"
DT = torch.bfloat16
HQ, HKV, D = 16, 2, 512
FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (10, 11)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="SM100-family only")


def _paged_inputs(batch, seqlen_q, seqlen_k, page, hq=HQ, hkv=HKV):
    pages = (seqlen_k + page - 1) // page
    num_blocks = batch * pages + 3
    kc = torch.randn(num_blocks, page, hkv, D, device=DEV, dtype=DT)
    vc = torch.randn(num_blocks, page, hkv, D, device=DEV, dtype=DT)
    block_table = torch.randperm(num_blocks, device=DEV)[: batch * pages]
    block_table = block_table.view(batch, pages).to(torch.int32)
    q = torch.randn(batch * seqlen_q, hq, D, device=DEV, dtype=DT)
    cu_q = torch.arange(batch + 1, device=DEV, dtype=torch.int32) * seqlen_q
    seqused_k = torch.full((batch,), seqlen_k, device=DEV, dtype=torch.int32)
    return q, kc, vc, block_table, cu_q, seqused_k


def _reference(q, kc, vc, block_table, seqlen_q, seqlen_k, scale, causal):
    outs, lses = [], []
    for b, is_causal in enumerate(causal):
        k = kc[block_table[b].long()].flatten(0, 1)[:seqlen_k].float()
        v = vc[block_table[b].long()].flatten(0, 1)[:seqlen_k].float()
        k, v = (t.repeat_interleave(q.shape[1] // kc.shape[2], 1) for t in (k, v))
        s = torch.einsum("qhd,khd->hqk", q[b * seqlen_q : (b + 1) * seqlen_q].float(), k)
        s = s * scale
        if is_causal:
            qi = torch.arange(seqlen_q, device=DEV)[:, None] + seqlen_k - seqlen_q
            s = s.masked_fill(torch.arange(seqlen_k, device=DEV) > qi, float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), v))
        lses.append(s.logsumexp(-1))
    return torch.cat(outs), torch.cat(lses, dim=1)


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 256), (256, 1280), (256, 4352), (1, 1024)])
@pytest.mark.parametrize("causal", [False, True])
def test_hd512_gqa_paged(page, seqlen_q, seqlen_k, causal):
    torch.manual_seed(0)
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(2, seqlen_q, seqlen_k, page)
    scale = D**-0.5
    out = _flash_attn_fwd(
        q, kc, vc,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        page_table=block_table,
        softmax_scale=scale,
        causal=causal,
    )[0]
    ref, _ = _reference(q, kc, vc, block_table, seqlen_q, seqlen_k, scale, [causal] * 2)
    assert _rel_err(out, ref) < 1e-2


# Two KV-head counts sharing a group size must not reuse each other's compiled kernel.
@pytest.mark.parametrize("hq,hkv", [(16, 2), (32, 4)])
def test_hd512_gqa_lse(hq, hkv):
    torch.manual_seed(0)
    seqlen_q, seqlen_k = 256, 1280
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(2, seqlen_q, seqlen_k, 16, hq, hkv)
    scale = D**-0.5
    kwargs = dict(
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        page_table=block_table,
        softmax_scale=scale,
        return_lse=True,
    )
    out, lse = _flash_attn_fwd(q, kc, vc, **kwargs)[:2]
    ref, ref_lse = _reference(q, kc, vc, block_table, seqlen_q, seqlen_k, scale, [False] * 2)
    assert _rel_err(out, ref) < 1e-2
    assert lse.shape == (hq, q.shape[0])
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-3)

    lse_buf = torch.empty(q.shape[0], hq, device=DEV, dtype=torch.float32).mT
    _flash_attn_fwd(q, kc, vc, lse=lse_buf, **kwargs)
    torch.testing.assert_close(lse_buf, ref_lse, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("fp8_dtype", FP8)
def test_hd512_gqa_rejects_fp8(fp8_dtype):
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(1, 16, 64, 16)
    with pytest.raises(NotImplementedError, match="FP8"):
        _flash_attn_fwd(
            q, kc.to(fp8_dtype), vc.to(fp8_dtype),
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            max_seqlen_q=16,
            max_seqlen_k=64,
            page_table=block_table,
        )


# MHA (one query head per KV head) with varlen Q: O must not spill past seqused_q.
@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("qlens", [[100, 37, 200], [64, 128, 64]])
@pytest.mark.parametrize("causal", [False, True])
def test_hd512_mha_varlen_seqused_q(page, qlens, causal):
    torch.manual_seed(0)
    h, seqlen_k, sentinel = 4, 1000, 7.0
    _, kc, vc, block_table, _, seqused_k = _paged_inputs(len(qlens), 1, seqlen_k, page, h, h)
    q = torch.randn(sum(qlens), h, D, device=DEV, dtype=DT)
    starts = [sum(qlens[:b]) for b in range(len(qlens))]
    cu_q = torch.tensor(starts + [sum(qlens)], device=DEV, dtype=torch.int32)
    used = [qlens[0], 0, qlens[2] // 2]
    out = torch.full_like(q, sentinel)
    scale = D**-0.5
    _flash_attn_fwd(
        q, kc, vc,
        out=out,
        cu_seqlens_q=cu_q,
        seqused_q=torch.tensor(used, device=DEV, dtype=torch.int32),
        seqused_k=seqused_k,
        max_seqlen_q=max(qlens),
        max_seqlen_k=seqlen_k,
        page_table=block_table,
        softmax_scale=scale,
        causal=causal,
    )
    for b, (start, n) in enumerate(zip(starts, used)):
        assert (out[start + n : start + qlens[b]] == sentinel).all()
        if n > 0:
            ref, _ = _reference(
                q[start : start + n], kc, vc, block_table[b : b + 1], n, seqlen_k, scale, [causal]
            )
            assert _rel_err(out[start : start + n], ref) < 1e-2


@pytest.mark.parametrize("hq,hkv", [(6, 2), (12, 1)])
def test_hd512_gqa_rejects_head_ratio(hq, hkv):
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(1, 16, 64, 16, hq, hkv)
    with pytest.raises(NotImplementedError, match="num_head / num_head_kv"):
        _flash_attn_fwd(
            q, kc, vc,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            max_seqlen_q=16,
            max_seqlen_k=64,
            page_table=block_table,
        )


def test_hd512_mqa128_rejects_seqused_q():
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(2, 16, 64, 16, 128, 1)
    with pytest.raises(NotImplementedError, match="seqused_q"):
        _flash_attn_fwd(
            q, kc, vc,
            cu_seqlens_q=cu_q,
            seqused_q=torch.tensor([16, 0], device=DEV, dtype=torch.int32),
            seqused_k=seqused_k,
            max_seqlen_q=16,
            max_seqlen_k=64,
            page_table=block_table,
        )


def test_hd512_gqa_rejects_backward():
    q, kc, vc, block_table, cu_q, seqused_k = _paged_inputs(1, 16, 64, 16)
    with pytest.raises(NotImplementedError, match="backward"):
        _flash_attn_fwd(
            q.requires_grad_(), kc, vc,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            max_seqlen_q=16,
            max_seqlen_k=64,
            page_table=block_table,
        )
