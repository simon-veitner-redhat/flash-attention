# Copyright (c) 2026, Tri Dao.
"""Frozen C01-C18 SM90 native-E4M3 correctness and execution contracts."""
import math
import sys
import types
from dataclasses import dataclass

import pytest
import torch

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
import flash_attn.cute.interface as cute_interface
import flash_attn.cute.cute_dsl_utils as cute_dsl_utils
from flash_attn.cute.interface import _classify_descale_layout, _flash_attn_fwd
from flash_attn.cute.flash_fwd_sm90 import (
    _use_intra_wg_overlap_sm90,
    _use_paged_kv_overlap_sm90,
)

from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    get_scheduler_metadata,
)

IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
pytestmark = pytest.mark.skipif(not IS_SM90, reason="native E4M3 FA4 requires SM90")
FP8 = torch.float8_e4m3fn
OFFSETS = (0.23, 0.41, 0.67)
RAGGED_K = (33, 127, 255, 386, 577, 769, 1025)
RAGGED_SPLITS = (1, 1, 2, 3, 3, 3, 3)

@pytest.mark.parametrize(
    (
        "intra_wg_overlap",
        "has_qv",
        "native_fp8",
        "paged_kv_non_tma",
        "tile_n",
        "expected",
    ),
    (
        pytest.param(True, False, True, True, 128, (False, False), id="native-paged"),
        pytest.param(True, False, True, False, 128, (True, False), id="native-tma-or-nonpaged"),
        pytest.param(True, True, True, True, 128, (False, False), id="native-has-qv"),
        pytest.param(True, False, False, True, 128, (True, True), id="generic-paged-supported"),
        pytest.param(True, False, False, True, 96, (True, False), id="generic-paged-unsupported"),
        pytest.param(True, False, False, False, 128, (True, False), id="generic-tma-or-nonpaged"),
        pytest.param(True, True, False, True, 128, (False, False), id="generic-has-qv"),
        pytest.param(False, False, False, True, 128, (False, False), id="generic-overlap-disabled"),
    ),
)
def test_sm90_paged_overlap_decision_is_tile_gated(
    intra_wg_overlap,
    has_qv,
    native_fp8,
    paged_kv_non_tma,
    tile_n,
    expected,
):
    use_intra_wg_overlap = _use_intra_wg_overlap_sm90(
        intra_wg_overlap, has_qv, native_fp8, paged_kv_non_tma
    )
    use_paged_kv_overlap = _use_paged_kv_overlap_sm90(
        use_intra_wg_overlap, native_fp8, paged_kv_non_tma, tile_n
    )
    assert (use_intra_wg_overlap, use_paged_kv_overlap) == expected


@dataclass(frozen=True)
class FrozenCase:
    id: str
    d: int
    dv: int
    hq: int
    hkv: int
    batch: int
    sq: int
    sk: int
    causal: bool
    layout: str
    seeds: tuple[int, int, int]
    mode: str = "direct"
    num_splits: int = 1
    q_lens: tuple[int, ...] | None = None
    k_lens: tuple[int, ...] | None = None
    page_size: int | None = None
    shuffle_pages: bool = False


CASES = (
    FrozenCase("C01", 64, 64, 4, 4, 2, 17, 29, False, "scalar", (74, 84, 94)),
    FrozenCase("C02", 96, 96, 4, 2, 2, 17, 29, True, "contiguous", (106, 116, 126)),
    FrozenCase("C03", 128, 128, 8, 2, 2, 17, 29, False, "none", (138, 148, 158)),
    FrozenCase("C04", 192, 192, 4, 2, 2, 17, 29, True, "batch-broadcast", (202, 212, 222)),
    FrozenCase("C05", 256, 256, 4, 4, 2, 17, 29, False, "head-broadcast", (266, 276, 286)),
    FrozenCase("C06", 192, 128, 4, 4, 2, 1, 37, True, "contiguous", (101, 102, 103)),
    FrozenCase("C07", 192, 128, 8, 2, 2, 23, 37, True, "both-broadcast", (111, 112, 113), num_splits=2),
    FrozenCase("C08", 192, 128, 8, 1, 2, 23, 37, False, "scalar", (121, 122, 123)),
    FrozenCase("C09", 128, 128, 8, 2, 2, 19, 31, True, "contiguous", (201, 202, 203), "varlen", q_lens=(7, 19), k_lens=(31, 23)),
    FrozenCase("C10", 96, 96, 8, 8, 2, 1, 31, True, "head-broadcast", (211, 212, 213), "paged", 2, (1, 1), (31, 23), 16),
    FrozenCase("C11", 192, 128, 8, 2, 2, 1, 93, True, "batch-broadcast", (301, 302, 303), "paged", 2, (1, 1), (93, 77), 64),
    FrozenCase("C12", 128, 128, 8, 2, 2, 23, 37, True, "scalar", (311, 312, 313), num_splits=2),
    FrozenCase("C13", 128, 128, 16, 4, 7, 1, 1025, True, "contiguous", (42, 43, 44), "dynamic", 3, (1,) * 7, RAGGED_K, 32),
    FrozenCase("C14", 256, 256, 16, 16, 7, 1, 1025, True, "contiguous", (42, 43, 44), "dynamic", 3, (1,) * 7, RAGGED_K, 16),
    FrozenCase("C15", 192, 128, 16, 4, 7, 1, 1025, True, "contiguous", (42, 43, 44), "dynamic", 3, (1,) * 7, RAGGED_K, 64),
    FrozenCase("C16", 128, 128, 8, 2, 1, 1, 64, True, "scalar", (501, 502, 503), "graph", 2),
    FrozenCase("C17", 192, 128, 16, 4, 7, 1, 1025, True, "contiguous", (42, 43, 44), "piecewise-graph", 3, (1,) * 7, RAGGED_K, 64),
    FrozenCase("C18", 192, 128, 8, 1, 2, 23, 37, False, "both-broadcast", (121, 122, 123), "graph"),
)
CASES_BY_ID = {case.id: case for case in CASES}


def _case(case_id: str) -> FrozenCase:
    return CASES_BY_ID[case_id]



PAGED_LONG_D96_CASES = (
    FrozenCase(
        "paged-d96-mha-q65-k145", 96, 96, 1, 1, 1, 65, 145, True,
        "scalar", (821, 822, 823), "paged",
        q_lens=(65,), k_lens=(145,), page_size=16, shuffle_pages=True,
    ),
    FrozenCase(
        "paged-d96-gqa4-q17-k145", 96, 96, 4, 1, 1, 17, 145, True,
        "contiguous", (831, 832, 833), "paged",
        q_lens=(17,), k_lens=(145,), page_size=16, shuffle_pages=True,
    ),
    FrozenCase(
        "paged-d96-mha-q2048-k2065", 96, 96, 1, 1, 1, 2048, 2065, True,
        "scalar", (841, 842, 843), "paged",
        q_lens=(2048,), k_lens=(2065,), page_size=16, shuffle_pages=True,
    ),
)


def _logical_scale(batch: int, heads: int, offset: float) -> torch.Tensor:
    b = torch.arange(batch, device="cuda", dtype=torch.float32)[:, None]
    h = torch.arange(heads, device="cuda", dtype=torch.float32)[None, :]
    return (offset + 0.07 * b + 0.03 * h).contiguous()


def _layout_scale(batch: int, heads: int, offset: float, layout: str):
    if layout == "none":
        return None
    if layout == "scalar":
        return torch.tensor(offset, device="cuda", dtype=torch.float32)
    if layout == "contiguous":
        return _logical_scale(batch, heads, offset)
    if layout == "batch-broadcast":
        return (offset + 0.03 * torch.arange(heads, device="cuda", dtype=torch.float32))[None, :].expand(batch, heads)
    if layout == "head-broadcast":
        return (offset + 0.07 * torch.arange(batch, device="cuda", dtype=torch.float32))[:, None].expand(batch, heads)
    if layout == "both-broadcast":
        return torch.tensor(offset, device="cuda", dtype=torch.float32).reshape(1, 1).expand(batch, heads)
    raise AssertionError(layout)


def _expanded_scale(scale, batch: int, heads: int):
    if scale is None:
        return torch.ones((batch, heads), device="cuda", dtype=torch.float32)
    if scale.ndim == 0:
        return scale.expand(batch, heads)
    return scale

def _fa3_descale(scale, batch: int, heads: int):
    if scale is None:
        return None
    return scale.expand(batch, heads).contiguous()
def _fa3_result(case: FrozenCase, kwargs, *, return_lse: bool = False):
    q, k, v = (kwargs[name] for name in ("q", "k", "v"))
    cuq, cuk = kwargs.get("cu_seqlens_q"), kwargs.get("cu_seqlens_k")
    seqused_k = kwargs.get("seqused_k")
    descales = tuple(
        _fa3_descale(kwargs[name], case.batch, case.hkv)
        for name in ("q_descale", "k_descale", "v_descale")
    )
    if cuq is None:
        cuq = torch.arange(
            0, (case.batch + 1) * case.sq, case.sq,
            device="cuda", dtype=torch.int32,
        )
        cuk = torch.arange(
            0, (case.batch + 1) * case.sk, case.sk,
            device="cuda", dtype=torch.int32,
        )
        q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
    q_lens = case.q_lens or (case.sq,)
    k_lens = case.k_lens or (case.sk,)
    num_splits = (
        3 if case.mode in ("dynamic", "piecewise-graph") else case.num_splits
    )
    cache_lens = seqused_k
    if cache_lens is None:
        cache_lens = torch.tensor(
            case.k_lens or (case.sk,) * case.batch,
            device="cuda", dtype=torch.int32,
        )
    metadata = get_scheduler_metadata(
        case.batch, max(q_lens), max(k_lens), case.hq, case.hkv, case.d,
        cache_lens, qkv_dtype=q.dtype, headdim_v=case.dv,
        cu_seqlens_q=cuq, page_size=case.page_size, causal=case.causal,
        num_splits=num_splits,
    )
    result = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=max(q_lens), cu_seqlens_q=cuq,
        max_seqlen_k=max(k_lens), cu_seqlens_k=cuk, seqused_k=seqused_k,
        block_table=kwargs.get("page_table"), q_descale=descales[0],
        k_descale=descales[1], v_descale=descales[2], causal=case.causal,
        scheduler_metadata=metadata, num_splits=num_splits, fa_version=3,
        out=torch.empty(
            (q.shape[0], q.shape[1], v.shape[-1]),
            device="cuda", dtype=torch.bfloat16,
        ),
        return_softmax_lse=return_lse,
    )
    if return_lse:
        return result
    out = result[0] if isinstance(result, (tuple, list)) else result
    if case.mode not in ("varlen", "paged", "dynamic", "piecewise-graph"):
        out = out.unflatten(0, (case.batch, case.sq))
    return out




def _make_fp8(shape, scale, *, seed: int, q_heads: int | None = None):
    source = torch.randn(shape, generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda") * 0.7
    expanded = _expanded_scale(scale, shape[0], shape[-2])
    if q_heads is not None:
        expanded = expanded.repeat_interleave(q_heads // expanded.shape[1], dim=1)
    return (source / expanded[:, None, :, None]).clamp(torch.finfo(FP8).min, torch.finfo(FP8).max).to(FP8)
def _make_qkv(batch, sq, sk, hq, hkv, d, dv, scales, seeds):
    return (
        _make_fp8((batch, sq, hq, d), scales[0], seed=seeds[0], q_heads=hq),
        _make_fp8((batch, sk, hkv, d), scales[1], seed=seeds[1]),
        _make_fp8((batch, sk, hkv, dv), scales[2], seed=seeds[2]),
    )


def _native_call(q, k, v, scales, *, causal, **kwargs):
    return _flash_attn_fwd(
        q=q, k=k, v=v, q_descale=scales[0], k_descale=scales[1],
        v_descale=scales[2], softmax_scale=q.shape[-1] ** -0.5,
        causal=causal, out_dtype=torch.bfloat16, **kwargs,
    )




def _reference(q, k, v, scales, *, causal):
    qs, ks, vs = (_expanded_scale(s, q.shape[0], k.shape[2]) for s in scales)
    qf = q.float() * qs.repeat_interleave(q.shape[2] // k.shape[2], 1)[:, None, :, None]
    kf = (k.float() * ks[:, None, :, None]).repeat_interleave(q.shape[2] // k.shape[2], 2)
    vf = (v.float() * vs[:, None, :, None]).repeat_interleave(q.shape[2] // k.shape[2], 2)
    scores = torch.einsum("bthd,bshd->bhts", qf * (q.shape[-1] ** -0.5), kf)
    if causal:
        qpos = torch.arange(q.shape[1], device="cuda") + k.shape[1] - q.shape[1]
        kpos = torch.arange(k.shape[1], device="cuda")
        scores.masked_fill_(kpos[None, :] > qpos[:, None], -torch.inf)
    return torch.einsum("bhts,bshd->bthd", scores.softmax(-1), vf)


def _make_case(case: FrozenCase):
    scales = tuple(
        _layout_scale(case.batch, case.hkv, offset, case.layout)
        for offset in OFFSETS
    )
    return (
        *_make_qkv(
            case.batch, case.sq, case.sk, case.hq, case.hkv, case.d, case.dv,
            scales, case.seeds,
        ),
        scales,
    )


def _pack(x, lengths):
    return torch.cat([x[b, :length] for b, length in enumerate(lengths)])


def _arguments(case: FrozenCase):
    q, logical_k, logical_v, scales = _make_case(case)
    refs = []
    q_lens = case.q_lens or (case.sq,) * case.batch
    k_lens = case.k_lens or (case.sk,) * case.batch
    for b, (ql, kl) in enumerate(zip(q_lens, k_lens)):
        local_scales = tuple(None if s is None else (s if s.ndim == 0 else s[b:b + 1]) for s in scales)
        refs.append(_reference(q[b:b + 1, :ql], logical_k[b:b + 1, :kl], logical_v[b:b + 1, :kl], local_scales, causal=case.causal)[0])
    expected = torch.cat(refs) if case.mode in ("varlen", "paged", "dynamic", "piecewise-graph") else torch.stack(refs)
    cuq = torch.tensor((0, *torch.tensor(q_lens).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    q_arg = _pack(q, q_lens) if case.mode in ("varlen", "paged", "dynamic", "piecewise-graph") else q
    kwargs = dict(q=q_arg, q_descale=scales[0], k_descale=scales[1], v_descale=scales[2],
                  softmax_scale=case.d ** -0.5, causal=case.causal, num_splits=case.num_splits,
                  fp8_kv_dequant=False, return_lse=True)
    if case.mode == "varlen":
        cuk = torch.tensor((0, *torch.tensor(k_lens).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
        kwargs.update(k=_pack(logical_k, k_lens), v=_pack(logical_v, k_lens), cu_seqlens_q=cuq,
                      cu_seqlens_k=cuk, max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens))
    elif case.page_size:
        counts = tuple(math.ceil(length / case.page_size) for length in k_lens)
        total = sum(counts)
        ids = (
            torch.randperm(
                total, generator=torch.Generator(device="cpu").manual_seed(42)
            ).to("cuda", torch.int32)
            if case.shuffle_pages or case.mode in ("dynamic", "piecewise-graph")
            else torch.arange(total, device="cuda", dtype=torch.int32)
        )
        table = torch.full((case.batch, max(counts)), -1, device="cuda", dtype=torch.int32)
        kpages = torch.zeros((total, case.page_size, case.hkv, case.d), device="cuda", dtype=FP8)
        vpages = torch.zeros((total, case.page_size, case.hkv, case.dv), device="cuda", dtype=FP8)
        cursor = 0
        for b, count in enumerate(counts):
            row = ids[cursor:cursor + count]; table[b, :count] = row
            for p, physical in enumerate(row.tolist()):
                begin, end = p * case.page_size, min((p + 1) * case.page_size, k_lens[b])
                kpages[physical, :end - begin].copy_(logical_k[b, begin:end])
                vpages[physical, :end - begin].copy_(logical_v[b, begin:end])
            cursor += count
        kwargs.update(k=kpages, v=vpages, cu_seqlens_q=cuq, seqused_k=torch.tensor(k_lens, device="cuda", dtype=torch.int32),
                      max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens), page_table=table)
        if case.mode in ("dynamic", "piecewise-graph"):
            kwargs["num_splits_dynamic_ptr"] = torch.tensor(RAGGED_SPLITS, device="cuda", dtype=torch.int32)
    else:
        kwargs.update(k=logical_k, v=logical_v)
    return kwargs, expected


def _run(kwargs, dtype, *, out=None, lse=None):
    call = dict(kwargs, out_dtype=dtype)
    if out is not None: call["out"] = out
    if lse is not None: call["lse"] = lse
    result = _flash_attn_fwd(**call)
    return result[0], result[1]


def _assert_result(out, lse, expected, dtype):
    assert out.dtype == dtype and out.shape == expected.shape
    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(out.float(), expected.float(), atol=0.30, rtol=0.20)
def _capture_graph(kwargs, dtype, eager_out, eager_lse):
    out, lse = torch.empty_like(eager_out), torch.empty_like(eager_lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(kwargs, dtype, out=out, lse=lse)
    return graph, out, lse


def _assert_graph_replays(graph, out, lse, expected_out, expected_lse):
    for _ in range(100):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected_out)
        assert torch.equal(lse, expected_lse)


def _assert_no_allocations(fn, device):
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device)
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device)
    for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
        assert after[key] == before[key]


def _assert_no_copy_ops(fn):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
    ) as profile:
        for _ in range(100):
            fn()
    forbidden = {"aten::contiguous", "aten::clone", "aten::copy_"}
    assert not forbidden.intersection(event.key for event in profile.key_averages())




@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_frozen_c01_c18_matrix(case, dtype):
    kwargs, expected = _arguments(case)
    out, lse = _run(kwargs, dtype)
    _assert_result(out, lse, expected, dtype)
    if dtype == torch.float16:
        bf16, _ = _run(kwargs, torch.bfloat16)
        torch.testing.assert_close(out.float(), bf16.float(), atol=0.30, rtol=0.20)
    else:
        out3 = _fa3_result(case, kwargs)
        torch.testing.assert_close(out.float(), out3.float(), atol=0.30, rtol=0.20)

def test_native_fp8_d128_gqa_paged_page16_long_query_matches_fa3():
    case = FrozenCase(
        "d128-gqa-paged-page16-long-query",
        128,
        128,
        32,
        8,
        1,
        17,
        193,
        True,
        "contiguous",
        (811, 812, 813),
        mode="paged",
        q_lens=(17,),
        k_lens=(193,),
        page_size=16,
        shuffle_pages=True,
    )
    kwargs, expected = _arguments(case)
    page_table = kwargs["page_table"]
    assert not torch.equal(
        page_table[0], torch.arange(page_table.shape[1], device="cuda", dtype=torch.int32)
    )

    out4, lse4 = _run(kwargs, torch.bfloat16)
    _assert_result(out4, lse4, expected, torch.bfloat16)

    out3, lse3 = _fa3_result(case, kwargs, return_lse=True)
    assert out3.shape == out4.shape
    assert lse3.shape == lse4.shape
    assert lse3.dtype == torch.float32
    assert torch.isfinite(out3).all() and torch.isfinite(lse3).all()
    torch.testing.assert_close(out4.float(), out3.float(), atol=0.30, rtol=0.20)
    torch.testing.assert_close(lse4, lse3, atol=0.30, rtol=0.20)


@pytest.mark.parametrize("case", PAGED_LONG_D96_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_native_fp8_paged_long_d96_matches_reference(case, dtype):
    assert case.sq * (case.hq // case.hkv) > 64
    assert case.page_size == 16
    assert case.sk > 64 and case.sk % case.page_size and case.sk % 64
    kwargs, expected = _arguments(case)
    page_table = kwargs["page_table"]
    assert not torch.equal(
        page_table[0],
        torch.arange(page_table.shape[1], device="cuda", dtype=torch.int32),
    )

    eager_out, eager_lse = _run(kwargs, dtype)
    _assert_result(eager_out, eager_lse, expected, dtype)

    if case.id == "paged-d96-mha-q65-k145":
        graph, out, lse = _capture_graph(kwargs, dtype, eager_out, eager_lse)
        _assert_graph_replays(graph, out, lse, eager_out, eager_lse)
        _assert_result(out, lse, expected, dtype)


def test_native_fp8_output_dtype_precedence() -> None:
    kwargs, expected = _arguments(_case("C01"))

    default_out, default_lse = _flash_attn_fwd(**kwargs)[:2]
    _assert_result(default_out, default_lse, expected, torch.bfloat16)

    supplied_out = torch.empty_like(default_out, dtype=torch.float16)
    result = _flash_attn_fwd(**kwargs, out=supplied_out)
    assert result[0] is supplied_out
    _assert_result(result[0], result[1], expected, torch.float16)


@pytest.mark.parametrize(
    "out_dtype,out_buffer_dtype,error",
    (
        (torch.float32, None, "out_dtype must be"),
        (torch.bfloat16, torch.float16, "does not match"),
        (None, torch.float32, "out must have dtype"),
    ),
)
def test_native_fp8_invalid_output_dtype_rejects_before_compile(
    monkeypatch, out_dtype, out_buffer_dtype, error
) -> None:
    kwargs, _ = _arguments(_case("C01"))
    out = (
        torch.empty(
            (2, 17, 4, 64), device="cuda", dtype=out_buffer_dtype
        )
        if out_buffer_dtype is not None
        else None
    )
    compile_called = False

    def unexpected_compile(*args, **compile_kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("compile must not be invoked")

    monkeypatch.setattr("flash_attn.cute.interface.cute.compile", unexpected_compile)
    with pytest.raises(AssertionError, match=error):
        _flash_attn_fwd(**kwargs, out=out, out_dtype=out_dtype)
    assert not compile_called




def test_dynamic_scheduler_counter_default_and_public_api_compatibility():
    kwargs, _ = _arguments(_case("C09"))
    omitted_out, omitted_lse = _run(kwargs, torch.bfloat16)
    counter = torch.full((1,), 0x5A5A, device=kwargs["q"].device, dtype=torch.int32)
    explicit_out, explicit_lse = _run(
        dict(kwargs, dynamic_scheduler_counter=counter), torch.bfloat16
    )
    assert torch.equal(explicit_out, omitted_out)
    assert torch.equal(explicit_lse, omitted_lse)

    public_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name not in ("fp8_kv_dequant", "return_lse")
    }
    public_result = cute_interface.flash_attn_varlen_func(
        **public_kwargs,
        return_lse=True,
        out_dtype=torch.bfloat16,
        dynamic_scheduler_counter=counter,
    )
    assert torch.equal(public_result[0], omitted_out)
    assert torch.equal(public_result[1], omitted_lse)


def test_invalid_dynamic_scheduler_counters_reject_before_compile(monkeypatch):
    kwargs, _ = _arguments(_case("C09"))
    compile_called = False

    def unexpected_compile(*args, **compile_kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("compile must not be invoked")

    monkeypatch.setattr("flash_attn.cute.interface.cute.compile", unexpected_compile)
    invalid = (
        torch.zeros((2,), device="cuda", dtype=torch.int32),
        torch.zeros((1,), device="cuda", dtype=torch.int64),
        torch.zeros((1,), device="cpu", dtype=torch.int32),
    )
    for counter in invalid:
        with pytest.raises(AssertionError, match="dynamic_scheduler_counter"):
            _flash_attn_fwd(**kwargs, dynamic_scheduler_counter=counter)
    inactive_kwargs, _ = _arguments(_case("C01"))
    with pytest.raises(AssertionError, match="dynamic_scheduler_counter"):
        _flash_attn_fwd(
            **inactive_kwargs,
            dynamic_scheduler_counter=torch.zeros(
                (2,), device=inactive_kwargs["q"].device, dtype=torch.int32
            ),
        )
    assert not compile_called


def test_dynamic_scheduler_counter_is_zeroed_at_main_dispatch(monkeypatch):
    kwargs, _ = _arguments(_case("C09"))
    counter = torch.zeros((1,), device=kwargs["q"].device, dtype=torch.int32)
    kwargs["dynamic_scheduler_counter"] = counter
    _run(kwargs, torch.bfloat16)
    observed = False
    cache_type = type(_flash_attn_fwd.compile_cache)
    original_getitem = cache_type.__getitem__

    def intercept_getitem(cache, key):
        compiled = original_getitem(cache, key)

        def wrapped(*args):
            nonlocal observed
            if any(arg is counter for arg in args):
                assert len(args) >= 4
                assert args[-4] is counter
                assert counter.item() == 0
                observed = True
            return compiled(*args)

        return wrapped

    monkeypatch.setattr(cache_type, "__getitem__", intercept_getitem)

    counter.fill_(0x5A5A)
    _run(kwargs, torch.bfloat16)
    assert observed


def test_explicit_dynamic_scheduler_counter_has_stable_storage_and_no_allocations():
    kwargs, _ = _arguments(_case("C09"))
    counter = torch.zeros((1,), device=kwargs["q"].device, dtype=torch.int32)
    kwargs["dynamic_scheduler_counter"] = counter
    eager_out, eager_lse = _run(kwargs, torch.bfloat16)
    out, lse = torch.empty_like(eager_out), torch.empty_like(eager_lse)
    for _ in range(3):
        counter.fill_(0x5A5A)
        _run(kwargs, torch.bfloat16, out=out, lse=lse)
    torch.cuda.synchronize()
    pointer = counter.data_ptr()
    def measured_call():
        counter.fill_(0x5A5A)
        _run(kwargs, torch.bfloat16, out=out, lse=lse)
        assert counter.data_ptr() == pointer

    _assert_no_allocations(measured_call, kwargs["q"].device)
    assert torch.equal(out, eager_out)
    assert torch.equal(lse, eager_lse)


def test_explicit_counter_alternates_non_split_varlen_work_without_sync():
    base, _ = _arguments(_case("C09"))
    small = dict(base)
    large = dict(base)
    small["cu_seqlens_q"] = torch.tensor((0, 1, 26), device="cuda", dtype=torch.int32)
    small["cu_seqlens_k"] = torch.tensor((0, 1, 54), device="cuda", dtype=torch.int32)
    large["cu_seqlens_q"] = torch.tensor((0, 13, 26), device="cuda", dtype=torch.int32)
    large["cu_seqlens_k"] = torch.tensor((0, 27, 54), device="cuda", dtype=torch.int32)
    counter = torch.zeros((1,), device=base["q"].device, dtype=torch.int32)
    small["dynamic_scheduler_counter"] = counter
    large["dynamic_scheduler_counter"] = counter
    small_ref = _run(small, torch.bfloat16)
    large_ref = _run(large, torch.bfloat16)
    small_out, small_lse = torch.empty_like(small_ref[0]), torch.empty_like(small_ref[1])
    large_out, large_lse = torch.empty_like(large_ref[0]), torch.empty_like(large_ref[1])
    for _ in range(50):
        _run(large, torch.bfloat16, out=large_out, lse=large_lse)
        _run(small, torch.bfloat16, out=small_out, lse=small_lse)
    torch.cuda.synchronize()
    assert torch.equal(small_out, small_ref[0])
    assert torch.equal(small_lse, small_ref[1])
    assert torch.equal(large_out, large_ref[0])
    assert torch.equal(large_lse, large_ref[1])


def test_explicit_counter_non_split_graph_replays_bitwise():
    kwargs, _ = _arguments(_case("C09"))
    counter = torch.zeros((1,), device=kwargs["q"].device, dtype=torch.int32)
    kwargs["dynamic_scheduler_counter"] = counter
    eager_out, eager_lse = _run(kwargs, torch.bfloat16)
    graph, out, lse = _capture_graph(
        kwargs, torch.bfloat16, eager_out, eager_lse
    )
    _assert_graph_replays(graph, out, lse, eager_out, eager_lse)


def test_isolated_counters_on_interleaved_non_default_streams():
    kwargs, _ = _arguments(_case("C09"))
    counters = [
        torch.zeros((1,), device=kwargs["q"].device, dtype=torch.int32)
        for _ in range(2)
    ]
    references = [
        _run(
            dict(kwargs, dynamic_scheduler_counter=counter),
            torch.bfloat16,
        )
        for counter in counters
    ]
    outputs = [
        (torch.empty_like(references[i][0]), torch.empty_like(references[i][1]))
        for i in range(2)
    ]
    streams = [torch.cuda.Stream(device=kwargs["q"].device) for _ in range(2)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(kwargs["q"].device))
    for _ in range(50):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                _run(
                    dict(kwargs, dynamic_scheduler_counter=counters[i]),
                    torch.bfloat16,
                    out=outputs[i][0],
                    lse=outputs[i][1],
                )
    torch.cuda.synchronize(kwargs["q"].device)
    for output, reference in zip(outputs, references):
        assert torch.equal(output[0], reference[0])
        assert torch.equal(output[1], reference[1])


@pytest.mark.parametrize("case_id", ("C16", "C18"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_full_graph_replay_bitwise(case_id, dtype):
    case = _case(case_id)
    kwargs, expected = _arguments(case)
    eager_out, eager_lse = _run(kwargs, dtype)
    graph, out, lse = _capture_graph(kwargs, dtype, eager_out, eager_lse)
    _assert_graph_replays(graph, out, lse, eager_out, eager_lse)
    _assert_result(out, lse, expected, dtype)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_c17_piecewise_graph_dynamic_split_bitwise(dtype):
    case = _case("C17")
    kwargs, expected = _arguments(case)
    kwargs["dynamic_scheduler_counter"] = torch.zeros(
        (1,), device=kwargs["q"].device, dtype=torch.int32
    )
    eager_out, eager_lse = _run(kwargs, dtype)
    graph, out, lse = _capture_graph(kwargs, dtype, eager_out, eager_lse)
    for counts in (RAGGED_SPLITS, (1, 1, 1, 2, 2, 3, 3)):
        kwargs["num_splits_dynamic_ptr"].copy_(
            torch.tensor(counts, device="cuda", dtype=torch.int32)
        )
        reference_out, reference_lse = _run(kwargs, dtype)
        _assert_graph_replays(graph, out, lse, reference_out, reference_lse)
        _assert_result(out, lse, expected, dtype)


@pytest.mark.parametrize("case_id", ("C08", "C02", "C04", "C05"))
def test_descale_eager_allocation_and_profile_window(case_id):
    case = _case(case_id)
    kwargs, _ = _arguments(case)
    eager_out, eager_lse = _run(kwargs, torch.bfloat16)
    out, lse = torch.empty_like(eager_out), torch.empty_like(eager_lse)
    call = lambda: _run(kwargs, torch.bfloat16, out=out, lse=lse)
    for _ in range(3):
        call()
    _assert_no_allocations(call, kwargs["q"].device)
    _assert_no_copy_ops(call)


@pytest.mark.parametrize("case_id", ("C17", "C18"))
def test_descale_graph_allocation_and_profile_window(case_id):
    case = _case(case_id)
    kwargs, _ = _arguments(case)
    eager_out, eager_lse = _run(kwargs, torch.bfloat16)
    graph, out, lse = _capture_graph(
        kwargs, torch.bfloat16, eager_out, eager_lse
    )
    states = (
        (RAGGED_SPLITS, (1, 1, 1, 2, 2, 3, 3))
        if case_id == "C17" else (None,)
    )
    for counts in states:
        if counts is not None:
            kwargs["num_splits_dynamic_ptr"].copy_(
                torch.tensor(counts, device="cuda", dtype=torch.int32)
            )
        graph.replay()
        torch.cuda.synchronize()
        _assert_no_allocations(graph.replay, kwargs["q"].device)
        _assert_no_copy_ops(graph.replay)


def _observe_split_workspace(monkeypatch, kwargs):
    allocations = []
    original_empty = torch.empty

    def observe_empty(*shape, **kwargs):
        result = original_empty(*shape, **kwargs)
        if kwargs.get("dtype") == torch.float32:
            allocations.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch, "empty", observe_empty)
    result = _flash_attn_fwd(**kwargs, out_dtype=torch.bfloat16)
    return result, allocations


def _static_split_case(
    case_id, seed, *, d=256, hq=4, hkv=1, batch=1, sk=128,
    mode="direct", num_splits=1, k_lens=None, page_size=None,
):
    varlen = mode != "direct"
    return FrozenCase(
        case_id, d, d, hq, hkv, batch, 1, sk, False, "scalar",
        (seed, seed + 1, seed + 2), mode, num_splits,
        (1,) * batch if varlen else None,
        k_lens if k_lens is not None else (sk,) * batch if varlen else None,
        page_size,
    )


@pytest.mark.parametrize(
    "case,expected_splits",
    (
        pytest.param(
            _static_split_case("static-boundary", 701, num_splits=9),
            2,
            id="direct-k128-request9-caps2",
        ),
        pytest.param(
            _static_split_case(
                "static-tail", 711, batch=2, sk=129, mode="varlen",
                num_splits=9, k_lens=(129, 65),
            ),
            3,
            id="varlen-k129-request9-caps3",
        ),
        pytest.param(
            _static_split_case(
                "static-below-available", 721, batch=2, sk=257, mode="paged",
                num_splits=2, k_lens=(257, 129), page_size=64,
            ),
            2,
            id="paged-request2-remains2",
        ),
        pytest.param(
            _static_split_case(
                "static-occupancy-below-boundary", 751, d=128, hq=16, hkv=4,
                batch=7, sk=4096, mode="paged", num_splits=3, page_size=16,
            ),
            3,
            id="paged-b7-hkv4-request3-remains3",
        ),
        pytest.param(
            _static_split_case(
                "static-occupancy-cap2", 761, d=128, hq=16, hkv=4,
                batch=16, sk=4096, mode="paged", num_splits=3, page_size=16,
            ),
            2,
            id="paged-b16-hkv4-request3-caps2",
        ),
        pytest.param(
            _static_split_case(
                "static-occupancy-cap1", 771, d=128, hq=16, hkv=4,
                batch=17, sk=4096, mode="paged", num_splits=3, page_size=16,
            ),
            1,
            id="paged-b17-hkv4-request3-caps1",
        ),
        pytest.param(
            _static_split_case("static-single", 741),
            1,
            id="direct-request1-remains-single",
        ),
    ),
)
def test_static_native_fp8_split_workspace_is_capped_to_n_tiles_and_occupancy(
    monkeypatch, case, expected_splits
):
    kwargs, expected = _arguments(case)
    result, allocations = _observe_split_workspace(monkeypatch, kwargs)

    observed_partial_shapes = [
        shape
        for shape in allocations
        if len(shape) == kwargs["q"].ndim + 1
        and shape[1:] == tuple(kwargs["q"].shape)
    ]
    partial_shape = (expected_splits, *kwargs["q"].shape)
    if expected_splits > 1:
        assert observed_partial_shapes == [partial_shape]
    else:
        assert observed_partial_shapes == []
    if case.num_splits != expected_splits:
        assert (case.num_splits, *kwargs["q"].shape) not in allocations
    _assert_result(result[0], result[1], expected, torch.bfloat16)


def test_dynamic_native_fp8_split_workspace_and_counts_are_not_capped(monkeypatch):
    case = _static_split_case(
        "dynamic-occupancy-unchanged", 731, d=128, hq=16, hkv=4, batch=17,
        mode="paged", num_splits=2, page_size=64,
    )
    kwargs, expected = _arguments(case)
    requested_counts = torch.tensor(
        (2, 1) * 8 + (2,), device="cuda", dtype=torch.int32
    )
    kwargs["num_splits_dynamic_ptr"] = requested_counts
    counts_before = requested_counts.clone()

    result, allocations = _observe_split_workspace(monkeypatch, kwargs)

    assert (case.num_splits, *kwargs["q"].shape) in allocations
    assert torch.equal(requested_counts, counts_before)
    _assert_result(result[0], result[1], expected, torch.bfloat16)


def test_contiguous_singleton_descales_compile_without_materialization(monkeypatch):
    batch, sq, sk, hq, hkv, d = 1, 5, 7, 4, 1, 64
    scales = tuple(
        _layout_scale(batch, hkv, offset, "contiguous") for offset in OFFSETS
    )
    assert all(
        scale.shape == (1, 1)
        and scale.stride() == (1, 1)
        and scale.is_contiguous()
        for scale in scales
    )
    data_ptrs = tuple(scale.data_ptr() for scale in scales)
    q, k, v = _make_qkv(
        batch, sq, sk, hq, hkv, d, d, scales, (681, 682, 683)
    )
    expected = _reference(q, k, v, scales, causal=False)
    original_to_cute_tensor = cute_interface.to_cute_tensor
    converted_descales = []
    compile_cache = _flash_attn_fwd.compile_cache
    original_contains = type(compile_cache).__contains__

    def force_compile(cache, key):
        if cache is compile_cache:
            return False
        return original_contains(cache, key)


    def observe_descale_conversion(t, *args, **kwargs):
        if any(t is scale for scale in scales):
            converted_descales.append((t, kwargs.copy()))
        return original_to_cute_tensor(t, *args, **kwargs)

    monkeypatch.setattr(cute_interface, "to_cute_tensor", observe_descale_conversion)
    monkeypatch.setattr(type(compile_cache), "__contains__", force_compile)
    result = _native_call(
        q, k, v, scales, causal=False, return_lse=True
    )

    assert len(converted_descales) == len(scales)
    assert all(
        converted is scale
        for (converted, _), scale in zip(converted_descales, scales)
    )
    assert all(
        kwargs["leading_dim"] == 1 and not kwargs.get("fully_dynamic", False)
        for _, kwargs in converted_descales
    )
    assert tuple(scale.data_ptr() for scale in scales) == data_ptrs
    assert all(scale.stride() == (1, 1) for scale in scales)
    _assert_result(result[0], result[1], expected, torch.bfloat16)


def test_descale_layouts_are_independent_in_real_forward():
    batch, sq, sk, hq, hkv, d = 2, 17, 29, 8, 2, 128
    scales = (
        _layout_scale(batch, hkv, OFFSETS[0], "scalar"),
        _layout_scale(batch, hkv, OFFSETS[1], "batch-broadcast"),
        _layout_scale(batch, hkv, OFFSETS[2], "head-broadcast"),
    )
    q, k, v = _make_qkv(
        batch, sq, sk, hq, hkv, d, d, scales, (701, 702, 703)
    )
    expected = _reference(q, k, v, scales, causal=True)
    result = _native_call(
        q, k, v, scales, causal=True, fp8_kv_dequant=False, return_lse=True
    )
    _assert_result(result[0], result[1], expected, torch.bfloat16)




@pytest.mark.parametrize(
    "layout,expected",
    [
        ("none", "none"),
        ("scalar", "scalar"),
        ("both-broadcast", "both_broadcast"),
        ("batch-broadcast", "batch_broadcast"),
        ("head-broadcast", "head_broadcast"),
        ("contiguous", "contiguous"),
    ],
)
def test_descale_semantic_classification(layout: str, expected: str) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    scale = _layout_scale(2, 3, 0.5, layout)
    assert _classify_descale_layout(scale, "scale", 2, 3, device) == expected


@pytest.mark.parametrize("bad_kind", ["rank", "shape", "dtype", "device", "transposed", "arbitrary"])
def test_native_fp8_rejects_descale_before_compilation(monkeypatch, bad_kind: str) -> None:
    batch, sq, sk, hq, hkv, d = 2, 5, 7, 4, 2, 64
    good = _layout_scale(batch, hkv, 0.5, "contiguous")
    if bad_kind == "rank":
        bad = torch.ones(hkv, device="cuda", dtype=torch.float32)
    elif bad_kind == "shape":
        bad = torch.ones(batch, hkv + 1, device="cuda", dtype=torch.float32)
    elif bad_kind == "dtype":
        bad = torch.ones(batch, hkv, device="cuda", dtype=torch.float16)
    elif bad_kind == "device":
        bad = torch.ones(batch, hkv, device="cpu", dtype=torch.float32)
    elif bad_kind == "transposed":
        bad = torch.ones(hkv, batch, device="cuda", dtype=torch.float32).t()
    else:
        bad = torch.empty(8, device="cuda", dtype=torch.float32).as_strided(
            (batch, hkv), (3, 1)
        )
    q, k, v = _make_qkv(
        batch, sq, sk, hq, hkv, d, d, (good,) * 3, (630, 631, 632)
    )
    cache_lookup_called = False

    def unexpected_cache_lookup(cache, key):
        nonlocal cache_lookup_called
        cache_lookup_called = True
        raise AssertionError("compile cache must not be queried")

    monkeypatch.setattr(
        type(_flash_attn_fwd.compile_cache), "__contains__", unexpected_cache_lookup
    )
    with pytest.raises(AssertionError):
        _native_call(q, k, v, (bad, good, good), causal=False)
    assert not cache_lookup_called


class _NegativeStrideDescale:
    ndim = 2
    shape = (2, 3)
    dtype = torch.float32
    device = torch.device("cuda", torch.cuda.current_device())
    is_cuda = True

    @staticmethod
    def stride():
        return (-3, 1)


def test_descale_negative_stride_rejected() -> None:
    with pytest.raises(AssertionError, match="negative strides"):
        _classify_descale_layout(
            _NegativeStrideDescale(), "scale", 2, 3, _NegativeStrideDescale.device
        )


def test_descale_compile_key_uses_layout_not_values(monkeypatch) -> None:
    batch, sq, sk, hq, hkv, d = 2, 5, 7, 4, 2, 64
    compile_cache = _flash_attn_fwd.compile_cache
    original_contains = type(compile_cache).__contains__
    looked_up_keys = []

    def observe_cache_lookup(cache, key):
        if cache is compile_cache:
            looked_up_keys.append(key)
        return original_contains(cache, key)

    monkeypatch.setattr(type(compile_cache), "__contains__", observe_cache_lookup)

    def run(offset: float, layouts: tuple[str, str, str]):
        scales = [
            _layout_scale(batch, hkv, offset + delta, layout)
            for delta, layout in zip((0.0, 0.1, 0.2), layouts)
        ]
        q, k, v = _make_qkv(
            batch, sq, sk, hq, hkv, d, d, scales, (640, 641, 642)
        )
        lookups_before = len(looked_up_keys)
        _native_call(q, k, v, scales, causal=False)
        new_lookup_keys = looked_up_keys[lookups_before:]
        assert new_lookup_keys
        assert all(key == new_lookup_keys[0] for key in new_lookup_keys)
        return new_lookup_keys[0]

    layouts = (
        "none",
        "scalar",
        "both-broadcast",
        "batch-broadcast",
        "head-broadcast",
        "contiguous",
    )
    baseline = ("contiguous",) * 3
    baseline_key = run(0.3, baseline)
    assert run(0.7, baseline) == baseline_key

    separated_keys = {baseline_key}
    for slot in range(3):
        slot_keys = []
        for layout in layouts:
            candidate = list(baseline)
            candidate[slot] = layout
            slot_keys.append(run(0.7, tuple(candidate)))
        assert len(set(slot_keys)) == len(layouts)
        separated_keys.update(slot_keys)
    assert len(separated_keys) == 1 + 3 * (len(layouts) - 1)


def _generic_e5m2_inputs(requires_grad: bool = False):
    q = torch.zeros(
        2,
        5,
        4,
        64,
        dtype=torch.float8_e5m2,
        device="cuda",
        requires_grad=requires_grad,
    )
    k = torch.zeros(
        2,
        7,
        2,
        64,
        dtype=torch.float8_e5m2,
        device="cuda",
        requires_grad=requires_grad,
    )
    v = torch.zeros_like(k, requires_grad=requires_grad)
    return q, k, v


def test_generic_e5m2_descale_validation_precedes_compile(monkeypatch):
    q, k, v = _generic_e5m2_inputs()
    bad_descale = torch.ones(2, 1, dtype=torch.float32, device="cuda")
    compile_called = False

    def unexpected_compile(*args, **kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("compile must not be invoked")

    monkeypatch.setattr(cute_interface.cute, "compile", unexpected_compile)
    with pytest.raises(AssertionError, match=r"q_descale shape.*expected \(2, 2\)"):
        _flash_attn_fwd(
            q,
            k,
            v,
            q_descale=bad_descale,
            _arch=100,
        )
    assert not compile_called


def test_generic_e5m2_backward_remains_rejected_before_compile(monkeypatch):
    q, k, v = _generic_e5m2_inputs(requires_grad=True)
    descale = torch.ones(2, 2, dtype=torch.float32, device="cuda")
    compile_called = False

    def unexpected_compile(*args, **kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("compile must not be invoked")

    monkeypatch.setattr(cute_interface.cute, "compile", unexpected_compile)
    with pytest.raises(NotImplementedError, match="FP8 backward is not supported"):
        _flash_attn_fwd(
            q,
            k,
            v,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            _arch=100,
        )
    assert not compile_called


@pytest.mark.parametrize(
    "torch_dtype,cute_dtype,ffi_dtype",
    (
        pytest.param(
            torch.float8_e4m3fn,
            cute_interface.cutlass.Float8E4M3FN,
            cute_interface.cutlass.Uint8,
            id="e4m3-uint8-ffi",
        ),
        pytest.param(
            torch.float8_e5m2,
            cute_interface.cutlass.Float8E5M2,
            cute_interface.cutlass.Uint8,
            id="e5m2-uint8-ffi",
        ),
        pytest.param(
            torch.bfloat16,
            cute_interface.cutlass.BFloat16,
            cute_interface.cutlass.BFloat16,
            id="bf16-unchanged",
        ),
        pytest.param(
            torch.int32,
            cute_interface.cutlass.Int32,
            cute_interface.cutlass.Int32,
            id="int32-unchanged",
        ),
    ),
)
def test_compile_only_tensor_uses_storage_ffi_and_semantic_cute_dtype(
    monkeypatch, torch_dtype, cute_dtype, ffi_dtype
):
    constructed = []

    class ReadOnlyFakeTensor:
        def __init__(self, element_type):
            self._element_type = element_type

        @property
        def element_type(self):
            return self._element_type

    def make_dynamic_fake(element_type, shape, **kwargs):
        constructed.append((element_type, shape, kwargs))
        return ReadOnlyFakeTensor(element_type)

    monkeypatch.setattr(cute_dsl_utils, "fake_tensor", make_dynamic_fake)
    spec = cute_interface._CompileOnlyTensorSpec((2, 3), torch_dtype)

    tensor = cute_dsl_utils.to_cute_tensor(spec, assumed_align=16, leading_dim=1)

    assert spec.cute_element_type == cute_dtype
    assert spec.ffi_element_type == ffi_dtype
    assert len(constructed) == 1
    assert constructed[0][0] == cute_dtype
    assert tensor.element_type == cute_dtype
    if ffi_dtype != cute_dtype:
        assert tensor._tvm_ffi_tensor.dtype == cute_dsl_utils.tvm_ffi.dtype("uint8")
        assert tensor._tvm_ffi_tensor.device.type == "cuda"
        assert tensor._tvm_ffi_tensor.device.index == 0
    else:
        assert not hasattr(tensor, "_tvm_ffi_tensor")


class _CompiledKernelMustNotDispatch:
    def __init__(self, spec):
        self.spec = spec
        self.dispatches = 0

    def __call__(self, *args, **kwargs):
        self.dispatches += 1
        raise AssertionError("compile-only API must not dispatch a compiled kernel")




def _mock_compile_only_caches(monkeypatch):
    forward_cache = {}
    combine_cache = {}
    compiled = []

    def compile_without_dispatch(*args, **kwargs):
        kernel = _CompiledKernelMustNotDispatch(args[0])
        compiled.append(kernel)
        return kernel

    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", forward_cache)
    monkeypatch.setattr(
        cute_interface._flash_attn_fwd_combine, "compile_cache", combine_cache
    )
    monkeypatch.setattr(cute_interface.cute, "compile", compile_without_dispatch)
    monkeypatch.setattr(cute_interface, "_get_device_arch", lambda: 90)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(multi_processor_count=132),
    )
    return forward_cache, combine_cache, compiled


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_compile_only_paged_scalar_descale_static_split_compiles_combine(
    monkeypatch, out_dtype
):
    forward_cache, combine_cache, compiled = _mock_compile_only_caches(monkeypatch)

    result = cute_interface.compile_flash_attn_varlen_func_from_specs(
        q_shape=(1, 8, 128),
        k_shape=(256, 16, 2, 128),
        v_shape=(256, 16, 2, 128),
        q_dtype=FP8,
        out_dtype=out_dtype,
        cu_seqlens_q_shape=(2,),
        seqused_k_shape=(1,),
        seqused_k_stride=(1,),
        page_table_shape=(1, 256),
        page_table_stride=(256, 1),
        q_descale_shape=(),
        q_descale_stride=(),
        k_descale_shape=(),
        k_descale_stride=(),
        v_descale_shape=(),
        v_descale_stride=(),
        max_seqlen_q=1,
        max_seqlen_k=4096,
        num_splits=2,
        return_lse=True,
    )

    assert result[0].shape == (1, 8, 128)
    assert result[0].dtype == out_dtype
    assert len(forward_cache) == 1
    assert len(combine_cache) == 1
    assert len(compiled) == 2
    assert all(kernel.dispatches == 0 for kernel in compiled)
    forward = compiled[0].spec
    assert forward.is_split_kv
    assert forward.native_fp8
    assert not forward.fp8_kv_dequant
    assert forward.o_dtype == cute_interface.torch2cute_dtype_map[out_dtype]
    combine = compiled[1].spec
    assert not combine.skip_single_split
    assert combine.compact_varlen_grid
    assert combine.num_threads == 256


def test_compile_only_dynamic_diffkv_compiles_forward_and_combine(monkeypatch):
    forward_cache, combine_cache, compiled = _mock_compile_only_caches(monkeypatch)
    forwarded = {}
    original_forward = cute_interface._flash_attn_fwd

    def observe_forward(**kwargs):
        forwarded.update(kwargs)
        return original_forward(**kwargs)

    observe_forward.compile_cache = forward_cache
    monkeypatch.setattr(cute_interface, "_flash_attn_fwd", observe_forward)

    result = cute_interface.compile_flash_attn_varlen_func_from_specs(
        q_shape=(1, 16, 192),
        k_shape=(7, 16, 2, 192),
        v_shape=(7, 16, 2, 128),
        q_dtype=FP8,
        out_dtype=torch.bfloat16,
        v_stride=(10240, 640, 320, 1),
        cu_seqlens_q_shape=(2,),
        seqused_k_shape=(1,),
        seqused_k_stride=(1,),
        page_table_shape=(1, 7),
        page_table_stride=(7, 1),
        num_splits_dynamic_ptr_shape=(1,),
        num_splits_dynamic_ptr_stride=(1,),
        learnable_sink_shape=(16,),
        learnable_sink_stride=(1,),
        dynamic_scheduler_counter_shape=(1,),
        dynamic_scheduler_counter_stride=(1,),
        q_descale_shape=(),
        k_descale_shape=(),
        v_descale_shape=(),
        max_seqlen_q=1,
        max_seqlen_k=100,
        window_size=(127, 0),
        num_splits=32,
    )

    assert result[0].shape == (1, 16, 128)
    assert forwarded["learnable_sink"].shape == (16,)
    assert forwarded["learnable_sink"].dtype == torch.bfloat16
    assert forwarded["learnable_sink"].stride() == (1,)
    assert forwarded["dynamic_scheduler_counter"].shape == (1,)
    assert forwarded["dynamic_scheduler_counter"].dtype == torch.int32
    assert forwarded["dynamic_scheduler_counter"].stride() == (1,)
    assert forwarded["num_splits_dynamic_ptr"].shape == (1,)
    assert forwarded["num_splits_dynamic_ptr"].stride() == (1,)
    assert forwarded["v"].shape == (7, 16, 2, 128)
    assert forwarded["v"].stride() == (10240, 640, 320, 1)
    assert forwarded["out"].shape == (1, 16, 128)
    assert forwarded["compile_only"] is True
    assert len(forward_cache) == 1
    assert len(combine_cache) == 1
    assert len(compiled) == 2
    assert all(kernel.dispatches == 0 for kernel in compiled)
    combine = compiled[1].spec
    assert combine.skip_single_split
    assert combine.compact_varlen_grid
    assert combine.num_threads == 128


def test_compile_only_unsplit_does_not_compile_combine(monkeypatch):
    forward_cache, combine_cache, compiled = _mock_compile_only_caches(monkeypatch)

    cute_interface.compile_flash_attn_varlen_func_from_specs(
        q_shape=(1, 8, 128),
        k_shape=(16, 128, 2, 128),
        v_shape=(16, 128, 2, 128),
        q_dtype=FP8,
        out_dtype=torch.bfloat16,
        cu_seqlens_q_shape=(2,),
        seqused_k_shape=(1,),
        page_table_shape=(1, 16),
        q_descale_shape=(),
        k_descale_shape=(),
        v_descale_shape=(),
        max_seqlen_q=1,
        max_seqlen_k=2048,
        num_splits=1,
    )

    assert len(forward_cache) == 1
    assert combine_cache == {}
    assert len(compiled) == 1
    assert compiled[0].dispatches == 0


def test_compile_only_multibatch_scheduler_five_neighbors(monkeypatch):
    forward_cache, combine_cache, compiled = _mock_compile_only_caches(monkeypatch)
    neighbors = (
        ("native-diffkv-static", FP8, 192, 128, False, False),
        ("non-native-diffkv-dynamic", torch.bfloat16, 192, 128, False, True),
        ("native-d192-dynamic", FP8, 192, 192, False, True),
        ("native-d128-dynamic", FP8, 128, 128, False, True),
        ("native-diffkv-dynamic-split", FP8, 192, 128, True, True),
    )

    def compile_neighbor(q_dtype, head_dim, head_dim_v, dynamic_split):
        kwargs = dict(
            q_shape=(2048, 16, head_dim),
            k_shape=(32, 16, 2, head_dim),
            v_shape=(32, 16, 2, head_dim_v),
            q_dtype=q_dtype,
            cu_seqlens_q_shape=(17,),
            seqused_k_shape=(16,),
            seqused_k_stride=(1,),
            page_table_shape=(16, 32),
            page_table_stride=(32, 1),
            max_seqlen_q=128,
            max_seqlen_k=129,
            num_splits=32 if dynamic_split else 1,
        )
        if q_dtype == FP8:
            kwargs.update(
                out_dtype=torch.bfloat16,
                q_descale_shape=(),
                q_descale_stride=(),
                k_descale_shape=(),
                k_descale_stride=(),
                v_descale_shape=(),
                v_descale_stride=(),
            )
        if dynamic_split:
            kwargs.update(
                num_splits_dynamic_ptr_shape=(16,),
                num_splits_dynamic_ptr_stride=(1,),
            )
        cute_interface.compile_flash_attn_varlen_func_from_specs(**kwargs)

    for index, (
        name, q_dtype, head_dim, head_dim_v, dynamic_split,
        expected_dynamic_varlen,
    ) in enumerate(neighbors, start=1):
        compiled_before = len(compiled)
        compile_neighbor(q_dtype, head_dim, head_dim_v, dynamic_split)
        assert len(forward_cache) == index, name
        assert len(compiled) == compiled_before + 1 + int(dynamic_split), name
        forward = compiled[compiled_before].spec
        assert forward.dtype == cute_interface.torch2cute_dtype_map[q_dtype], name
        assert forward.tile_hdim == head_dim, name
        assert forward.tile_hdimv == head_dim_v, name
        assert forward.is_split_kv == dynamic_split, name
        assert not forward.use_persistent_varlen, name
        assert forward.use_dynamic_varlen == expected_dynamic_varlen, name
        assert forward.use_dynamic_splits == dynamic_split, name
        assert forward.persistent_scheduler_sm_count == (
            132 if expected_dynamic_varlen else None
        ), name

        cache_sizes = (len(forward_cache), len(combine_cache), len(compiled))
        compile_neighbor(q_dtype, head_dim, head_dim_v, dynamic_split)
        assert (len(forward_cache), len(combine_cache), len(compiled)) == cache_sizes

    assert len(forward_cache) == len(neighbors)
    assert len(combine_cache) == 1
    assert len(compiled) == len(neighbors) + 1
    assert all(kernel.dispatches == 0 for kernel in compiled)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"seqused_k_stride": (1,)}, "stride requires a shape"),
        (
            {"seqused_k_shape": (1,), "seqused_k_stride": (1, 1)},
            "stride rank 2 != shape rank 1",
        ),
        (
            {"seqused_k_shape": (1,), "seqused_k_stride": (2,)},
            "sequence length tensors must be contiguous",
        ),
        (
            {"page_table_shape": (1, 16), "page_table_stride": (16, 2)},
            "page_table must be contiguous in the last dimension",
        ),
        (
            {"q_descale_shape": (1,), "q_descale_stride": (1,)},
            "q_descale must be rank 0 or rank 2",
        ),
        (
            {"learnable_sink_shape": (1, 8)},
            "learnable_sink must be rank 1",
        ),
        (
            {"dynamic_scheduler_counter_shape": (2,)},
            r"dynamic_scheduler_counter must have shape \(1,\)",
        ),
        (
            {
                "dynamic_scheduler_counter_shape": (1,),
                "dynamic_scheduler_counter_stride": (2,),
            },
            "dynamic_scheduler_counter must be contiguous",
        ),
    ],
)
def test_compile_only_metadata_and_stride_validation(monkeypatch, kwargs, match):
    _mock_compile_only_caches(monkeypatch)
    base = dict(
        q_shape=(1, 8, 128),
        k_shape=(16, 128, 2, 128),
        v_shape=(16, 128, 2, 128),
        q_dtype=FP8,
        out_dtype=torch.bfloat16,
        cu_seqlens_q_shape=(2,),
        max_seqlen_q=1,
        max_seqlen_k=2048,
    )
    base.update(kwargs)
    with pytest.raises(AssertionError, match=match):
        cute_interface.compile_flash_attn_varlen_func_from_specs(**base)
