"""Immutable SM90 native-E4M3 FA3/FA4 benchmark contract."""
import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from vllm.vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    get_scheduler_metadata,
)


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    seqlen_q: int
    seqlen_k: int
    heads_q: int
    heads_kv: int
    d: int
    dv: int
    causal: bool
    num_splits: int
    page_size: int | None = None
    window_size: tuple[int, int] = (-1, -1)


CASES = (
    Case("d64_mha_prefill", 4, 512, 512, 16, 16, 64, 64, False, 1),
    Case("d96_gqa_prefill_causal", 4, 512, 1024, 16, 4, 96, 96, True, 1),
    Case("d128_gqa_decode_split", 16, 1, 4096, 16, 4, 128, 128, True, 2),
    Case("d96_mha_prefill", 1, 1024, 1024, 32, 32, 96, 96, True, 1),
    Case("d96_mha_paged_decode_512", 16, 1, 512, 32, 32, 96, 96, True, 9, 16),
    Case("d96_mha_paged_decode_512_nosplit", 16, 1, 512, 32, 32, 96, 96, True, 1, 16),
    Case("d192_gqa_prefill", 2, 256, 1024, 16, 4, 192, 192, False, 1),
    Case("d256_mha_decode_split", 8, 1, 4096, 8, 8, 256, 256, True, 2),
    Case("d256_mha_paged_decode_128", 16, 1, 128, 16, 16, 256, 256, True, 9, 16),
    Case("d256_mha_paged_decode_512", 16, 1, 512, 16, 16, 256, 256, True, 9, 16),
    Case("d256_mha_paged64_decode_128", 16, 1, 128, 16, 16, 256, 256, True, 9, 64),
    Case("d256_mha_paged64_decode_512", 16, 1, 512, 16, 16, 256, 256, True, 9, 64),
    Case("d256_mha_paged_decode_128_nosplit", 16, 1, 128, 16, 16, 256, 256, True, 1, 16),
    Case("d256_mha_paged_decode_512_nosplit", 16, 1, 512, 16, 16, 256, 256, True, 1, 16),
    Case("diffkv_d192_dv128_gqa_prefill", 2, 256, 1024, 16, 4, 192, 128, True, 2),
    Case("diffkv_d192_dv128_gqa_decode", 16, 1, 4096, 16, 4, 192, 128, True, 2),
    Case("diffkv_mimo_prefill", 1, 400, 400, 16, 1, 192, 128, True, 1),
    Case("diffkv_mimo_decode", 32, 1, 288, 16, 1, 192, 128, True, 1, 16),
    Case("diffkv_mimo_swa_decode", 32, 1, 288, 16, 1, 192, 128, True, 1, 16, (127, 0)),
    Case("diffkv_mimo_swa_paged64_decode", 32, 1, 320, 16, 1, 192, 128, True, 1, 64, (127, 0)),
    Case("diffkv_mimo_swa_paged64_mixed", 16, 100, 192, 16, 1, 192, 128, True, 1, 64, (127, 0)),
    Case("diffkv_mimo_swa_paged64_decode_k64", 32, 1, 64, 16, 1, 192, 128, True, 1, 64, (127, 0)),
    Case("diffkv_mimo_swa_paged64_decode_k128", 32, 1, 128, 16, 1, 192, 128, True, 1, 64, (127, 0)),
    Case("diffkv_mimo_swa_paged64_decode_k192", 32, 1, 192, 16, 1, 192, 128, True, 1, 64, (127, 0)),
    Case("diffkv_mimo_swa_paged64_decode_k256", 32, 1, 256, 16, 1, 192, 128, True, 1, 64, (127, 0)),
)

DYNAMIC_CASES = (
    Case("dynamic_d128_gqa", 7, 1, 1025, 16, 4, 128, 128, True, 3, 32),
    Case("static_d128_gqa", 7, 1, 1025, 16, 4, 128, 128, True, 3, 32),
    Case("dynamic_d256_mha", 7, 1, 1025, 16, 16, 256, 256, True, 3, 16),
    Case("static_d256_mha", 7, 1, 1025, 16, 16, 256, 256, True, 3, 16),
    Case("dynamic_diffkv_gqa", 7, 1, 1025, 16, 4, 192, 128, True, 3, 64),
    Case("static_diffkv_gqa", 7, 1, 1025, 16, 4, 192, 128, True, 3, 64),
)
_K_LENS = (33, 127, 255, 386, 577, 769, 1025)
_SPLIT_COUNTS = (1, 1, 2, 3, 3, 3, 3)


def _scale(batch: int, heads_kv: int, base: float) -> torch.Tensor:
    b = torch.arange(batch, device="cuda", dtype=torch.float32)[:, None]
    h = torch.arange(heads_kv, device="cuda", dtype=torch.float32)[None, :]
    return (base + 0.07 * b + 0.11 * h).contiguous()
def _scaled_fp8(shape, scale, seed, q_heads=None):
    if q_heads is not None:
        scale = scale.repeat_interleave(q_heads // scale.shape[1], dim=1)
    source = torch.randn(
        shape, generator=torch.Generator(device="cuda").manual_seed(seed),
        device="cuda",
    ) * 0.7
    fp8 = torch.float8_e4m3fn
    return (source / scale[:, None, :, None]).clamp(
        torch.finfo(fp8).min, torch.finfo(fp8).max
    ).to(fp8)




def _canonical_inputs(case: Case):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randn((case.batch, case.seqlen_q, case.heads_q, case.d), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn((case.batch, case.seqlen_k, case.heads_kv, case.d), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn((case.batch, case.seqlen_k, case.heads_kv, case.dv), generator=generator, device="cuda").to(torch.float8_e4m3fn)
    scales = tuple(
        _scale(case.batch, case.heads_kv, base) for base in (0.37, 0.61, 0.89)
    )
    if case.name == "diffkv_mimo_prefill":
        assert all(
            scale.shape == (1, 1)
            and scale.stride() == (1, 1)
            and scale.is_contiguous()
            for scale in scales
        )
    return q, k, v, scales


def _dynamic_inputs(case: Case):
    scales = tuple(
        (
            base
            + 0.07 * torch.arange(case.batch, device="cuda", dtype=torch.float32)[:, None]
            + 0.03 * torch.arange(case.heads_kv, device="cuda", dtype=torch.float32)[None, :]
        ).contiguous()
        for base in (0.23, 0.41, 0.67)
    )
    specs = (
        ((case.batch, 1, case.heads_q, case.d), 42, case.heads_q),
        ((case.batch, case.seqlen_k, case.heads_kv, case.d), 43, None),
        ((case.batch, case.seqlen_k, case.heads_kv, case.dv), 44, None),
    )
    tensors = tuple(
        _scaled_fp8(shape, scale, seed, q_heads)
        for (shape, seed, q_heads), scale in zip(specs, scales)
    )
    return (*tensors, scales)


def _prepare(case: Case):
    dynamic = case.name.startswith(("dynamic_", "static_"))
    q, logical_k, logical_v, scales = _dynamic_inputs(case) if dynamic else _canonical_inputs(case)
    q_flat = q.flatten(0, 1)
    cu_q = torch.arange(0, (case.batch + 1) * case.seqlen_q, case.seqlen_q, device="cuda", dtype=torch.int32)
    if dynamic:
        k_lens = _K_LENS
    else:
        k_lens = (case.seqlen_k,) * case.batch
    seqused_k = torch.tensor(k_lens, device="cuda", dtype=torch.int32) if case.page_size else None
    if case.page_size is None:
        k, v = logical_k.flatten(0, 1), logical_v.flatten(0, 1)
        cu_k = torch.arange(0, (case.batch + 1) * case.seqlen_k, case.seqlen_k, device="cuda", dtype=torch.int32)
        block_table = None
    else:
        pages_per_seq = tuple(math.ceil(length / case.page_size) for length in k_lens)
        total_pages = sum(pages_per_seq)
        if dynamic:
            physical_ids = torch.randperm(total_pages, generator=torch.Generator(device="cpu").manual_seed(42)).to(device="cuda", dtype=torch.int32)
        else:
            physical_ids = torch.arange(total_pages, device="cuda", dtype=torch.int32)
        block_table = torch.full((case.batch, max(pages_per_seq)), -1, device="cuda", dtype=torch.int32)
        k = torch.zeros((total_pages, case.page_size, case.heads_kv, case.d), device="cuda", dtype=torch.float8_e4m3fn)
        v = torch.zeros((total_pages, case.page_size, case.heads_kv, case.dv), device="cuda", dtype=torch.float8_e4m3fn)
        cursor = 0
        for batch_idx, count in enumerate(pages_per_seq):
            row_ids = physical_ids[cursor:cursor + count]
            block_table[batch_idx, :count] = row_ids
            length = k_lens[batch_idx]
            for logical_page, physical_page in enumerate(row_ids.tolist()):
                begin = logical_page * case.page_size
                end = min(begin + case.page_size, length)
                k[physical_page, :end - begin].copy_(logical_k[batch_idx, begin:end])
                v[physical_page, :end - begin].copy_(logical_v[batch_idx, begin:end])
            cursor += count
        cu_k = None
    out3 = torch.empty((case.batch * case.seqlen_q, case.heads_q, case.dv), device="cuda", dtype=torch.bfloat16)
    out4 = torch.empty_like(out3)
    return q_flat, k, v, scales, cu_q, cu_k, seqused_k, block_table, out3, out4


def _median_graph_ms(fn, output: torch.Tensor, warmup: int, repeats: int) -> float:
    fn()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        fn()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record(); graph.replay(); end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def _bench_case(case: Case, warmup: int, repeats: int) -> tuple[float, float]:
    q, k, v, (qd, kd, vd), cuq, cuk, seqused, table, out3, out4 = _prepare(case)
    cache_lens = seqused if seqused is not None else torch.full((case.batch,), case.seqlen_k, device="cuda", dtype=torch.int32)
    fa3_packed_metadata = get_scheduler_metadata(
        case.batch, case.seqlen_q, case.seqlen_k, case.heads_q, case.heads_kv, case.d,
        cache_lens, qkv_dtype=q.dtype, headdim_v=case.dv, cu_seqlens_q=cuq,
        page_size=case.page_size, causal=case.causal, window_size=case.window_size,
        num_splits=case.num_splits,
    )
    dynamic_counts = torch.tensor(_SPLIT_COUNTS, device="cuda", dtype=torch.int32) if case.name.startswith("dynamic_") else None
    dynamic_scheduler_counter = torch.zeros((1,), device=q.device, dtype=torch.int32)
    common = dict(max_seqlen_q=case.seqlen_q, cu_seqlens_q=cuq, max_seqlen_k=case.seqlen_k,
                  cu_seqlens_k=cuk, seqused_k=seqused, block_table=table,
                  q_descale=qd, k_descale=kd, v_descale=vd, causal=case.causal,
                  window_size=case.window_size, softmax_scale=case.d ** -0.5)
    def fa3():
        return flash_attn_varlen_func(q, k, v, out=out3, fa_version=3,
            scheduler_metadata=fa3_packed_metadata, num_splits=case.num_splits, **common)
    def fa4():
        return flash_attn_varlen_func(
            q, k, v, out=out4, out_dtype=torch.bfloat16, fa_version=4,
            scheduler_metadata=dynamic_counts, num_splits=case.num_splits,
            dynamic_scheduler_counter=dynamic_scheduler_counter, **common
        )
    fa3_ms = _median_graph_ms(fa3, out3, warmup, repeats)
    fa4_ms = _median_graph_ms(fa4, out4, warmup, repeats)
    expected_shape = (case.batch * case.seqlen_q, case.heads_q, case.dv)
    assert out3.dtype == out4.dtype == torch.bfloat16
    assert out3.shape == out4.shape == expected_shape
    assert torch.isfinite(out3).all() and torch.isfinite(out4).all()
    torch.testing.assert_close(out4.float(), out3.float(), atol=0.30, rtol=0.20)
    return fa3_ms, fa4_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    parser.add_argument("--dynamic-case", action="append", choices=[case.name for case in DYNAMIC_CASES])
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("native E4M3 comparison requires an SM90 GPU")
    if args.case and args.dynamic_case:
        parser.error("--case and --dynamic-case are mutually exclusive")
    selected = DYNAMIC_CASES if args.dynamic_case else CASES
    requested = args.dynamic_case or args.case
    if requested:
        selected = tuple(case for case in selected if case.name in requested)
        if len(selected) != len(set(requested)):
            raise RuntimeError("selection did not resolve to one result per requested case")
    rows = []
    for case in selected:
        fa3_ms, fa4_ms = _bench_case(case, args.warmup, args.repeats)
        rows.append((case, fa3_ms, fa4_ms, fa4_ms / fa3_ms))
    if requested is None and len(rows) != 25:
        raise RuntimeError(f"canonical benchmark incomplete: expected 25 rows, got {len(rows)}")
    print("case,batch,seqlen_q,seqlen_k,heads_q,heads_kv,head_dim,head_dim_v,causal,num_splits,page_size,window_left,window_right,output_dtype,warmup,repeats,timing,fa3_median_ms,fa4_median_ms,fa4_over_fa3")
    for case, fa3_ms, fa4_ms, ratio in rows:
        page = "" if case.page_size is None else case.page_size
        print(f"{case.name},{case.batch},{case.seqlen_q},{case.seqlen_k},{case.heads_q},{case.heads_kv},{case.d},{case.dv},{int(case.causal)},{case.num_splits},{page},{case.window_size[0]},{case.window_size[1]},bfloat16,{args.warmup},{args.repeats},cuda_graph,{fa3_ms!r},{fa4_ms!r},{ratio!r}")


if __name__ == "__main__":
    main()
