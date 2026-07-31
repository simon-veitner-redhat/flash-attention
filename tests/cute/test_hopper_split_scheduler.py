import pytest
import torch

from flash_attn.cute.split_scheduler import plan_hopper_split_schedule
from flash_attn.cute.interface import _flash_attn_fwd


def _plan(monkeypatch, **overrides):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Props", (), {"multi_processor_count": 132})(),
    )
    args = {
        "query_start_loc_cpu": torch.arange(5, dtype=torch.int32),
        "seq_lens_cpu": torch.full((4,), 4096, dtype=torch.int32),
        "device": torch.device("cuda"),
        "num_heads_q": 16,
        "num_heads_kv": 8,
        "head_dim": 256,
        "head_dim_v": 256,
        "has_qv": False,
        "cp_world_size": 1,
    }
    args.update(overrides)
    return plan_hopper_split_schedule(**args)


@pytest.mark.parametrize("major", [10, 11, 12])
def test_split_planner_is_hopper_only(monkeypatch, major):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device: (major, 0)
    )
    assert (
        plan_hopper_split_schedule(
            object(),
            object(),
            device=torch.device("cuda"),
            num_heads_q=16,
            num_heads_kv=8,
            head_dim=256,
            head_dim_v=256,
            has_qv=False,
            cp_world_size=1,
        )
        is None
    )


def test_standard_d256_split_planner_matches_fa3(monkeypatch):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([512, 4096], dtype=torch.int32),
    )
    assert plan is not None
    assert plan.num_splits == 13
    assert plan.split_counts == [2, 13]


@pytest.mark.parametrize(
    ("query_start_loc_cpu", "seq_lens_cpu", "split_counts"),
    (
        (
            [0, 257, 258, 259, 260, 261, 262, 263, 264, 268, 272, 276, 405, 534],
            [257, 1057, 1057, 1057, 1057, 1057, 1057, 1057, 2051, 2051, 2051, 4093, 4093],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
        ),
        (
            [0, 769, 770, 771, 772, 780, 788, 1045],
            [769, 6143, 6143, 6143, 3073, 3073, 8191],
            [1, 2, 2, 2, 1, 1, 2],
        ),
    ),
)
def test_standard_d256_mixed_split_planner_matches_fa3(
    monkeypatch, query_start_loc_cpu, seq_lens_cpu, split_counts
):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor(query_start_loc_cpu, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens_cpu, dtype=torch.int32),
    )
    assert plan is not None
    assert plan.num_splits == 2
    assert plan.split_counts == split_counts


def test_mla_split_planner_rejects_multitoken_queries(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            query_start_loc_cpu=torch.tensor([0, 1, 5], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([512, 4096], dtype=torch.int32),
            num_heads_q=16,
            num_heads_kv=1,
            head_dim=64,
            head_dim_v=256,
            has_qv=True,
        )
        is None
    )


def test_standard_d256_split_planner_rejects_pure_prefill(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            query_start_loc_cpu=torch.tensor([0, 257, 514], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4093, 4093], dtype=torch.int32),
        )
        is None
    )


def test_standard_d256_single_request_uses_static_actual_split(monkeypatch):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([4096], dtype=torch.int32),
    )
    assert plan is not None
    assert plan.num_splits == 13
    assert plan.split_counts is None


def test_standard_d256_single_request_graph_plan_is_dynamic(monkeypatch):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        cuda_graph_max_num_splits=32,
    )
    assert plan is not None
    assert plan.num_splits == 15
    assert plan.split_counts == [1]


def test_mla_single_request_graph_plan_keeps_configured_bound(monkeypatch):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        num_heads_q=16,
        num_heads_kv=1,
        head_dim=64,
        head_dim_v=256,
        has_qv=True,
        cuda_graph_max_num_splits=32,
    )
    assert plan is not None
    assert plan.num_splits == 32
    assert plan.split_counts == [1]


@pytest.mark.parametrize("has_qv", [False, True], ids=("mha", "mla"))
def test_graph_split_bound_one_returns_no_dynamic_metadata(
    monkeypatch, has_qv
):
    plan = _plan(
        monkeypatch,
        num_heads_q=16,
        num_heads_kv=1 if has_qv else 8,
        head_dim=64 if has_qv else 256,
        head_dim_v=256,
        has_qv=has_qv,
        cuda_graph_max_num_splits=1,
    )
    assert plan is None


def test_mla_dv512_split_planner_keeps_short_heterogeneous_batches(monkeypatch):
    result = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.arange(5, dtype=torch.int32),
        seq_lens_cpu=torch.tensor([512, 1024, 2048, 4096], dtype=torch.int32),
        num_heads_q=16,
        num_heads_kv=1,
        head_dim=64,
        head_dim_v=512,
        has_qv=True,
    )
    assert result is not None
    assert result.split_counts is not None
    assert max(result.split_counts) > 1


def test_mla_dv512_split_planner_rejects_medium_homogeneous_batches(
    monkeypatch,
):
    assert (
        _plan(
            monkeypatch,
            num_heads_q=16,
            num_heads_kv=1,
            head_dim=64,
            head_dim_v=512,
            has_qv=True,
        )
        is None
    )


def test_mla_dv512_h64_batch16_short_uses_static_four(monkeypatch):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.arange(17, dtype=torch.int32),
        seq_lens_cpu=torch.full((16,), 512, dtype=torch.int32),
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=64,
        head_dim_v=512,
        has_qv=True,
    )
    assert plan is not None
    assert plan.num_splits == 4
    assert plan.split_counts is None


@pytest.mark.parametrize(
    ("seqlen_k", "split_count"), ((1, 1), (512, 4), (4096, 8))
)
def test_mla_dv512_h64_batch16_graph_plan_is_stable(
    monkeypatch, seqlen_k, split_count
):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.arange(17, dtype=torch.int32),
        seq_lens_cpu=torch.full((16,), seqlen_k, dtype=torch.int32),
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=64,
        head_dim_v=512,
        has_qv=True,
        cuda_graph_max_num_splits=32,
    )
    assert plan is not None
    assert plan.num_splits == 8
    assert plan.split_counts == [split_count] * 16


@pytest.mark.parametrize(
    ("seqlen_k", "split_count"), ((1, 1), (4096, 32), (8192, 32))
)
def test_mla_dv512_h16_batch1_graph_plan_is_stable(
    monkeypatch, seqlen_k, split_count
):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([seqlen_k], dtype=torch.int32),
        num_heads_q=16,
        num_heads_kv=1,
        head_dim=64,
        head_dim_v=512,
        has_qv=True,
        cuda_graph_max_num_splits=32,
    )
    assert plan is not None
    assert plan.num_splits == 32
    assert plan.split_counts == [split_count]


@pytest.mark.parametrize(
    (
        "batch_size",
        "num_heads_q",
        "seqlen_k",
        "num_splits",
        "split_count",
    ),
    (
        (8, 64, 4096, 15, 13),
        (16, 32, 512, 8, 4),
        (32, 64, 512, 4, 3),
        (64, 16, 512, 4, 2),
        (128, 128, 32768, 4, 1),
    ),
)
def test_mla_dv512_graph_plan_is_shape_stable(
    monkeypatch,
    batch_size,
    num_heads_q,
    seqlen_k,
    num_splits,
    split_count,
):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.arange(batch_size + 1, dtype=torch.int32),
        seq_lens_cpu=torch.full(
            (batch_size,), seqlen_k, dtype=torch.int32
        ),
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        head_dim=64,
        head_dim_v=512,
        has_qv=True,
        cuda_graph_max_num_splits=32,
    )
    assert plan is not None
    assert plan.num_splits == num_splits
    assert plan.split_counts == [split_count] * batch_size


@pytest.mark.parametrize("cuda_graph_max_num_splits", range(1, 8))
def test_mla_dv512_h64_batch16_short_respects_graph_split_bound(
    monkeypatch, cuda_graph_max_num_splits
):
    assert (
        _plan(
            monkeypatch,
            query_start_loc_cpu=torch.arange(17, dtype=torch.int32),
            seq_lens_cpu=torch.full((16,), 512, dtype=torch.int32),
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=64,
            head_dim_v=512,
            has_qv=True,
            cuda_graph_max_num_splits=cuda_graph_max_num_splits,
        )
        is None
    )


def test_mla_dv512_h16_batch1_respects_graph_split_bound(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
            num_heads_q=16,
            num_heads_kv=1,
            head_dim=64,
            head_dim_v=512,
            has_qv=True,
            cuda_graph_max_num_splits=1,
        )
        is None
    )


@pytest.mark.parametrize(
    ("batch_size", "num_heads_q", "seqlen_k"),
    ((8, 64, 512), (32, 64, 512), (16, 32, 512)),
)
def test_mla_dv512_h64_batch16_short_static_four_is_exact(
    monkeypatch, batch_size, num_heads_q, seqlen_k
):
    assert (
        _plan(
            monkeypatch,
            query_start_loc_cpu=torch.arange(
                batch_size + 1, dtype=torch.int32
            ),
            seq_lens_cpu=torch.full(
                (batch_size,), seqlen_k, dtype=torch.int32
            ),
            num_heads_q=num_heads_q,
            num_heads_kv=1,
            head_dim=64,
            head_dim_v=512,
            has_qv=True,
        )
        is None
    )


@pytest.mark.parametrize("major", [10, 11, 12])
def test_mla_dv512_h64_batch16_short_static_four_is_hopper_only(
    monkeypatch, major
):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device: (major, 0)
    )
    assert (
        plan_hopper_split_schedule(
            torch.arange(17, dtype=torch.int32),
            torch.full((16,), 512, dtype=torch.int32),
            device=torch.device("cuda"),
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=64,
            head_dim_v=512,
            has_qv=True,
            cp_world_size=1,
            cuda_graph_max_num_splits=32,
        )
        is None
    )


@pytest.mark.parametrize("major", [10, 11, 12])
def test_mla_dv512_graph_stable_plans_are_hopper_only(monkeypatch, major):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device: (major, 0)
    )
    for batch_size, num_heads_q in (
        (1, 16),
        (8, 64),
        (16, 32),
        (16, 64),
        (32, 128),
        (64, 96),
        (128, 128),
    ):
        assert (
            plan_hopper_split_schedule(
                torch.arange(batch_size + 1, dtype=torch.int32),
                torch.ones(batch_size, dtype=torch.int32),
                device=torch.device("cuda"),
                num_heads_q=num_heads_q,
                num_heads_kv=1,
                head_dim=64,
                head_dim_v=512,
                has_qv=True,
                cp_world_size=1,
                cuda_graph_max_num_splits=32,
            )
            is None
        )


IS_SM90 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


def _reference(q, k, v, seqlens):
    outputs = []
    scale = q.shape[-1] ** -0.5
    for batch_idx, seqlen in enumerate(seqlens.tolist()):
        q_cur = q[batch_idx].float()
        k_cur = k[batch_idx, :seqlen].float()
        v_cur = v[batch_idx, :seqlen].float()
        scores = torch.einsum("hd,khd->hk", q_cur, k_cur) * scale
        outputs.append(
            torch.einsum("hk,khd->hd", scores.softmax(dim=-1), v_cur)
        )
    return torch.stack(outputs)


def _mixed_query_reference(q, k, v, q_lens, kv_lens):
    outputs = []
    q_offset = 0
    scale = q.shape[-1] ** -0.5
    qheads_per_kvhead = q.shape[1] // k.shape[2]
    for batch_idx, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        q_cur = q[q_offset : q_offset + q_len].float()
        k_cur = k[batch_idx, :kv_len].float().repeat_interleave(
            qheads_per_kvhead, dim=1
        )
        v_cur = v[batch_idx, :kv_len].float().repeat_interleave(
            qheads_per_kvhead, dim=1
        )
        scores = torch.einsum("qhd,khd->qhk", q_cur, k_cur) * scale
        causal = torch.arange(kv_len, device=q.device)[None, :] <= (
            torch.arange(q_len, device=q.device)[:, None] + kv_len - q_len
        )
        scores.masked_fill_(~causal[:, None, :], float("-inf"))
        outputs.append(
            torch.einsum("qhk,khd->qhd", scores.softmax(dim=-1), v_cur)
        )
        q_offset += q_len
    return torch.cat(outputs)


def _mla_reference(q, qv, k, v, seqlens):
    outputs = []
    scale = (q.shape[-1] + qv.shape[-1]) ** -0.5
    for batch_idx, seqlen in enumerate(seqlens.tolist()):
        q_cur = q[batch_idx].float()
        qv_cur = qv[batch_idx].float()
        k_cur = k[batch_idx, :seqlen, 0].float()
        v_cur = v[batch_idx, :seqlen, 0].float()
        scores = (
            torch.einsum("hd,kd->hk", q_cur, k_cur)
            + torch.einsum("hd,kd->hk", qv_cur, v_cur)
        ) * scale
        outputs.append(
            torch.einsum("hk,kd->hd", scores.softmax(dim=-1), v_cur)
        )
    return torch.stack(outputs)


@pytest.mark.skipif(not IS_SM90, reason="Hopper-only mixed SplitKV schedule")
def test_standard_d256_mixed_query_split_schedule_cuda_graph_replay():
    torch.manual_seed(3)
    device = "cuda"
    dtype = torch.bfloat16
    q_lens = [65, 1, 4]
    kv_lens = [129, 257, 129]
    batch_size, max_seqlen_k = len(q_lens), max(kv_lens)
    num_heads, num_heads_kv, head_dim = 4, 2, 256
    block_size = 32
    blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size

    q = torch.randn(
        sum(q_lens), num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size,
        blocks_per_seq * block_size,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    k_cache = k.flatten(0, 1).view(
        -1, block_size, num_heads_kv, head_dim
    )
    v_cache = v.flatten(0, 1).view_as(k_cache)
    page_table = torch.arange(
        batch_size * blocks_per_seq, device=device, dtype=torch.int32
    ).view(batch_size, blocks_per_seq)
    cu_seqlens_q = torch.tensor(
        [0, 65, 66, 70], device=device, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    dynamic_splits = torch.tensor(
        [1, 2, 1], device=device, dtype=torch.int32
    )

    def run():
        return _flash_attn_fwd(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            causal=True,
            num_splits=2,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()
    graph.replay()
    torch.cuda.synchronize()

    expected = _mixed_query_reference(q, k, v, q_lens, kv_lens)
    torch.testing.assert_close(
        graph_out.float(), expected, atol=3e-2, rtol=3e-2
    )


@pytest.mark.skipif(not IS_SM90, reason="Hopper-only FA-owned split schedule")
def test_fa_owned_split_schedule_cuda_graph_replay():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, max_seqlen_k = 4, 4096
    num_heads, num_heads_kv, head_dim = 16, 8, 256
    block_size = 128
    blocks_per_seq = max_seqlen_k // block_size

    q = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size,
        max_seqlen_k,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    page_table = torch.arange(
        batch_size * blocks_per_seq, device=device, dtype=torch.int32
    ).view(batch_size, blocks_per_seq)
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    seqused_k = torch.full(
        (batch_size,), max_seqlen_k, device=device, dtype=torch.int32
    )
    query_start_loc_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
    initial_plan = plan_hopper_split_schedule(
        query_start_loc_cpu,
        seqused_k.cpu(),
        device=torch.device(device),
        num_heads_q=num_heads,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        head_dim_v=head_dim,
        has_qv=False,
        cp_world_size=1,
        cuda_graph_max_num_splits=32,
    )
    assert initial_plan is not None and initial_plan.split_counts is not None
    dynamic_splits = torch.tensor(
        initial_plan.split_counts,
        device=device,
        dtype=torch.int32,
    )

    def run():
        return _flash_attn_fwd(
            q,
            k.flatten(0, 1).view(-1, block_size, num_heads_kv, head_dim),
            v.flatten(0, 1).view(-1, block_size, num_heads_kv, head_dim),
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            num_splits=32,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()

    for lengths in ([512, 4096, 1024, 2048], [4096, 256, 3072, 128]):
        lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
        seqused_k.copy_(lengths_cpu.to(device))
        replay_plan = plan_hopper_split_schedule(
            query_start_loc_cpu,
            lengths_cpu,
            device=torch.device(device),
            num_heads_q=num_heads,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim,
            has_qv=False,
            cp_world_size=1,
            cuda_graph_max_num_splits=32,
        )
        assert replay_plan is not None and replay_plan.split_counts is not None
        dynamic_splits.copy_(
            torch.tensor(
                replay_plan.split_counts,
                device=device,
                dtype=torch.int32,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        k_mqa = k.repeat_interleave(num_heads // num_heads_kv, dim=2)
        v_mqa = v.repeat_interleave(num_heads // num_heads_kv, dim=2)
        expected = _reference(q, k_mqa, v_mqa, seqused_k)
        torch.testing.assert_close(
            graph_out.float(), expected, atol=3e-2, rtol=3e-2
        )


@pytest.mark.parametrize("head_dim_v", [256, 512])
@pytest.mark.skipif(not IS_SM90, reason="Hopper-only FA-owned MLA split schedule")
def test_fa_owned_mla_split_schedule_cuda_graph_replay(head_dim_v):
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    max_seqlen_k = 4096 if head_dim_v == 256 else 8192
    num_heads, head_dim = 16, 64
    block_size = 128
    blocks_per_seq = max_seqlen_k // block_size

    q = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=dtype
    )
    qv = torch.randn(
        batch_size, num_heads, head_dim_v, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size,
        max_seqlen_k,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        batch_size,
        max_seqlen_k,
        1,
        head_dim_v,
        device=device,
        dtype=dtype,
    )
    page_table = torch.arange(
        batch_size * blocks_per_seq, device=device, dtype=torch.int32
    ).view(batch_size, blocks_per_seq)
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    seqused_k = torch.tensor(
        [max_seqlen_k, max_seqlen_k // 2, 1024, 512],
        device=device,
        dtype=torch.int32,
    )
    query_start_loc_cpu = torch.arange(batch_size + 1, dtype=torch.int32)
    initial_plan = plan_hopper_split_schedule(
        query_start_loc_cpu,
        seqused_k.cpu(),
        device=torch.device(device),
        num_heads_q=num_heads,
        num_heads_kv=1,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        has_qv=True,
        cp_world_size=1,
        cuda_graph_max_num_splits=32,
    )
    assert initial_plan is not None and initial_plan.split_counts is not None
    dynamic_splits = torch.tensor(
        initial_plan.split_counts,
        device=device,
        dtype=torch.int32,
    )

    def run():
        return _flash_attn_fwd(
            q,
            k.flatten(0, 1).view(-1, block_size, 1, head_dim),
            v.flatten(0, 1).view(-1, block_size, 1, head_dim_v),
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=512,
            page_table=page_table,
            num_splits=32,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()

    for lengths in (
        [max_seqlen_k, max_seqlen_k // 2, 1024, 512],
        [512, 2048, max_seqlen_k, 1024],
    ):
        lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
        seqused_k.copy_(lengths_cpu.to(device))
        replay_plan = plan_hopper_split_schedule(
            query_start_loc_cpu,
            lengths_cpu,
            device=torch.device(device),
            num_heads_q=num_heads,
            num_heads_kv=1,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            has_qv=True,
            cp_world_size=1,
            cuda_graph_max_num_splits=32,
        )
        assert replay_plan is not None and replay_plan.split_counts is not None
        dynamic_splits.copy_(
            torch.tensor(
                replay_plan.split_counts,
                device=device,
                dtype=torch.int32,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = _mla_reference(q, qv, k, v, seqused_k)
        torch.testing.assert_close(
            graph_out.float(), expected, atol=4e-2, rtol=4e-2
        )


@pytest.mark.parametrize("head_dim_v", [256, 512])
@pytest.mark.skipif(
    not IS_SM90,
    reason="Hopper-only batch-one direct SplitKV schedule",
)
def test_batch_one_mla_split_schedule_cuda_graph_replay(head_dim_v):
    torch.manual_seed(2)
    device = "cuda"
    dtype = torch.bfloat16
    max_seqlen_k = 4096 if head_dim_v == 256 else 8192
    replay_lengths = (
        (max_seqlen_k, 512)
        if head_dim_v == 256
        else (max_seqlen_k, 4096)
    )
    num_heads, head_dim = 16, 64
    block_size = 128
    blocks_per_seq = max_seqlen_k // block_size

    q = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
    qv = torch.randn(
        1, num_heads, head_dim_v, device=device, dtype=dtype
    )
    k = torch.randn(
        1,
        max_seqlen_k,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        1,
        max_seqlen_k,
        1,
        head_dim_v,
        device=device,
        dtype=dtype,
    )
    page_table = torch.arange(
        blocks_per_seq, device=device, dtype=torch.int32
    ).view(1, blocks_per_seq)
    cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    seqused_k = torch.ones(1, device=device, dtype=torch.int32)
    dynamic_splits = torch.ones(1, device=device, dtype=torch.int32)

    def run():
        return _flash_attn_fwd(
            q,
            k.flatten(0, 1).view(-1, block_size, 1, head_dim),
            v.flatten(0, 1).view(-1, block_size, 1, head_dim_v),
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=1,
            page_table=page_table,
            num_splits=32,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()

    query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
    for length in replay_lengths:
        lengths_cpu = torch.tensor([length], dtype=torch.int32)
        replay_plan = plan_hopper_split_schedule(
            query_start_loc_cpu,
            lengths_cpu,
            device=torch.device(device),
            num_heads_q=num_heads,
            num_heads_kv=1,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            has_qv=True,
            cp_world_size=1,
            cuda_graph_max_num_splits=32,
        )
        assert replay_plan is not None
        assert replay_plan.split_counts is not None
        seqused_k.fill_(length)
        dynamic_splits.fill_(replay_plan.split_counts[0])
        graph.replay()
        torch.cuda.synchronize()
        expected = _mla_reference(q, qv, k, v, seqused_k)
        torch.testing.assert_close(
            graph_out.float(), expected, atol=4e-2, rtol=4e-2
        )
