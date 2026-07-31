from types import SimpleNamespace

import pytest
import torch

import flash_attn.cute.interface as interface
import flash_attn.cute.tile_scheduler as tile_scheduler
from flash_attn.cute.flash_fwd_sm90 import _use_paged_kv_overlap_sm90
from flash_attn.cute.interface import (
    _combine_max_seqlen_q_sm90,
    _flash_attn_fwd,
    _num_splits_sm90,
    _tile_size_fwd_sm90,
    _use_page16_paged_d64_loader_sm90,
    _use_wide_paged_d64_sm90,
    _use_batch_one_dynamic_split_varlen_scheduler_sm90,
    _use_dynamic_split_varlen_scheduler_sm90,
    _use_dynamic_varlen_scheduler_sm90,
    num_splits_heuristic,
)
from flash_attn.cute.tile_scheduler import (
    DynamicPersistentVarlenTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentVarlenTileScheduler,
    _decode_dynamic_split_mh_block,
    _sm90_gqa_l2_divisor,
)
from flash_attn.cute.testing import attention_ref, maybe_fake_tensor_mode


@pytest.mark.parametrize(
    "total_mblocks,num_n_blocks,batch_size,qhead_per_kvhead,paged_kv,"
    "head_dim,head_dim_v,expected",
    [
        (1, 3, 2, 4, False, 128, 128, 1),
        (26, 4, 2, 4, False, 128, 128, 4),
        (27, 4, 2, 4, False, 128, 128, 1),
        (27, 5, 2, 4, False, 128, 128, 4),
        (2, 63, 1, 4, False, 128, 128, 63),
        (2, 64, 1, 4, False, 128, 128, 16),
        (2, 191, 1, 4, False, 128, 128, 47),
        (2, 192, 1, 4, False, 128, 128, 24),
        (66, 5, 1, 8, True, 64, 64, 2),
        (67, 5, 1, 8, True, 64, 64, 2),
        (131, 5, 1, 8, True, 64, 64, 2),
        (132, 5, 1, 8, True, 64, 64, 1),
        (67, 5, 1, 8, False, 64, 64, 1),
        (67, 5, 1, 8, True, 128, 128, 1),
        (67, 5, 2, 8, True, 64, 64, 1),
        (133, 5, 2, 8, False, 128, 128, 1),
    ],
)
def test_num_splits_sm90_boundaries(
    total_mblocks,
    num_n_blocks,
    batch_size,
    qhead_per_kvhead,
    paged_kv,
    head_dim,
    head_dim_v,
    expected,
):
    assert (
        _num_splits_sm90(
            total_mblocks,
            132,
            num_n_blocks,
            128,
            batch_size,
            qhead_per_kvhead,
            paged_kv,
            head_dim,
            head_dim_v,
        )
        == expected
    )


def test_generic_num_splits_never_returns_zero():
    assert num_splits_heuristic(133, 132, 5, 128) == 1


@pytest.mark.parametrize(
    "is_local,sparse_block_size_q,paged_kv_non_tma,expected",
    [
        (False, None, False, interface.FwdConfig(192, 144, False, True)),
        (False, None, True, interface.FwdConfig(192, 128, False, True)),
        (True, None, False, interface.FwdConfig(192, 128, False, True)),
        (False, 384, False, interface.FwdConfig(192, 144, False, True)),
        (False, 128, False, interface.FwdConfig(128, 128, False, True)),
    ],
    ids=[
        "dense",
        "paged-non-tma",
        "local",
        "sparse-divisible",
        "sparse-fallback",
    ],
)
def test_sm90_d96_tile_policy(
    is_local, sparse_block_size_q, paged_kv_non_tma, expected
):
    assert (
        _tile_size_fwd_sm90(
            96,
            96,
            False,
            is_local,
            sparse_block_size_q=sparse_block_size_q,
            paged_kv_non_tma=paged_kv_non_tma,
        )
        == expected
    )


@pytest.mark.parametrize(
    "scheduler,sm_count,expected_grid_x",
    [
        (StaticPersistentVarlenTileScheduler, 114, 228),
        (DynamicPersistentVarlenTileScheduler, 132, 132),
    ],
)
def test_persistent_grid_uses_forward_sm_count(
    monkeypatch, scheduler, sm_count, expected_grid_x
):
    monkeypatch.setattr(
        SingleTileVarlenScheduler,
        "get_grid_shape",
        staticmethod(lambda params: (1000, 1, 1)),
    )

    class WrongDeviceHardwareInfo:
        def get_device_multiprocessor_count(self):
            raise AssertionError("persistent scheduler queried another device")

    monkeypatch.setattr(
        tile_scheduler, "HardwareInfo", WrongDeviceHardwareInfo
    )
    grid = scheduler.get_grid_shape(
        SimpleNamespace(is_split_kv=False), sm_count=sm_count
    )
    assert grid[0] == expected_grid_x


def test_dynamic_split_grid_uses_one_cta_per_forward_sm(monkeypatch):
    def reject_static_grid(_params):
        raise AssertionError("split-persistent scheduler queried the static grid")

    monkeypatch.setattr(
        SingleTileVarlenScheduler,
        "get_grid_shape",
        staticmethod(reject_static_grid),
    )
    grid = DynamicPersistentVarlenTileScheduler.get_grid_shape(
        SimpleNamespace(is_split_kv=True), sm_count=132
    )
    assert tuple(grid) == (132, 1, 1)


@pytest.mark.parametrize("num_m_blocks,num_heads,num_splits", [(1, 8, 2), (3, 4, 3)])
def test_dynamic_split_linear_mapping(num_m_blocks, num_heads, num_splits):
    coords = [
        _decode_dynamic_split_mh_block(idx, num_m_blocks, num_splits)
        for idx in range(num_m_blocks * num_heads * num_splits)
    ]
    assert coords == [
        (block, head, split)
        for head in range(num_heads)
        for split in range(num_splits)
        for block in range(num_m_blocks)
    ]
    packed_splits = [split | (num_splits << 16) for _, _, split in coords]
    assert all(packed >> 16 == num_splits for packed in packed_splits)
    assert [packed & 0xFFFF for packed in packed_splits] == [
        split for _, _, split in coords
    ]


@pytest.mark.parametrize(
    "arch,is_packed_varlen,expected",
    [
        (80, True, None),
        (90, True, 17),
        (90, False, None),
        (100, True, None),
        (110, True, None),
        (120, True, None),
    ],
)
def test_combine_max_seqlen_q_is_sm90_packed_only(
    arch, is_packed_varlen, expected
):
    assert (
        _combine_max_seqlen_q_sm90(arch, 17, is_packed_varlen)
        == expected
    )


@maybe_fake_tensor_mode()
def test_combine_max_seqlen_q_compile_specialization(monkeypatch):
    compile_calls = []

    def record_compile(*args):
        compile_calls.append(args)
        return object()

    monkeypatch.setattr(interface, "_compile_fwd_combine", record_compile)
    interface._flash_attn_fwd_combine.compile_cache.clear()

    out_partial = torch.empty(
        (2, 8, 4, 64), device="cuda", dtype=torch.bfloat16
    )
    lse_partial = torch.empty(
        (2, 8, 4), device="cuda", dtype=torch.float32
    )
    out = torch.empty((8, 4, 64), device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.empty((3,), device="cuda", dtype=torch.int32)

    try:
        interface._flash_attn_fwd_combine(
            out_partial, lse_partial, out, cu_seqlens=cu_seqlens
        )
        assert compile_calls[-1][-1] is False

        interface._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            cu_seqlens=cu_seqlens,
            max_seqlen_q=4,
        )
        assert compile_calls[-1][-1] is True
    finally:
        interface._flash_attn_fwd_combine.compile_cache.clear()


@maybe_fake_tensor_mode()
def test_combine_semaphore_compile_specialization(monkeypatch):
    compile_calls = []

    def record_compile(*args):
        compile_calls.append(args)
        return object()

    monkeypatch.setattr(interface, "_compile_fwd_combine", record_compile)
    interface._flash_attn_fwd_combine.compile_cache.clear()

    out_partial = torch.empty(
        (2, 8, 4, 64), device="cuda", dtype=torch.bfloat16
    )
    lse_partial = torch.empty(
        (2, 8, 4), device="cuda", dtype=torch.float32
    )
    out = torch.empty((8, 4, 64), device="cuda", dtype=torch.bfloat16)
    semaphore = torch.empty((1,), device="cuda", dtype=torch.int32)

    try:
        interface._flash_attn_fwd_combine(out_partial, lse_partial, out)
        assert compile_calls[-1][-4] is False

        interface._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            semaphore_to_reset=semaphore,
        )
        assert compile_calls[-1][-4] is True
    finally:
        interface._flash_attn_fwd_combine.compile_cache.clear()


def test_combine_compile_argument_binding(monkeypatch):
    compile_calls = []

    def record_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(interface.cute, "compile", record_compile)
    common_args = (
        interface.torch2cute_dtype_map[torch.bfloat16],
        interface.Float32,
        64,
        16,
        64,
        4,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        None,
        False,
    )

    interface._compile_fwd_combine(*common_args, False)
    args, _ = compile_calls[-1]
    assert args[0].compact_varlen_grid is False
    assert args[-2] is None

    compact_args = (*common_args[:-1], True)
    interface._compile_fwd_combine(*compact_args, True)
    args, _ = compile_calls[-1]
    assert args[0].compact_varlen_grid is True
    assert isinstance(args[-2], interface.Int32)


@pytest.mark.parametrize(
    "ratio,expected",
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (16, 16), (32, 16)],
)
def test_sm90_gqa_l2_divisor(ratio, expected):
    assert _sm90_gqa_l2_divisor(ratio) == expected


@pytest.mark.parametrize(
    "requested,paged_non_tma,tile_n,expected",
    [
        (True, True, 128, True),
        (True, True, 240, True),
        (True, True, 80, True),
        (True, True, 64, False),
        (True, False, 128, False),
        (False, True, 128, False),
    ],
)
def test_sm90_paged_overlap_uses_measured_tile(
    requested, paged_non_tma, tile_n, expected
):
    assert (
        _use_paged_kv_overlap_sm90(requested, paged_non_tma, tile_n)
        is expected
    )


def _wide_paged_d64_selector(**overrides):
    args = {
        "arch": 90,
        "tile_mn": None,
        "paged_kv_non_tma": True,
        "max_seqlen_q": 1,
        "max_seqlen_k": 32 * 1024,
        "head_dim": 64,
        "head_dim_v": 64,
        "local": False,
        "use_block_sparsity": False,
        "pack_gqa": True,
        "qhead_per_kvhead": 8,
        "total_mblocks": 512,
    }
    args.update(overrides)
    return _use_wide_paged_d64_sm90(**args)


@pytest.mark.parametrize("arch", [100, 110, 120])
def test_wide_paged_d64_is_sm90_only(arch):
    assert not _wide_paged_d64_selector(arch=arch)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tile_mn": (64, 128)},
        {"paged_kv_non_tma": False},
        {"max_seqlen_q": 2},
        {"max_seqlen_k": 32 * 1024 - 1},
        {"head_dim": 128, "head_dim_v": 128},
        {"local": True},
        {"use_block_sparsity": True},
        {"pack_gqa": False, "qhead_per_kvhead": 8},
        {"total_mblocks": 256},
        {"total_mblocks": 128, "qhead_per_kvhead": 128},
    ],
)
def test_wide_paged_d64_excludes_adjacent_policies(overrides):
    assert not _wide_paged_d64_selector(**overrides)


def test_wide_paged_d64_accepts_measured_gqa_and_mha_regimes():
    assert _wide_paged_d64_selector()
    assert _wide_paged_d64_selector(
        pack_gqa=False, qhead_per_kvhead=1, total_mblocks=1024
    )


def _page16_paged_d64_loader_selector(**overrides):
    args = {
        "arch": 90,
        "tile_mn": None,
        "paged_kv_non_tma": True,
        "page_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": 32 * 1024,
        "head_dim": 64,
        "head_dim_v": 64,
        "local": False,
        "use_block_sparsity": False,
        "pack_gqa": True,
        "qhead_per_kvhead": 8,
        "total_mblocks": 32,
    }
    args.update(overrides)
    return _use_page16_paged_d64_loader_sm90(**args)


@pytest.mark.parametrize(
    "max_seqlen_k,total_mblocks",
    [(8 * 1024, 512), (16 * 1024, 128), (32 * 1024, 32)],
)
def test_page16_paged_d64_loader_accepts_measured_boundary(
    max_seqlen_k, total_mblocks
):
    assert _page16_paged_d64_loader_selector(
        max_seqlen_k=max_seqlen_k,
        total_mblocks=total_mblocks,
    )
    assert _page16_paged_d64_loader_selector(
        max_seqlen_k=max_seqlen_k,
        pack_gqa=False,
        qhead_per_kvhead=1,
        total_mblocks=total_mblocks,
    )


@pytest.mark.parametrize("arch", [100, 110, 120])
def test_page16_paged_d64_loader_is_sm90_only(arch):
    assert not _page16_paged_d64_loader_selector(arch=arch)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tile_mn": (192, 240)},
        {"paged_kv_non_tma": False},
        {"page_size": 32},
        {"page_size": 128},
        {"max_seqlen_q": 2},
        {"max_seqlen_k": 8 * 1024 - 1, "total_mblocks": 512},
        {"head_dim": 128},
        {"head_dim_v": 128},
        {"local": True},
        {"use_block_sparsity": True},
        {"pack_gqa": False, "qhead_per_kvhead": 8},
        {"max_seqlen_k": 8 * 1024, "total_mblocks": 511},
        {"max_seqlen_k": 16 * 1024, "total_mblocks": 127},
        {"max_seqlen_k": 32 * 1024, "total_mblocks": 31},
    ],
)
def test_page16_paged_d64_loader_excludes_adjacent_policies(overrides):
    assert not _page16_paged_d64_loader_selector(**overrides)


@pytest.mark.parametrize("seqlen_k", [8 * 1024, 16 * 1024, 32 * 1024])
@pytest.mark.parametrize("page_order", [(16, 32), (32, 16)])
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9),
    reason="SM90 paged-loader compile-cache ordering",
)
def test_page16_paged_d64_loader_compile_order(seqlen_k, page_order):
    torch.manual_seed(0)
    batch_size = 512
    num_heads = num_heads_kv = 1
    head_dim = 64
    q = torch.randn(
        batch_size,
        1,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_dense = torch.randn(
        1,
        seqlen_k,
        num_heads_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_dense = torch.randn_like(k_dense)
    out_ref, _ = attention_ref(
        q,
        k_dense.expand(batch_size, -1, -1, -1),
        v_dense.expand(batch_size, -1, -1, -1),
    )

    _flash_attn_fwd.compile_cache.clear()
    try:
        for page_size in page_order:
            pages_per_seq = seqlen_k // page_size
            page_table_row = torch.randperm(
                pages_per_seq, device="cuda", dtype=torch.int32
            )
            page_table = page_table_row.expand(batch_size, -1)
            k_paged = torch.empty(
                pages_per_seq,
                page_size,
                num_heads_kv,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            v_paged = torch.empty_like(k_paged)
            k_paged[page_table_row.long()] = k_dense.view_as(k_paged)
            v_paged[page_table_row.long()] = v_dense.view_as(v_paged)

            out, *_ = _flash_attn_fwd(
                q,
                k_paged,
                v_paged,
                max_seqlen_k=seqlen_k,
                page_table=page_table,
                num_splits=1,
            )
            torch.testing.assert_close(
                out.float(), out_ref.float(), atol=2e-2, rtol=2e-2
            )
        assert len(_flash_attn_fwd.compile_cache.cache) == 2
    finally:
        _flash_attn_fwd.compile_cache.clear()


def _dynamic_selector(**overrides):
    args = {
        "arch": 90,
        "batch_size": 1,
        "num_head": 32,
        "num_head_kv": 4,
        "head_dim": 128,
        "head_dim_v": 128,
        "max_seqlen_q": 128,
        "max_seqlen_k": 8192,
        "no_explicit_window": True,
        "local": False,
        "mask_mod": None,
        "aux_tensors": None,
    }
    args.update(overrides)
    return _use_dynamic_varlen_scheduler_sm90(**args)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        (
            {
                "batch_size": 2,
                "head_dim": 64,
                "no_explicit_window": False,
                "local": True,
            },
            True,
        ),
        (
            {
                "num_head_kv": 32,
                "head_dim": 64,
                "max_seqlen_q": 2,
                "no_explicit_window": False,
                "local": True,
            },
            True,
        ),
        ({"num_head_kv": 32, "head_dim": 96, "max_seqlen_q": 1}, False),
        ({"batch_size": 2, "mask_mod": object()}, False),
        ({"batch_size": 2, "head_dim_v": 512}, False),
        ({"num_head_kv": 32, "aux_tensors": []}, False),
        ({"max_seqlen_k": 16 * 1024 - 1}, False),
        ({"num_head": 64, "max_seqlen_q": 8192, "max_seqlen_k": 8192}, True),
        ({"num_head": 60, "max_seqlen_q": 8192, "max_seqlen_k": 8192}, False),
        ({"num_head": 64, "max_seqlen_q": 8191, "max_seqlen_k": 8192}, False),
        ({"num_head": 64, "max_seqlen_q": 8192, "max_seqlen_k": 8191}, False),
        ({"max_seqlen_k": 16 * 1024, "mask_mod": object()}, False),
        ({"max_seqlen_k": 16 * 1024, "aux_tensors": []}, False),
        ({"max_seqlen_k": 16 * 1024, "head_dim": 64}, False),
        ({}, False),
    ],
)
def test_dynamic_varlen_selector(overrides, expected):
    assert _dynamic_selector(**overrides) is expected


@pytest.mark.parametrize(
    "no_explicit_window,local,expected",
    [
        (False, False, False),  # Explicit right=0 canonicalizes to non-local.
        (True, False, True),  # Ordinary causal attention has no raw window.
        (False, True, False),  # Explicit local window.
    ],
)
def test_dynamic_gqa_window_boundaries(
    no_explicit_window, local, expected
):
    assert _dynamic_selector(
        max_seqlen_k=16 * 1024,
        no_explicit_window=no_explicit_window,
        local=local,
    ) is expected


@pytest.mark.parametrize("arch", [80, 100, 110, 120])
def test_dynamic_varlen_selector_rejects_non_sm90(arch):
    assert not _dynamic_selector(
        arch=arch,
        batch_size=2,
        num_head=32,
        num_head_kv=32,
    )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        ({"arch": 100}, False),
        ({"batch_size": 1}, False),
        ({"cp_world_size": 2}, False),
        ({"is_split_kv": False}, False),
        ({"use_dynamic_splits": False}, False),
        ({"has_varlen_q": False}, False),
        ({"mask_mod": object()}, False),
        ({"aux_tensors": []}, False),
        ({"use_block_sparsity": True}, False),
    ],
)
def test_dynamic_split_varlen_selector(overrides, expected):
    args = {
        "arch": 90,
        "batch_size": 2,
        "cp_world_size": 1,
        "is_split_kv": True,
        "use_dynamic_splits": True,
        "has_varlen_q": True,
        "mask_mod": None,
        "aux_tensors": None,
        "use_block_sparsity": False,
    }
    args.update(overrides)
    assert _use_dynamic_split_varlen_scheduler_sm90(**args) is expected


def test_uniform_dv512_graph_uses_static_dynamic_split_grid():
    assert not _use_dynamic_split_varlen_scheduler_sm90(
        arch=90,
        batch_size=16,
        cp_world_size=1,
        is_split_kv=True,
        use_dynamic_splits=True,
        has_varlen_q=True,
        mask_mod=None,
        aux_tensors=None,
        use_block_sparsity=False,
        num_head=64,
        num_head_kv=1,
        head_dim=64,
        head_dim_v=512,
        max_seqlen_q=1,
        num_m_blocks=1,
        num_splits=8,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("batch_size", 15),
        ("num_head", 32),
        ("num_head_kv", 2),
        ("head_dim", 128),
        ("head_dim_v", 256),
        ("max_seqlen_q", 2),
        ("num_m_blocks", 2),
        ("num_splits", 4),
    ),
)
def test_uniform_dv512_static_dynamic_grid_is_exact(name, value):
    args = {
        "arch": 90,
        "batch_size": 16,
        "cp_world_size": 1,
        "is_split_kv": True,
        "use_dynamic_splits": True,
        "has_varlen_q": True,
        "mask_mod": None,
        "aux_tensors": None,
        "use_block_sparsity": False,
        "num_head": 64,
        "num_head_kv": 1,
        "head_dim": 64,
        "head_dim_v": 512,
        "max_seqlen_q": 1,
        "num_m_blocks": 1,
        "num_splits": 8,
    }
    args[name] = value
    assert _use_dynamic_split_varlen_scheduler_sm90(**args)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        ({"arch": 100}, False),
        ({"arch": 110}, False),
        ({"arch": 120}, False),
        ({"batch_size": 2}, False),
        ({"max_seqlen_q": 2}, False),
        ({"num_m_blocks": 2}, False),
        ({"cp_world_size": 2}, False),
        ({"is_split_kv": False}, False),
        ({"use_dynamic_splits": False}, False),
        ({"has_varlen_q": False}, False),
    ],
)
def test_batch_one_dynamic_split_varlen_selector(overrides, expected):
    args = {
        "arch": 90,
        "batch_size": 1,
        "max_seqlen_q": 1,
        "num_m_blocks": 1,
        "cp_world_size": 1,
        "is_split_kv": True,
        "use_dynamic_splits": True,
        "has_varlen_q": True,
    }
    args.update(overrides)
    assert (
        _use_batch_one_dynamic_split_varlen_scheduler_sm90(**args)
        is expected
    )


@pytest.mark.parametrize(
    "arch,expected", [(90, True), (100, False), (110, False), (120, False)]
)
def test_direct_single_split_output_is_hopper_only(arch, expected):
    assert (
        _use_dynamic_split_varlen_scheduler_sm90(
            arch=arch,
            batch_size=4,
            cp_world_size=1,
            is_split_kv=True,
            use_dynamic_splits=True,
            has_varlen_q=True,
            mask_mod=None,
            aux_tensors=None,
            use_block_sparsity=False,
        )
        is expected
    )


IS_SM90 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic scheduler regression")
def test_dynamic_scheduler_uses_internal_state(monkeypatch):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k = 2, 1, 256
    num_heads, head_dim = 4, 64
    q_padded = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k_padded = torch.randn(
        batch_size,
        seqlen_k,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_padded = torch.randn_like(k_padded)
    q = q_padded.flatten(0, 1)
    k = k_padded.flatten(0, 1)
    v = v_padded.flatten(0, 1)
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * seqlen_k,
        seqlen_k,
        device=device,
        dtype=torch.int32,
    )
    selected = []
    original_init = interface.FlashAttentionForwardSm90.__init__

    def record_init(self, *args, **kwargs):
        selected.append(kwargs["use_dynamic_varlen"])
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(interface.FlashAttentionForwardSm90, "__init__", record_init)
    _flash_attn_fwd.compile_cache.clear()
    try:
        out, *_ = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            num_splits=1,
            causal=True,
        )
        torch.cuda.synchronize()
        assert selected == [True]
        out_ref, _ = attention_ref(q_padded, k_padded, v_padded, causal=True)
        torch.testing.assert_close(
            out.view_as(q_padded), out_ref, atol=1e-2, rtol=1e-2
        )
    finally:
        _flash_attn_fwd.compile_cache.clear()


@pytest.mark.parametrize(
    "split_counts",
    ([1, 1, 1, 1], [1, 3, 1, 7]),
    ids=("all-one", "mixed"),
)
@pytest.mark.parametrize("use_mla", [False, True], ids=("mha", "mla-dv512"))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic SplitKV direct output")
def test_dynamic_split_single_rows_write_final_output(
    split_counts, use_mla, dtype
):
    torch.manual_seed(0)
    device = "cuda"
    batch_size, seqlen_q, seqlen_k = 4, 1, 512
    num_heads, num_heads_kv = 4, 1
    head_dim, head_dim_v = (64, 512) if use_mla else (256, 256)
    q_padded = torch.randn(
        batch_size,
        seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim_v,
        device=device,
        dtype=dtype,
    )
    qv_padded = (
        torch.randn(
            batch_size,
            seqlen_q,
            num_heads,
            head_dim_v,
            device=device,
            dtype=dtype,
        )
        if use_mla
        else None
    )
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    dynamic_splits = torch.tensor(
        split_counts, device=device, dtype=torch.int32
    )

    out, lse, *_ = _flash_attn_fwd(
        q_padded.flatten(0, 1),
        k,
        v,
        qv=qv_padded.flatten(0, 1) if qv_padded is not None else None,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        num_splits=7,
        num_splits_dynamic_ptr=dynamic_splits,
        return_lse=True,
    )
    out_ref, _, lse_ref = attention_ref(
        q_padded, k, v, qv=qv_padded, return_lse=True
    )
    torch.testing.assert_close(
        out.view_as(out_ref).float(),
        out_ref.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        lse,
        lse_ref.permute(1, 0, 2).flatten(1).float(),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not IS_SM90, reason="SM90 dynamic SplitKV CUDA graph regression")
def test_dynamic_split_scheduler_resets_counter_on_cuda_graph_replay(monkeypatch):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, seqlen_k = 4, 1, 512
    num_heads, num_heads_kv, head_dim = 4, 1, 256
    num_splits = 7

    q = torch.randn(
        batch_size * seqlen_q,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    cu_seqlens_q = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    dynamic_splits = torch.tensor(
        [1, 3, 5, 7], device=device, dtype=torch.int32
    )
    scheduler_counter = torch.zeros(1, device=device, dtype=torch.int32)
    torch_zeros = torch.zeros

    def use_external_scheduler_counter(shape, *args, **kwargs):
        if (
            shape == (1,)
            and kwargs.get("dtype") == torch.int32
            and kwargs.get("device") == scheduler_counter.device
        ):
            return scheduler_counter
        return torch_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", use_external_scheduler_counter)

    def run():
        return _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            num_splits=num_splits,
            num_splits_dynamic_ptr=dynamic_splits,
        )[0]

    def reference():
        q_heads = q.view(
            batch_size, seqlen_q, num_heads, head_dim
        ).transpose(1, 2)
        k_heads = k.repeat_interleave(
            num_heads // num_heads_kv, dim=2
        ).transpose(1, 2)
        v_heads = v.repeat_interleave(
            num_heads // num_heads_kv, dim=2
        ).transpose(1, 2)
        scores = (
            q_heads.float() @ k_heads.float().transpose(-2, -1)
        ) / head_dim**0.5
        return (
            (scores.softmax(dim=-1) @ v_heads.float())
            .transpose(1, 2)
            .flatten(0, 1)
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run()
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()
    assert scheduler_counter.item() == 0

    for replay in range(3):
        q.copy_(torch.randn_like(q))
        graph.replay()
        torch.cuda.synchronize()
        assert scheduler_counter.item() == 0
        torch.testing.assert_close(
            graph_out.float(),
            reference(),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda msg: f"CUDA graph replay {replay}: {msg}",
        )
