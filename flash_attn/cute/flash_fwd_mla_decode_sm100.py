# Copyright (c) 2026, Colfax International.
"""SM100 sparse-MLA (DSA) decode kernel: the gathered KV tile is the A operand of both MMAs."""

import math
import operator
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.tcgen05 import CtaGroup, OperandMajorMode

import flash_attn.cute.blackwell_helpers as fa_sm100_utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.topk_gather_kv import CpasyncGatherKVManager
from flash_attn.cute.utils import get_batch_from_cu_tensor, warp_reduce


LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)

SMEM_CAP_BYTES = 232448  # the measured SM100 dynamic-smem ceiling

# 16-bit KV: h -> (latent ring depth, P^T aliased onto the rope ring); 1 rope stage, 1 P^T.
_RING = {8: (6, False), 16: (6, True), 32: (5, False)}
# fp8 KV with an e4m3 Q: kind::f8f6f4 MMAs straight on the gathered e4m3 stages (16 KB per
# latent stage). h -> latent ring depth; the rope ring has 2 stages and P^T is never aliased.
_RING_FP8_MMA = {8: 12, 16: 12, 32: 10}
# The O^T MMA's B is [P_hi | P_lo]^T in e4m3, stacked along N (N = 2h):
#   P_hi = e4m3(P * 2^8), P_lo = e4m3((P * 2^8 - P_hi) * 2^4),
#   O = (O_hi + 2^-4 * O_lo) * 2^-8 / rowsum  (relative error ~2^-8, like a bf16 P).
FP8_P_SCALE = 256.0
FP8_P_LO_SCALE = 16.0


class FlashAttentionMLADecodeSm100:
    # h is also the tcgen05 ld/st Repetition, a power-of-two enum: gate on membership, not h % 8
    SUPPORTED_HEADS = tuple(_RING)

    def __init__(
        self,
        topk_length: int = 2048,
        qhead_per_kvhead: int = 16,
        num_splits: int = 1,
        num_head_groups: int = 1,
    ):
        assert num_head_groups >= 1 and qhead_per_kvhead % num_head_groups == 0, (
            f"num_head_groups={num_head_groups} must divide qhead_per_kvhead={qhead_per_kvhead}"
        )
        h_group = qhead_per_kvhead // num_head_groups
        assert h_group in self.SUPPORTED_HEADS, (
            f"qhead_per_kvhead // num_head_groups must be in {self.SUPPORTED_HEADS}, "
            f"got {h_group}"
        )

        # e4m3 Q and KV (per-tensor scaled 576 B rows, set in __call__): kind::f8f6f4 MMAs on the
        # gathered stages. The caller folds q_scale * k_scale into softmax_scale and multiplies
        # the (bf16) O by v_scale.
        self.fp8_mma = False
        self.o_cols = None  # N of the O^T MMA: h, or 2h for the stacked [P_hi | P_lo]
        self.n_rope = 1
        # S^T accumulators and P^T buffers: 2 for fp8_mma, so S^T(b + 1) and softmax(b + 1)
        # overlap the correction and O^T of block b
        self.n_sp = 1
        self.topk_length = topk_length
        self.h = h_group
        self.num_head_groups = num_head_groups
        self.qhead_per_kvhead = qhead_per_kvhead
        self.hdim_rope = 64
        self.hdimv = 512
        self.hdim_total = self.hdimv + self.hdim_rope  # [latent 0..511][rope 512..575]

        self.tile_n = 128
        self.dv_chunk = 128
        self.num_chunks = self.hdimv // self.dv_chunk  # 4

        n_blocks_full = topk_length // self.tile_n
        assert num_splits >= 1 and n_blocks_full % num_splits == 0, (
            f"num_splits={num_splits} must divide topk_length // tile_n = {n_blocks_full}"
        )
        self.is_split_kv = num_splits > 1
        self.n_blocks_full = n_blocks_full
        self.blocks_per_split = n_blocks_full // num_splits
        # head group fastest: the G CTAs that gather the SAME top-k rows stay co-resident
        self.num_ctas_per_token = num_head_groups * num_splits

        # ---- warp roles ------------------------------------------------------------
        self.num_threads = 512
        self.num_softmax_threads = 128
        self.num_corr_threads = 128
        self.num_gather_threads = 128
        self.corr_warp_lo = 4
        self.mma_warp_id = 8
        self.gather_warp_lo = 12
        self.corr_tid_lo = self.corr_warp_lo * 32
        self.gather_tid_lo = self.gather_warp_lo * 32
        self.num_softmax_warps = self.num_softmax_threads // 32

        # named barrier ids (0 is bar.sync / cute.arch.barrier)
        self.bar_id_tmem = 1
        self.bar_id_softmax = 2
        self.bar_id_corr = 3

        self.num_stages_stats = 2

        self.dtype_acc = Float32
        # gemm_ptx_partial takes a raw TMEM column index: valid only when the alloc is all of TMEM
        self.tmem_alloc_cols = 512
        self.tmem_off_S = 0
        self.tmem_off_O = [self.h * (1 + j) for j in range(self.num_chunks)]
        assert self.tmem_off_O[-1] + self.h <= self.tmem_alloc_cols

        self.buffer_align_bytes = 1024

    # ------------------------------------------------------------------ traced setup
    @cute.jit
    def _make_mmas_and_layouts(self, dtype, n_lat):
        # must run inside a traced region; results passed to the kernel explicitly, never via self
        h, tn, dv = self.h, self.tile_n, self.dv_chunk
        no = self.o_cols
        mma_S = sm100_utils.make_trivial_tiled_mma(
            dtype, OperandMajorMode.K, OperandMajorMode.K, self.dtype_acc,
            CtaGroup.ONE, (tn, h),
        )
        mma_O = sm100_utils.make_trivial_tiled_mma(
            dtype, OperandMajorMode.MN, OperandMajorMode.MN, self.dtype_acc,
            CtaGroup.ONE, (dv, no),
        )
        # A of S^T (K-major) and A of O^T (MN-major) over the SAME bytes. For e4m3 both are
        # 128 B SW128 rows per token (16 B swizzle atomicity: legal for an 8-bit MN-major A).
        LlatK = sm100_utils.make_smem_layout_a(mma_S, (tn, h, dv), dtype, n_lat)
        LlatMN = sm100_utils.make_smem_layout_a(mma_O, (dv, no, tn), dtype, n_lat)
        Lrope = sm100_utils.make_smem_layout_a(mma_S, (tn, h, self.hdim_rope), dtype, self.n_rope)
        # B of S^T: four 128-dim Q^T chunks + one 64-dim rope Q^T.
        LqtC = sm100_utils.make_smem_layout_b(mma_S, (tn, h, dv), dtype, self.num_chunks)
        LqtR = sm100_utils.make_smem_layout_b(mma_S, (tn, h, self.hdim_rope), dtype, 1)
        # B of O^T, MN-major (heads contiguous). 8-bit MN-major B needs N % 16 == 0: N = 2h.
        Lpt = sm100_utils.make_smem_layout_b(mma_O, (dv, no, tn), dtype, self.n_sp)

        assert cute.cosize(LlatK) == n_lat * tn * dv
        assert cute.cosize(LlatMN) == n_lat * tn * dv
        assert cute.cosize(Lrope) == self.n_rope * tn * self.hdim_rope
        assert cute.cosize(LqtC) == self.num_chunks * h * dv
        assert cute.cosize(LqtR) == h * self.hdim_rope
        assert cute.cosize(Lpt) == self.n_sp * tn * no
        return mma_S, mma_O, LlatK, LlatMN, Lrope, LqtC, LqtR, Lpt

    def _get_shared_storage_cls(self, dtype, n_lat, alias_pt_on_rope):
        align = self.buffer_align_bytes
        h = self.h
        tn = self.tile_n
        nw = self.num_softmax_warps
        nst = self.num_stages_stats

        lat_elems = n_lat * tn * self.dv_chunk
        rope_elems = self.n_rope * tn * self.hdim_rope
        qt_elems = h * self.hdim_total
        pt_elems = 0 if alias_pt_on_rope else self.n_sp * tn * self.o_cols

        # 128B-align the fp32 arrays so racecheck does not report hazards across their boundary
        @cute.struct
        class SharedStorage:
            mbar_lat: cute.struct.MemRange[Int64, 2 * n_lat]
            mbar_rope: cute.struct.MemRange[Int64, 2 * self.n_rope]
            mbar_S: cute.struct.MemRange[Int64, 2 * self.n_sp]
            mbar_P: cute.struct.MemRange[Int64, 2 * self.n_sp]
            mbar_O: cute.struct.MemRange[Int64, 2]
            mbar_stats: cute.struct.MemRange[Int64, 2 * nst]
            mbar_Q: cute.struct.MemRange[Int64, 1]
            tmem_holding: cute.struct.MemRange[Int32, 1]
            sRed: cute.struct.Align[cute.struct.MemRange[Float32, nw * h], 128]
            sSmMax: cute.struct.Align[cute.struct.MemRange[Float32, h], 128]
            sSmSum: cute.struct.Align[cute.struct.MemRange[Float32, h], 128]
            sInv: cute.struct.Align[cute.struct.MemRange[Float32, h], 128]
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, nst * h], 128]
            sPt: cute.struct.Align[cute.struct.MemRange[dtype, pt_elems], align]
            sQt: cute.struct.Align[cute.struct.MemRange[dtype, qt_elems], align]
            sRope: cute.struct.Align[cute.struct.MemRange[dtype, rope_elems], align]
            sLat: cute.struct.Align[cute.struct.MemRange[dtype, lat_elems], align]

        return SharedStorage

    # ------------------------------------------------------------------ host entry
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,              # (total_q, h, 64)
        mQv: cute.Tensor,             # (total_q, h, 512)
        mK: cute.Tensor,              # (total_k, h_k, 64)
        mV: cute.Tensor,              # (total_k, h_k, 512)
        mO: cute.Tensor,              # (total_q, h, 512)  or (S, total_q, h, 512) fp32
        mLSE: Optional[cute.Tensor],  # (S, h, total_q) split, (total_q, h) unsplit, None if unused
        softmax_scale: Float32,
        mCuSeqlensQ: cute.Tensor,
        mIndexTopk: cute.Tensor,                      # (total_q, topk)
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mTopkValidLen: Optional[cute.Tensor] = None,  # (total_q,)
        stream: cuda.CUstream = None,
    ):
        assert not self.is_split_kv or mLSE is not None, (
            "split-KV needs the partial LSE tensor: flash_fwd_combine reduces on it"
        )

        dtype = mV.element_type
        assert dtype in (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN), (
            "sparse decode requires 16-bit or e4m3 KV"
        )
        self.fp8_mma = dtype == cutlass.Float8E4M3FN
        self.dtype_O = mO.element_type

        new_stride = lambda mX: (
            *(cute.assume(s, divby=128 // mX.element_type.width) for s in mX.stride[:-1]),
            mX.stride[-1],
        )
        mQ, mQv, mK, mV, mO = [
            cute.make_tensor(mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX)))
            for mX in (mQ, mQv, mK, mV, mO)
        ]
        # (total, h, d) -> (total, d, h)
        mQ, mQv = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=[0, 2, 1]))
            for mX in (mQ, mQv)
        ]
        if const_expr(self.is_split_kv):
            # fold the split mode to the back: (S, total_q, h, dv) -> (total_q, dv, h, S)
            mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[1, 3, 2, 0]))
            # (S, h, total_q) -> (total_q, h, S)   (query dim contiguous in gmem)
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
        else:
            mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[0, 2, 1]))
        mK, mV = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=[0, 2, 1]))
            for mX in (mK, mV)
        ]
        # (total_q, topk) -> (topk, total_q)
        mIndexTopk = cute.make_tensor(
            mIndexTopk.iterator, cute.select(mIndexTopk.layout, mode=[1, 0])
        )

        self.o_cols = 2 * self.h if const_expr(self.fp8_mma) else self.h
        if const_expr(self.fp8_mma):
            # S^T stages at [0, h) and [h, 2h); O_j^T = [O_hi | O_lo] at 2h + 2hj
            self.n_sp, self.n_rope = 2, 2
            self.tmem_off_O = [2 * self.h * (1 + j) for j in range(self.num_chunks)]
            assert self.tmem_off_O[-1] + 2 * self.h <= self.tmem_alloc_cols
            n_lat, alias_pt = _RING_FP8_MMA[self.h], False
        else:
            n_lat, alias_pt = _RING[self.h]
        SharedStorage = self._get_shared_storage_cls(dtype, n_lat, alias_pt)
        smem_bytes = SharedStorage.size_in_bytes()
        assert smem_bytes <= SMEM_CAP_BYTES, (
            f"h={self.h}: {smem_bytes} B exceeds the {SMEM_CAP_BYTES} B smem cap"
        )
        self.n_lat = n_lat
        self.alias_pt_on_rope = alias_pt

        mma_S, mma_O, LlatK, LlatMN, Lrope, LqtC, LqtR, Lpt = self._make_mmas_and_layouts(
            dtype, n_lat
        )

        total_q = cute.size(mQv.shape[0])
        grid_dim = (total_q * self.num_ctas_per_token, mV.shape[2], 1)

        self.kernel(
            mQ, mQv, mK, mV, mO, mLSE, mIndexTopk, mTopkValidLen,
            mCuSeqlensQ, mCuSeqlensK, mSeqUsedK,
            mma_S, mma_O, LlatK, LlatMN, Lrope, LqtC, LqtR, Lpt,
            softmax_scale * LOG2_E,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=(self.num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    # ------------------------------------------------------------------ kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQv: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mIndexTopk: cute.Tensor,
        mTopkValidLen: Optional[cute.Tensor],
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mma_S: cute.TiledMma,
        mma_O: cute.TiledMma,
        LlatK: cute.ComposedLayout,
        LlatMN: cute.ComposedLayout,
        Lrope: cute.ComposedLayout,
        LqtC: cute.ComposedLayout,
        LqtR: cute.ComposedLayout,
        Lpt: cute.ComposedLayout,
        softmax_scale_log2: Float32,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, head_kv, _ = cute.arch.block_idx()
        dtype = mV.element_type
        h = const_expr(self.h)

        # ==== grid decode: x = (token, KV split, head group), head group fastest =====
        m_idx = bidx // self.num_ctas_per_token
        rem = bidx % self.num_ctas_per_token
        hg_idx = rem % self.num_head_groups
        split_idx = rem // self.num_head_groups

        batch_idx = get_batch_from_cu_tensor(m_idx, mCuSeqlensQ)
        seqlen = SeqlenInfoQK.create(
            batch_idx, Int32(1), Int32(1),
            mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=None, mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_n, tile_n=self.tile_n,
        )
        seqlen_k_limit = seqlen.seqlen_k

        if const_expr(mTopkValidLen is None):
            n_valid_blocks = Int32(self.n_blocks_full)
        else:
            n_valid_blocks = max(
                Int32(1),
                min(
                    cute.ceil_div(mTopkValidLen[m_idx], self.tile_n),
                    Int32(self.n_blocks_full),
                ),
            )
        if const_expr(not self.is_split_kv):
            n_block_lo = Int32(0)
            num_n_blocks = n_valid_blocks
        else:
            # a split entirely past the valid window still walks one all-masked block
            n_block_lo = split_idx * Int32(self.blocks_per_split)
            num_n_blocks = max(
                Int32(1),
                min(n_valid_blocks, n_block_lo + Int32(self.blocks_per_split)) - n_block_lo,
            )

        # ==== smem ==================================================================
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        lat_ptr = storage.sLat.data_ptr()
        rope_ptr = storage.sRope.data_ptr()
        sLatK = cute.make_tensor(cute.recast_ptr(lat_ptr, LlatK.inner, dtype), LlatK.outer)
        sLatMN = cute.make_tensor(cute.recast_ptr(lat_ptr, LlatMN.inner, dtype), LlatMN.outer)
        sRope = cute.make_tensor(cute.recast_ptr(rope_ptr, Lrope.inner, dtype), Lrope.outer)
        qt_ptr = storage.sQt.data_ptr()
        sQtC = cute.make_tensor(cute.recast_ptr(qt_ptr, LqtC.inner, dtype), LqtC.outer)
        sQtR = cute.make_tensor(
            cute.recast_ptr(qt_ptr + self.num_chunks * h * self.dv_chunk, LqtR.inner, dtype),
            LqtR.outer,
        )
        if const_expr(self.alias_pt_on_rope):
            # sound because pl_rope's consumer release is delayed until after the O^T MMAs
            sPt = cute.make_tensor(cute.recast_ptr(rope_ptr, Lpt.inner, dtype), Lpt.outer)
        else:
            sPt = storage.sPt.get_tensor(Lpt.outer, swizzle=Lpt.inner)

        sRed = storage.sRed.get_tensor(cute.make_layout((self.num_softmax_warps, h)))
        sSmMax = storage.sSmMax.get_tensor(cute.make_layout(h))
        sSmSum = storage.sSmSum.get_tensor(cute.make_layout(h))
        sInv = storage.sInv.get_tensor(cute.make_layout(h))
        sScale = storage.sScale.get_tensor(cute.make_layout((self.num_stages_stats, h)))

        # no zero-fill needed: masked rows gather with cp.async src_size = 0, which zero-fills

        # ==== pipelines =============================================================
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        gather_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_gather_threads)
        mma_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads)
        corr_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_corr_threads)

        pl_lat = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_lat.data_ptr(), num_stages=self.n_lat,
            producer_group=gather_grp, consumer_group=mma_grp,
            cta_layout_vmnk=cta_layout_vmnk, defer_sync=True,
        )
        pl_rope = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_rope.data_ptr(), num_stages=self.n_rope,
            producer_group=gather_grp, consumer_group=mma_grp,
            cta_layout_vmnk=cta_layout_vmnk, defer_sync=True,
        )
        # arrive.noinc does not bump the pending count: arrive count must equal the gather threads
        assert pl_lat.sync_object_full.arrive_count == self.num_gather_threads
        assert pl_rope.sync_object_full.arrive_count == self.num_gather_threads

        pl_S = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S.data_ptr(), num_stages=self.n_sp,
            producer_group=mma_grp, consumer_group=sm_grp,
            cta_layout_vmnk=cta_layout_vmnk, defer_sync=True,
        )
        pl_P = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P.data_ptr(), num_stages=self.n_sp,
            producer_group=sm_grp, consumer_group=mma_grp,
            cta_layout_vmnk=cta_layout_vmnk, defer_sync=True,
        )
        pl_O = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O.data_ptr(), num_stages=1,
            producer_group=mma_grp, consumer_group=corr_grp,
            cta_layout_vmnk=cta_layout_vmnk, defer_sync=True,
        )
        pl_stats = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_stats.data_ptr(), num_stages=self.num_stages_stats,
            producer_group=sm_grp, consumer_group=corr_grp, defer_sync=True,
        )

        mbar_Q = storage.mbar_Q.data_ptr()
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_Q, self.num_corr_threads)

        # ==== TMEM ==================================================================
        tmem_bar = pipeline.NamedBarrier(
            barrier_id=self.bar_id_tmem, num_threads=self.num_threads
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding.data_ptr(),
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=0,
            is_two_cta=False,
        )
        tmem.allocate(self.tmem_alloc_cols)

        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        thr_mma_S = mma_S.get_slice(0)
        thr_mma_O = mma_O.get_slice(0)
        accS_fake = thr_mma_S.make_fragment_C(thr_mma_S.partition_shape_C((self.tile_n, h)))
        accO_fake = thr_mma_O.make_fragment_C(thr_mma_O.partition_shape_C((self.dv_chunk, h)))

        accS = cute.make_tensor(tmem_ptr + self.tmem_off_S, accS_fake.layout)[(None, None), 0, 0]
        accO = [
            cute.make_tensor(tmem_ptr + self.tmem_off_O[j], accO_fake.layout)[(None, None), 0, 0]
            for j in range(self.num_chunks)
        ]
        mma_O_acc = mma_O
        if const_expr(self.fp8_mma):
            # h-wide views of the [O_hi | O_lo] accumulators: hi j at 2j, lo j at 2j + 1
            # (layout only: an N = h MMA; B K-major so h = 8 is a legal 8-bit shape)
            mma_O_acc = sm100_utils.make_trivial_tiled_mma(
                dtype, OperandMajorMode.MN, OperandMajorMode.K, self.dtype_acc,
                CtaGroup.ONE, (self.dv_chunk, h),
            )
            thr_acc = mma_O_acc.get_slice(0)
            accO_fake = thr_acc.make_fragment_C(thr_acc.partition_shape_C((self.dv_chunk, h)))
            accO = [
                cute.make_tensor(
                    tmem_ptr + self.tmem_off_O[j // 2] + h * (j % 2), accO_fake.layout
                )[(None, None), 0, 0]
                for j in range(2 * self.num_chunks)
            ]

        # ==== role dispatch =========================================================
        if warp_idx < self.corr_warp_lo:
            self.softmax_loop(
                mIndexTopk, mma_S, accS, sRed, sSmMax, sSmSum, sScale, sPt,
                pl_S, pl_P, pl_stats, tidx, warp_idx, m_idx, seqlen_k_limit,
                softmax_scale_log2, n_block_lo, num_n_blocks,
            )
        elif warp_idx < self.mma_warp_id:
            self.corr_epilogue_loop(
                mQ, mQv, mO, mLSE, mma_O_acc, accO, sQtC, sQtR,
                sScale, sSmMax, sSmSum, sInv, pl_O, pl_stats, mbar_Q,
                tidx, head_kv, m_idx, hg_idx, split_idx,
                softmax_scale_log2, num_n_blocks,
            )
        elif warp_idx == self.mma_warp_id:
            self.mma_loop(
                mma_S, mma_O, sLatK, sLatMN, sRope, sQtC, sQtR, sPt,
                pl_lat, pl_rope, pl_S, pl_P, pl_O, mbar_Q, num_n_blocks,
            )
        elif warp_idx >= self.gather_warp_lo:
            self.gather_loop(
                mIndexTopk, mK, mV, sLatK, sRope, pl_lat, pl_rope,
                tidx, warp_idx, m_idx, seqlen, batch_idx, head_kv, seqlen_k_limit,
                dtype, n_block_lo, num_n_blocks,
            )

        # ==== TMEM teardown (all 512 threads) =======================================
        tmem.relinquish_alloc_permit()
        tmem_bar.arrive_and_wait()
        tmem.free(tmem_ptr)

    # ================================================================ gather warps
    @cute.jit
    def gather_loop(
        self, mIndexTopk, mK, mV, sLatK, sRope, pl_lat, pl_rope,
        tidx, warp_idx, m_idx, seqlen, batch_idx, head_kv, seqlen_k_limit, dtype,
        n_block_lo, num_n_blocks,
    ):
        mgr = CpasyncGatherKVManager.create(
            mIndexTopk[None, m_idx],
            Int32(0),  # cta_rank_in_cluster
            tidx - self.gather_tid_lo,
            warp_idx - self.gather_warp_lo,
            self.topk_length,
            seqlen_k_limit,
            self.tile_n,
            self.hdim_rope,
            self.hdimv,
            self.num_chunks,  # num_hdimv_splits: one hdim_v split per latent chunk
            self.num_gather_threads,
            dtype,
            1,  # cta_group_size
            disable_bitmask=False,
        )
        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=2)[None, None, head_kv]
        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=2)[None, None, head_kv]

        ps_lat = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.n_lat)
        ps_rope = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.n_rope)

        for n_block in cutlass.range(num_n_blocks, unroll=1):
            mgr.load_index_topk(n_block_lo + n_block, transpose=False)
            for c in cutlass.range_constexpr(self.num_chunks):
                pl_lat.producer_acquire(ps_lat)
                mgr.load_X(
                    mV_cur, sLatK[None, None, None, ps_lat.index], False, "V", c * self.dv_chunk
                )
                cute.arch.cp_async_commit_group()
                pl_lat.sync_object_full.arrive_cp_async_mbarrier(ps_lat.index)
                ps_lat.advance()
            pl_rope.producer_acquire(ps_rope)
            mgr.load_X(mK_cur, sRope[None, None, None, ps_rope.index], False, "K", 0)
            cute.arch.cp_async_commit_group()
            pl_rope.sync_object_full.arrive_cp_async_mbarrier(ps_rope.index)
            ps_rope.advance()

        # every empty-barrier arrive (a tcgen05 commit from the MMA warp) must have landed
        pl_lat.producer_tail(ps_lat)
        pl_rope.producer_tail(ps_rope)

    # ================================================================== MMA warp
    @cute.jit
    def mma_loop(
        self, mma_S, mma_O, sLatK, sLatMN, sRope, sQtC, sQtR, sPt,
        pl_lat, pl_rope, pl_S, pl_P, pl_O, mbar_Q, num_n_blocks,
    ):
        tSrLat = mma_S.make_fragment_A(sLatK)
        tSrRope = mma_S.make_fragment_A(sRope)
        tSrQtC = mma_S.make_fragment_B(sQtC)
        tSrQtR = mma_S.make_fragment_B(sQtR)
        tOrLat = mma_O.make_fragment_A(sLatMN)
        tOrPt = mma_O.make_fragment_B(sPt)

        Producer = pipeline.PipelineUserType.Producer
        Consumer = pipeline.PipelineUserType.Consumer
        states = (
            pipeline.make_pipeline_state(Consumer, self.n_lat),  # cs_lat: S^T reads
            pipeline.make_pipeline_state(Consumer, self.n_lat),  # use_lat: O^T reads
            pipeline.make_pipeline_state(Consumer, self.n_rope),  # cs_rope
            pipeline.make_pipeline_state(Consumer, self.n_sp),  # cs_P
            pipeline.make_pipeline_state(Producer, self.n_sp),
            pipeline.make_pipeline_state(Producer, 1),
        )

        # Q^T is filled by the correction warps with plain STS
        cute.arch.mbarrier_wait(mbar_Q, Int32(0))

        if const_expr(self.fp8_mma):
            # S^T(b + 1) is issued before waiting for P^T(b), into the other S^T stage, so the
            # softmax of block b + 1 runs under the correction and O^T of block b
            states = self._mma_S(
                mma_S, sLatK, sRope, sQtC, sQtR, tSrLat, tSrRope, tSrQtC, tSrQtR,
                pl_lat, pl_rope, pl_S, states,
            )
            if num_n_blocks > 1:
                states = self._mma_S(
                    mma_S, sLatK, sRope, sQtC, sQtR, tSrLat, tSrRope, tSrQtC, tSrQtR,
                    pl_lat, pl_rope, pl_S, states,
                )
            states = self._mma_O(
                mma_O, sLatMN, sPt, tOrLat, tOrPt, pl_lat, pl_rope, pl_P, pl_O, states, True,
            )
            for b in cutlass.range(1, num_n_blocks, unroll=1):
                if b + 1 < num_n_blocks:
                    states = self._mma_S(
                        mma_S, sLatK, sRope, sQtC, sQtR, tSrLat, tSrRope, tSrQtC, tSrQtR,
                        pl_lat, pl_rope, pl_S, states,
                    )
                states = self._mma_O(
                    mma_O, sLatMN, sPt, tOrLat, tOrPt, pl_lat, pl_rope, pl_P, pl_O, states,
                    False,
                )
            return

        states = self._mma_block(
            mma_S, mma_O, sLatK, sLatMN, sRope, sQtC, sQtR, sPt,
            tSrLat, tSrRope, tSrQtC, tSrQtR, tOrLat, tOrPt,
            pl_lat, pl_rope, pl_S, pl_P, pl_O, states, True,
        )
        for _ in cutlass.range(num_n_blocks - 1, unroll=1):
            states = self._mma_block(
                mma_S, mma_O, sLatK, sLatMN, sRope, sQtC, sQtR, sPt,
                tSrLat, tSrRope, tSrQtC, tSrQtR, tOrLat, tOrPt,
                pl_lat, pl_rope, pl_S, pl_P, pl_O, states, False,
            )

    @cute.jit
    def _mma_block(
        self, mma_S, mma_O, sLatK, sLatMN, sRope, sQtC, sQtR, sPt,
        tSrLat, tSrRope, tSrQtC, tSrQtR, tOrLat, tOrPt,
        pl_lat, pl_rope, pl_S, pl_P, pl_O, states,
        is_first: cutlass.Constexpr[bool],
    ):
        states = self._mma_S(
            mma_S, sLatK, sRope, sQtC, sQtR, tSrLat, tSrRope, tSrQtC, tSrQtR,
            pl_lat, pl_rope, pl_S, states,
        )
        return self._mma_O(
            mma_O, sLatMN, sPt, tOrLat, tOrPt, pl_lat, pl_rope, pl_P, pl_O, states, is_first,
        )

    @cute.jit
    def _mma_S(
        self, mma_S, sLatK, sRope, sQtC, sQtR, tSrLat, tSrRope, tSrQtC, tSrQtR,
        pl_lat, pl_rope, pl_S, states,
    ):
        cs_lat, use_lat, cs_rope, cs_P, ps_S, ps_O = states

        # ---- S^T = KV . Q^T : four latent K-chunks + the rope chunk ---------------
        pl_S.producer_acquire(ps_S)
        acc = Int32(self.tmem_off_S)
        if const_expr(self.n_sp > 1):
            acc = acc + ps_S.index * self.h  # S^T stage ps_S
        for c in cutlass.range_constexpr(self.num_chunks):
            pl_lat.consumer_wait(cs_lat)
            # orders this thread's generic-proxy view of the stage before its async-proxy MMA reads
            cute.arch.fence_view_async_shared()
            # stays in the loop: the c == 0 fence orders pl_S.producer_acquire before the first MMA
            fa_sm100_utils.fence_tcgen05_after_thread_sync()
            st = cs_lat.index
            fa_sm100_utils.gemm_ptx_partial(
                mma_S.op, acc,
                tSrLat[None, None, None, st], tSrQtC[None, None, None, c],
                sLatK[None, None, None, st], sQtC[None, None, None, c],
                zero_init=const_expr(c == 0), cta_group=1,
            )
            cs_lat.advance()
        pl_rope.consumer_wait(cs_rope)
        cute.arch.fence_view_async_shared()
        fa_sm100_utils.fence_tcgen05_after_thread_sync()
        fa_sm100_utils.gemm_ptx_partial(
            mma_S.op, acc,
            tSrRope[None, None, None, cs_rope.index], tSrQtR[None, None, None, 0],
            sRope[None, None, None, cs_rope.index], sQtR[None, None, None, 0],
            zero_init=False, cta_group=1,
        )
        pl_S.producer_commit(ps_S)
        ps_S.advance()
        if const_expr(self.fp8_mma):
            # nothing aliases the rope stage: free it as soon as the S^T MMAs have read it
            pl_rope.consumer_release(cs_rope)
            cs_rope.advance()
        return cs_lat, use_lat, cs_rope, cs_P, ps_S, ps_O

    @cute.jit
    def _mma_O(
        self, mma_O, sLatMN, sPt, tOrLat, tOrPt, pl_lat, pl_rope, pl_P, pl_O, states,
        is_first: cutlass.Constexpr[bool],
    ):
        cs_lat, use_lat, cs_rope, cs_P, ps_S, ps_O = states

        # ---- O^T += V^T . P^T, one MMA per dv chunk, A aliased on the latent ring --
        pl_P.consumer_wait(cs_P)
        pl_O.producer_acquire(ps_O)
        fa_sm100_utils.fence_tcgen05_after_thread_sync()
        for c in cutlass.range_constexpr(self.num_chunks):
            st = use_lat.index
            fa_sm100_utils.gemm_ptx_partial(
                mma_O.op, Int32(self.tmem_off_O[c]),
                tOrLat[None, None, None, st], tOrPt[None, None, None, cs_P.index],
                sLatMN[None, None, None, st], sPt[None, None, None, cs_P.index],
                zero_init=const_expr(is_first), cta_group=1,
            )
            # the latent stage stays pinned until exactly here
            pl_lat.consumer_release(use_lat)
            use_lat.advance()
        pl_O.producer_commit(ps_O)
        ps_O.advance()

        # released late: P^T may alias the rope stage, so the next gather must not overwrite it
        if const_expr(not self.fp8_mma):
            pl_rope.consumer_release(cs_rope)
            cs_rope.advance()
        pl_P.consumer_release(cs_P)
        cs_P.advance()
        return cs_lat, use_lat, cs_rope, cs_P, ps_S, ps_O

    # ================================================================ softmax warps
    @cute.jit
    def softmax_loop(
        self, mIndexTopk, mma_S, accS, sRed, sSmMax, sSmSum, sScale, sPt,
        pl_S, pl_P, pl_stats, tidx, warp_idx, m_idx, seqlen_k_limit,
        softmax_scale_log2, n_block_lo, num_n_blocks,
    ):
        h = const_expr(self.h)
        lane_idx = cute.arch.lane_idx()
        mIndexTopk_cur = mIndexTopk[None, m_idx]

        thr_mma_S = mma_S.get_slice(0)
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(h)), Float32
        )
        thr_ldS = tcgen05.make_tmem_copy(ld_atom, accS).get_slice(tidx)
        cS = cute.make_identity_tensor((self.tile_n, h))
        cS_t2r = thr_ldS.partition_D(thr_mma_S.partition_C(cS)[(None, None), 0, 0])
        assert cute.size(cS_t2r) == h
        tSaccS = thr_ldS.partition_S(accS)
        if const_expr(self.n_sp == 2):
            # fp8_mma: the second S^T accumulator, h columns on
            tSaccS = (tSaccS, thr_ldS.partition_S(cute.make_tensor(accS.iterator + h, accS.layout)))
        tSrS = cute.make_rmem_tensor(cS_t2r.shape, Float32)
        tSrP = cute.make_rmem_tensor(cS_t2r.shape, Float32)
        key_row = cS_t2r[0][0]

        pt_layout = cute.make_ordered_layout((self.o_cols, self.tile_n), order=(0, 1))

        states = (
            pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.n_sp),
            pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.n_sp),
            pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages_stats
            ),
        )
        states = self._softmax_block(
            mIndexTopk_cur, sRed, sSmMax, sSmSum, sScale, sPt, pt_layout,
            thr_ldS, tSaccS, tSrS, tSrP, cS_t2r, key_row,
            pl_S, pl_P, pl_stats, states, tidx, warp_idx, lane_idx,
            seqlen_k_limit, softmax_scale_log2, n_block_lo, True,
        )
        for i in cutlass.range(num_n_blocks - 1, unroll=1):
            states = self._softmax_block(
                mIndexTopk_cur, sRed, sSmMax, sSmSum, sScale, sPt, pt_layout,
                thr_ldS, tSaccS, tSrS, tSrP, cS_t2r, key_row,
                pl_S, pl_P, pl_stats, states, tidx, warp_idx, lane_idx,
                seqlen_k_limit, softmax_scale_log2, n_block_lo + i + 1, False,
            )
        _, ps_P, ps_st = states
        pl_P.producer_tail(ps_P)
        pl_stats.producer_tail(ps_st)

    @cute.jit
    def _softmax_block(
        self, mIndexTopk_cur, sRed, sSmMax, sSmSum, sScale, sPt, pt_layout,
        thr_ldS, tSaccS, tSrS, tSrP, cS_t2r, key_row,
        pl_S, pl_P, pl_stats, states, tidx, warp_idx, lane_idx,
        seqlen_k_limit, softmax_scale_log2, n_block: Int32,
        is_first: cutlass.Constexpr[bool],
    ):
        h = const_expr(self.h)
        cs_S, ps_P, ps_st = states
        bar_sm = pipeline.NamedBarrier(
            barrier_id=self.bar_id_softmax, num_threads=self.num_softmax_threads
        )

        topk_idx = mIndexTopk_cur[n_block * self.tile_n + key_row]
        key_valid = topk_idx >= 0 and topk_idx < seqlen_k_limit

        # ---- read S^T out of TMEM and release the accumulator ---------------------
        pl_S.consumer_wait(cs_S)
        fa_sm100_utils.fence_tcgen05_after_thread_sync()
        if const_expr(self.n_sp == 1):
            cute.copy(thr_ldS, tSaccS, tSrS)
        else:
            if cs_S.index == 0:
                cute.copy(thr_ldS, tSaccS[0], tSrS)
            else:
                cute.copy(thr_ldS, tSaccS[1], tSrS)
        cute.arch.fence_view_async_tmem_load()
        fa_sm100_utils.fence_tcgen05_before_thread_sync()
        pl_S.consumer_release(cs_S)
        cs_S.advance()

        for i in cutlass.range_constexpr(h):
            tSrS[i] = tSrS[i] if key_valid else -Float32.inf

        # ---- block max per head across the 128 lanes: warp butterfly + smem -------
        for i in cutlass.range_constexpr(h):
            tSrP[i] = warp_reduce(tSrS[i], cute.arch.fmax)
        if lane_idx == 0:
            for i in cutlass.range_constexpr(h):
                sRed[warp_idx, i] = tSrP[i]
        bar_sm.arrive_and_wait()

        pl_stats.producer_acquire(ps_st)
        st_idx = ps_st.index
        if tidx < h:
            m_blk = sRed[0, tidx]
            for w in cutlass.range_constexpr(1, self.num_softmax_warps):
                m_blk = cute.arch.fmax(m_blk, sRed[w, tidx])
            if const_expr(is_first):
                # is_first initializes the running stats; no separate init pass is needed
                m_new = m_blk
                sSmMax[tidx] = m_new
                sScale[st_idx, tidx] = Float32(0.0)
            else:
                m_old = sSmMax[tidx]
                m_new = cute.arch.fmax(m_old, m_blk)
                sSmMax[tidx] = m_new
                sScale[st_idx, tidx] = (
                    cute.math.exp2((m_old - m_new) * softmax_scale_log2, fastmath=True)
                    if m_old != -Float32.inf else Float32(0.0)
                )
        bar_sm.arrive_and_wait()

        # ---- P = exp2(S * scale_log2 - m_new * scale_log2) ------------------------
        for i in cutlass.range_constexpr(h):
            # Keep the true running max; only exponentiation needs a finite empty-row bias.
            row_max = sSmMax[cS_t2r[i][1]]
            safe_max = row_max if row_max != -Float32.inf else Float32(0.0)
            bias = -safe_max * softmax_scale_log2
            tSrP[i] = cute.math.exp2(tSrS[i] * softmax_scale_log2 + bias, fastmath=True)

        # ---- row sum per head across the 128 lanes -------------------------------
        for i in cutlass.range_constexpr(h):
            tSrS[i] = warp_reduce(tSrP[i], operator.add)
        if lane_idx == 0:
            for i in cutlass.range_constexpr(h):
                sRed[warp_idx, i] = tSrS[i]

        # ---- P^T staging: this thread's key, all h heads, MN-major ---------------
        pl_P.producer_acquire(ps_P)
        pt_nd = cute.composition(sPt[None, None, None, ps_P.index], pt_layout)
        if const_expr(not self.fp8_mma):
            for i in cutlass.range_constexpr(h):
                pt_nd[cS_t2r[i][1], key_row] = tSrP[i].to(sPt.element_type)
        else:
            # this key's row of [P_hi | P_lo]^T: 2h contiguous e4m3 (heads, then heads)
            rX = cute.make_rmem_tensor((h,), Float32)
            for i in cutlass.range_constexpr(h):
                rX[i] = tSrP[i] * FP8_P_SCALE
            rHi = cute.make_rmem_tensor((h,), cutlass.Float8E4M3FN)
            rHi.store(rX.load().to(cutlass.Float8E4M3FN))
            rRow = cute.make_rmem_tensor((2 * h,), cutlass.Float8E4M3FN)
            rLo = cute.make_rmem_tensor((h,), cutlass.Float8E4M3FN)
            rLo.store(((rX.load() - rHi.load().to(Float32)) * FP8_P_LO_SCALE).to(
                cutlass.Float8E4M3FN
            ))
            for i in cutlass.range_constexpr(h):
                rRow[cS_t2r[i][1]] = rHi[i]
                rRow[h + cS_t2r[i][1]] = rLo[i]
            cute.autovec_copy(rRow, pt_nd[None, key_row])
        cute.arch.fence_proxy("async.shared", space="cta")
        pl_P.producer_commit(ps_P)
        ps_P.advance()

        bar_sm.arrive_and_wait()
        if tidx < h:
            s = sRed[0, tidx]
            for w in cutlass.range_constexpr(1, self.num_softmax_warps):
                s = s + sRed[w, tidx]
            if const_expr(is_first):
                sSmSum[tidx] = s
            else:
                sSmSum[tidx] = sSmSum[tidx] * sScale[st_idx, tidx] + s
        # the next block's sRed writes must not race this block's row-sum reads
        bar_sm.arrive_and_wait()
        pl_stats.producer_commit(ps_st)
        ps_st.advance()
        return cs_S, ps_P, ps_st

    # =========================================== correction + Q^T load + epilogue
    @cute.jit
    def corr_epilogue_loop(
        self, mQ, mQv, mO, mLSE, mma_O, accO, sQtC, sQtR,
        sScale, sSmMax, sSmSum, sInv, pl_O, pl_stats, mbar_Q,
        tidx, head_kv, m_idx, hg_idx, split_idx,
        softmax_scale_log2, num_n_blocks,
    ):
        h = const_expr(self.h)
        ctid = tidx - self.corr_tid_lo
        # this CTA owns query heads [head_base, head_base + h), disjoint across CTAs
        head_base = head_kv * self.qhead_per_kvhead + hg_idx * h
        bar_corr = pipeline.NamedBarrier(
            barrier_id=self.bar_id_corr, num_threads=self.num_corr_threads
        )

        # ---- Q^T load: (h, 576) K-major, four 128-dim chunks + the rope chunk -----
        qtc_layout = cute.make_ordered_layout((h, self.dv_chunk), order=(0, 1))
        qtr_layout = cute.make_ordered_layout((h, self.hdim_rope), order=(0, 1))
        gQv = mQv[m_idx, None, None]  # (512, h_total)
        gQr = mQ[m_idx, None, None]  # (64,  h_total)
        for c in cutlass.range_constexpr(self.num_chunks):
            qt_nd = cute.composition(sQtC[None, None, None, c], qtc_layout)
            for n in cutlass.range_constexpr(h):
                qt_nd[n, ctid] = gQv[c * self.dv_chunk + ctid, head_base + n]
        qt_rope = cute.composition(sQtR[None, None, None, 0], qtr_layout)
        if ctid < self.hdim_rope:
            for n in cutlass.range_constexpr(h):
                qt_rope[n, ctid] = gQr[ctid, head_base + n]
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(mbar_Q)

        # ---- TMEM views of the four O^T accumulators -----------------------------
        thr_mma_O = mma_O.get_slice(0)
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(h)), Float32
        )
        st_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(h)), Float32
        )
        thr_ldO = tcgen05.make_tmem_copy(ld_atom, accO[0]).get_slice(ctid)
        thr_stO = tcgen05.make_tmem_copy(st_atom, accO[0]).get_slice(ctid)
        cO = cute.make_identity_tensor((self.dv_chunk, h))
        cO_t2r = thr_ldO.partition_D(thr_mma_O.partition_C(cO)[(None, None), 0, 0])
        assert cute.size(cO_t2r) == h
        tOrO = cute.make_rmem_tensor(cO_t2r.shape, Float32)
        n_acc = len(accO)  # num_chunks, or 2 * num_chunks: [O_hi, O_lo] per chunk (fp8_mma)
        tOaccO_ld = [thr_ldO.partition_S(accO[j]) for j in range(n_acc)]
        tOaccO_st = [thr_stO.partition_D(accO[j]) for j in range(n_acc)]

        Consumer = pipeline.PipelineUserType.Consumer
        cs_O = pipeline.make_pipeline_state(Consumer, 1)
        cs_st = pipeline.make_pipeline_state(Consumer, self.num_stages_stats)

        # block 0 needs no rescale, but the wait/release keeps the stats consumer index aligned
        pl_stats.consumer_wait(cs_st)
        pl_stats.consumer_release(cs_st)
        cs_st.advance()

        for _ in cutlass.range(num_n_blocks - 1, unroll=1):
            pl_stats.consumer_wait(cs_st)  # scale of block b, from softmax(b)
            pl_O.consumer_wait(cs_O)  # accumulator state after O^T(b-1)
            fa_sm100_utils.fence_tcgen05_after_thread_sync()
            for j in cutlass.range_constexpr(n_acc):
                cute.copy(thr_ldO, tOaccO_ld[j], tOrO)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(h):
                    tOrO[i] = tOrO[i] * sScale[cs_st.index, cO_t2r[i][1]]
                cute.copy(thr_stO, tOrO, tOaccO_st[j])
                # Complete the store before the next chunk reuses its source registers.
                cute.arch.fence_view_async_tmem_store()
            fa_sm100_utils.fence_tcgen05_before_thread_sync()
            pl_O.consumer_release(cs_O)
            cs_O.advance()
            pl_stats.consumer_release(cs_st)
            cs_st.advance()

        # ---- epilogue: O^T(n-1) is final -----------------------------------------
        pl_O.consumer_wait(cs_O)
        if ctid < h:
            rs = sSmSum[ctid]
            bad = rs == 0.0 or rs != rs
            sInv[ctid] = cute.arch.rcp_approx(rs if not bad else Float32(1.0))
            if const_expr(mLSE is not None):
                lse = (
                    (sSmMax[ctid] * softmax_scale_log2 + cute.math.log2(rs, fastmath=True)) * LN2
                    if not bad
                    else -Float32.inf
                )
                if const_expr(self.is_split_kv):
                    mLSE[m_idx, head_base + ctid, split_idx] = lse
                else:
                    mLSE[m_idx, head_base + ctid] = lse
        bar_corr.arrive_and_wait()

        fa_sm100_utils.fence_tcgen05_after_thread_sync()
        if const_expr(self.fp8_mma):
            tOrO_lo = cute.make_rmem_tensor(cO_t2r.shape, Float32)
        for j in cutlass.range_constexpr(self.num_chunks):
            if const_expr(not self.fp8_mma):
                cute.copy(thr_ldO, tOaccO_ld[j], tOrO)
                cute.arch.fence_view_async_tmem_load()
            else:
                # O = (O_hi + 2^-4 O_lo) * 2^-8: undo the P operand scales (powers of two)
                cute.copy(thr_ldO, tOaccO_ld[2 * j], tOrO)
                cute.copy(thr_ldO, tOaccO_ld[2 * j + 1], tOrO_lo)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(h):
                    tOrO[i] = (tOrO[i] + tOrO_lo[i] * (1.0 / FP8_P_LO_SCALE)) * (
                        1.0 / FP8_P_SCALE
                    )
            for i in cutlass.range_constexpr(h):
                dv = j * self.dv_chunk + cO_t2r[i][0]
                hh = cO_t2r[i][1]
                # split partials are already divided by this split's row sum, as combine expects
                if const_expr(self.is_split_kv):
                    mO[m_idx, dv, head_base + hh, split_idx] = (
                        tOrO[i] * sInv[hh]
                    ).to(self.dtype_O)
                else:
                    mO[m_idx, dv, head_base + hh] = (tOrO[i] * sInv[hh]).to(self.dtype_O)

        # orders the tcgen05 loads before the tmem_bar arrive that gates tcgen05.dealloc
        fa_sm100_utils.fence_tcgen05_before_thread_sync()
