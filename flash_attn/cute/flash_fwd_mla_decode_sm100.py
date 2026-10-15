# Copyright (c) 2026, Colfax International.
"""SM100 sparse-MLA (DSA) decode kernel: the gathered KV tile is the A operand of both MMAs."""

import dataclasses
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
from flash_attn.cute import utils
from flash_attn.cute.utils import get_batch_from_cu_tensor, warp_reduce


LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)

SMEM_CAP_BYTES = 232448  # the measured SM100 dynamic-smem ceiling

# fused combine: how long a finished split CTA waits for its group before leaving its slice to
# the group's last CTA (only reached if the group's CTAs are not all resident at once)
FUSED_COMBINE_WAIT_NS = 20000

# 16-bit KV: h -> (latent ring depth, P^T aliased onto the rope ring); 1 rope stage, 1 P^T.
_RING = {8: (6, False), 16: (6, True), 32: (5, False)}
# fp8 KV with an e4m3 Q: kind::f8f6f4 MMAs straight on the gathered e4m3 stages (16 KB per
# latent stage). h -> latent ring depth; the rope ring has 2 stages and P^T is never aliased.
_RING_FP8_MMA = {8: 12, 16: 12, 32: 10}
# The O^T MMA's B is [P_hi | P_lo]^T in e4m3, stacked along N (N = 2h):
#   P_hi = e4m3(P * 2^8), P_lo = e4m3((P * 2^8 - P_hi) * 2^4),
#   O = (O_hi + 2^-4 * O_lo) / rowsum(P * 2^8)  (relative error ~2^-8, like a bf16 P).
FP8_P_SCALE = 256.0
FP8_P_SCALE_LOG2 = 8.0  # folded into the exp2 offset: the softmax produces P * 2^8 directly
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
        fused_combine: bool = False,
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
        self.early_gather = False
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
        self.num_splits = num_splits
        # split-KV only: the S CTAs of a (token, head group) reduce their partials in-kernel
        # (see _fused_combine) instead of a separate FlashAttentionForwardCombine launch
        self.fused_combine = fused_combine and self.is_split_kv
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
        self.bar_id_gather_go = 4  # early_gather: warp 0 -> gather warps, mbarriers ready
        # early_gather: the setup barrier, which the gather warps only arrive on (bar.arrive on
        # barrier 0 against the count-less bar.sync 0 faults)
        self.bar_id_setup = 5

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
        # fused combine: [epoch, slot | claimed << 8, steal mask, mbarrier]; sits in the padding
        # before the 128-B aligned sRed, so the smem footprint does not change
        comb_words = 4 if self.fused_combine else 0
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
            # fp8_mma: [1] = the softmax's final row sums are written (stats publish early)
            mbar_Q: cute.struct.MemRange[Int64, 2 if self.fp8_mma else 1]
            tmem_holding: cute.struct.MemRange[Int32, 1]
            sComb: cute.struct.MemRange[Int64, comb_words]
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
        mOFinal: Optional[cute.Tensor] = None,     # fused combine: (total_q, h, 512) 16-bit
        mLSEFinal: Optional[cute.Tensor] = None,   # fused combine: (total_q, h) or None
        mSem: Optional[cute.Tensor] = None,        # fused combine: (groups, S + 1) Int64
    ):
        assert not self.is_split_kv or mLSE is not None, (
            "split-KV needs the partial LSE tensor: flash_fwd_combine reduces on it"
        )
        assert self.fused_combine == (mSem is not None and mOFinal is not None)

        dtype = mV.element_type
        assert dtype in (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN), (
            "sparse decode requires 16-bit or e4m3 KV"
        )
        self.fp8_mma = dtype == cutlass.Float8E4M3FN
        # fp8 split-KV grids (few tiles per CTA, so the head start matters): the gather warps
        # start as soon as their own pipelines exist. Unsplit grids keep the CTA-wide setup
        # barrier (an early gather measured 1.8% slower at h8 b128).
        self.early_gather = self.fp8_mma and self.is_split_kv
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
        if const_expr(self.fused_combine):
            mOFinal = cute.make_tensor(
                mOFinal.iterator,
                cute.make_layout(mOFinal.shape, stride=new_stride(mOFinal)),
            )
            # (total_q, h, dv) -> (total_q, dv, h), like the unsplit mO
            mOFinal = cute.make_tensor(mOFinal.iterator, cute.select(mOFinal.layout, mode=[0, 2, 1]))
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
            mOFinal, mLSEFinal, mSem,
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
        mOFinal: Optional[cute.Tensor],
        mLSEFinal: Optional[cute.Tensor],
        mSem: Optional[cute.Tensor],
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

        valid_len = None
        idx_first = None
        if const_expr(not self.fp8_mma):
            batch_idx = get_batch_from_cu_tensor(m_idx, mCuSeqlensQ)
        else:
            # loads that depend only on the CTA index go out first and are consumed after the
            # setup barrier, so their latency hides under the barrier init and TMEM alloc
            if const_expr(mTopkValidLen is not None):
                valid_len = mTopkValidLen[m_idx]
            if const_expr(not self.is_split_kv):
                idx_lo = Int32(0)
            else:
                idx_lo = split_idx * Int32(self.blocks_per_split * self.tile_n)
            idx_first = Int32(0)
            if warp_idx >= self.gather_warp_lo:
                # the gather's first top-k index (CpasyncGatherKVManager.load_index_topk map)
                idx_first = mIndexTopk[idx_lo + self._gather_index_row(tidx - self.gather_tid_lo), m_idx]
            # batch lookup: decode requests carry 1 or 2 queries (2 with MTP), so batch = m_idx
            # and m_idx / 2 are checked (four independent loads) before the binary search, and
            # the K lengths of batch m_idx are loaded speculatively alongside; all resolved
            # after the setup barrier (_resolve_batch)
            last_b = cute.size(mCuSeqlensQ) - 2
            g1 = min(m_idx, last_b)
            g2 = min(m_idx // 2, last_b)
            guess = (
                g1, g2,
                mCuSeqlensQ[g1], mCuSeqlensQ[g1 + 1], mCuSeqlensQ[g2], mCuSeqlensQ[g2 + 1],
            )
            seqlen_spec = SeqlenInfoQK.create(
                g1, Int32(1), Int32(1),
                mCuSeqlensQ=None, mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=None, mSeqUsedK=mSeqUsedK,
                tile_m=self.tile_n, tile_n=self.tile_n,
            )
        if const_expr(not self.fp8_mma):
            seqlen = SeqlenInfoQK.create(
                batch_idx, Int32(1), Int32(1),
                mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=None, mSeqUsedK=mSeqUsedK,
                tile_m=self.tile_n, tile_n=self.tile_n,
            )
            seqlen_k_limit = seqlen.seqlen_k

        if const_expr(not self.fp8_mma):
            n_block_lo, num_n_blocks = self._block_range(
                None if const_expr(mTopkValidLen is None) else mTopkValidLen[m_idx], split_idx
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
        if const_expr(self.early_gather):
            # the gather warps need only these two pipelines: release them as soon as their
            # mbarriers are initialized, not after every other init and the TMEM alloc
            bar_go = pipeline.NamedBarrier(
                barrier_id=self.bar_id_gather_go, num_threads=32 + self.num_gather_threads
            )
            if warp_idx == 0:
                cute.arch.mbarrier_init_fence()
                bar_go.arrive()

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
        mbar_sum = None
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_Q, self.num_corr_threads)
        if const_expr(self.fp8_mma):
            mbar_sum = mbar_Q + 1
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_sum, self.num_softmax_threads)
        if const_expr(self.fused_combine):
            if tidx == 0:
                # one-thread broadcast barrier of the fused combine (phase 0: ticket, 1: mask)
                cute.arch.mbarrier_init(storage.sComb.data_ptr() + 3, 1)

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
        if const_expr(not self.early_gather):
            cute.arch.barrier()
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
        else:
            # gather warps: no TMEM, so they only arrive on the setup and TMEM barriers (their
            # next use of either is the teardown's tmem_bar) and wait for their pipelines
            tmem_addr = Int32(0)
            if warp_idx < self.gather_warp_lo:
                cute.arch.barrier(barrier_id=self.bar_id_setup, number_of_threads=self.num_threads)
                tmem.wait_for_alloc()
                tmem_addr = storage.tmem_holding.get_tensor(cute.make_layout(1))[0]
            else:
                cute.arch.barrier_arrive(
                    barrier_id=self.bar_id_setup, number_of_threads=self.num_threads
                )
                tmem_bar.arrive()
                bar_go.arrive_and_wait()
            tmem_ptr = cute.make_ptr(
                Float32, tmem_addr, cute.AddressSpace.tmem, assumed_align=16
            )
        if const_expr(self.fp8_mma):
            n_block_lo, num_n_blocks = self._block_range(valid_len, split_idx)
            batch_idx, seqlen = self._resolve_batch(
                m_idx, guess, seqlen_spec, mCuSeqlensQ, mCuSeqlensK, mSeqUsedK
            )
            seqlen_k_limit = seqlen.seqlen_k

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
                softmax_scale_log2, n_block_lo, num_n_blocks, mbar_sum,
            )
        elif warp_idx < self.mma_warp_id:
            self.corr_epilogue_loop(
                mQ, mQv, mO, mLSE, mma_O_acc, accO, sQtC, sQtR,
                sScale, sSmMax, sSmSum, sInv, pl_O, pl_stats, mbar_Q,
                tidx, head_kv, m_idx, hg_idx, split_idx,
                softmax_scale_log2, num_n_blocks, mbar_sum,
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
                dtype, n_block_lo, num_n_blocks, idx_first,
            )

        if const_expr(self.fused_combine):
            self._fused_combine(
                mO, mLSE, mOFinal, mLSEFinal, mSem,
                storage.sComb.get_tensor(cute.make_layout(4)),
                tidx, warp_idx, head_kv, m_idx, hg_idx, mV.shape[2],
            )

        # ==== TMEM teardown (all 512 threads) =======================================
        tmem.relinquish_alloc_permit()
        tmem_bar.arrive_and_wait()
        tmem.free(tmem_ptr)

    # ============================================== fused split-KV combine (all warps)
    @cute.jit
    def _fused_combine(
        self, mO, mLSE, mOFinal, mLSEFinal, mSem, sComb,
        tidx, warp_idx, head_kv, m_idx, hg_idx, nheads_kv,
    ):
        """Reduce the S split partials of this CTA's (token, head group) in-kernel.

        The group's partial O (h x 512 fp32 per split) is cut into S equal slices. After its own
        partial is in gmem, every split CTA takes a ticket from the group counter; the ticket
        number is the slice it owns. A CTA that is not last waits (bounded by
        FUSED_COMBINE_WAIT_NS) until all S tickets are out, posts "claimed" for its slot and
        reduces its slice, or on timeout posts "abandoned" and leaves. The last CTA reduces its
        own slice, then reads the S-1 posts (each waiter is resident and decides within its
        timeout) and reduces the abandoned slices. Nothing waits unboundedly, so the scheme is
        correct even when a group's CTAs are not co-resident.

        Words are never reset: each launch adds exactly S to the group counter, so
        epoch = ticket // S + 1 numbers the launch; slot posts are 2 * epoch (claimed) or
        2 * epoch + 1 (abandoned) and only grow.

        Memory model (PTX ISA 9.3):
          * the corr threads' partial O / LSE stores -> bar.sync on bar_corr, which
            synchronizes (8.9.4 item 2) -> thread 0's ticket.
          * ticket = atom.acq_rel.gpu on the counter: a release pattern (8.8) covering those
            stores (causality is transitive, 8.9.5), and an acquire of every earlier ticket's
            release (observation order through the RMW chain, 8.9.2 item 2; 8.9.4 item 4).
          * waiter: ld.acquire.gpu reading the counter value epoch * S, written by the last
            ticket, synchronizes with all S releases.
          * thread 0 -> mbarrier.arrive (.release.cta) -> every thread's mbarrier wait (.acquire.cta)
            -> its weak loads of the partials (causality order is transitive, 8.9.5).
            All accesses are generic proxy to the same addresses (proxy-preserved, 8.9.5).
          * slot posts carry no data: the last CTA's own acquire already covers all partials.
        """
        S = const_expr(self.num_splits)
        h = const_expr(self.h)
        head_base = head_kv * self.qhead_per_kvhead + hg_idx * h
        gid = (m_idx * nheads_kv + head_kv) * self.num_head_groups + hg_idx
        ctr_ptr = mSem[gid, None].iterator  # [0] ticket counter, [1 + slot] slot posts
        # corr thread 0 (and later warp 0 of the last CTA) publishes through sComb and arrives
        # on this mbarrier (arrive is .release.cta, the waits are .acquire.cta: PTX 9.3
        # 9.7.14.16.16 / 9.7.14.16.19), so all 512 threads read sComb after the write
        mbar_comb = sComb.iterator + 3

        if warp_idx >= self.corr_warp_lo and warp_idx < self.mma_warp_id:
            ctid = tidx - self.corr_tid_lo
            # every corr thread's partial O and LSE stores precede the ticket
            pipeline.NamedBarrier(
                barrier_id=self.bar_id_corr, num_threads=self.num_corr_threads
            ).arrive_and_wait()
            if ctid == 0:
                ticket = cute.arch.atomic_add(ctr_ptr, Int64(1), sem="acq_rel", scope="gpu")
                epoch = ticket // S + 1
                slot = ticket - (epoch - 1) * S
                claimed = Int64(1)
                if slot != S - 1:
                    target = epoch * S
                    t_start = cute.arch.globaltimer()
                    seen = cute.arch.load(ctr_ptr, Int64, sem="acquire", scope="gpu")
                    while seen < target and cute.arch.globaltimer() - t_start < FUSED_COMBINE_WAIT_NS:
                        seen = cute.arch.load(ctr_ptr, Int64, sem="acquire", scope="gpu")
                    claimed = Int64(1) if seen >= target else Int64(0)
                    cute.arch.store(
                        ctr_ptr + 1 + Int32(slot), 2 * epoch + 1 - claimed,
                        sem="relaxed", scope="gpu",
                    )
                sComb[0] = epoch
                sComb[1] = slot | (claimed << 8)
                cute.arch.mbarrier_arrive(mbar_comb)
        cute.arch.mbarrier_wait(mbar_comb, Int32(0))
        epoch = sComb[0]
        packed = Int32(sComb[1])
        slot = packed & 0xFF
        if (packed >> 8) != 0:
            self._combine_slice(mO, mLSE, mOFinal, mLSEFinal, m_idx, head_base, slot, tidx)
        if slot == S - 1:
            # last CTA: collect the waiters' posts, then reduce every abandoned slice
            if warp_idx == 0:
                abandoned = False
                if tidx < S - 1:
                    post = cute.arch.load(ctr_ptr + 1 + tidx, Int64, sem="relaxed", scope="gpu")
                    while post < 2 * epoch:
                        post = cute.arch.load(ctr_ptr + 1 + tidx, Int64, sem="relaxed", scope="gpu")
                    abandoned = post == 2 * epoch + 1
                mask = cute.arch.vote_ballot_sync(abandoned)
                if tidx == 0:
                    sComb[2] = Int64(mask)
                    cute.arch.mbarrier_arrive(mbar_comb)
            cute.arch.mbarrier_wait(mbar_comb, Int32(1))
            mask = Int32(sComb[2])
            if mask != 0:
                for sl in cutlass.range(S - 1, unroll=1):
                    if (mask >> sl) & 1 != 0:
                        self._combine_slice(
                            mO, mLSE, mOFinal, mLSEFinal, m_idx, head_base, sl, tidx
                        )

    @cute.jit
    def _combine_slice(self, mO, mLSE, mOFinal, mLSEFinal, m_idx, head_base, sl, tidx):
        """out = sum_s exp(lse_s - lse) * O_s over one slice (FlashAttentionForwardCombine math).

        All loads are issued before any arithmetic so each thread waits one L2 round trip.
        """
        S = const_expr(self.num_splits)
        slice_vecs = const_expr(self.h * self.hdimv // (4 * S))  # 4-float vectors per slice
        nvec = const_expr(cute.ceil_div(slice_vecs, self.num_threads))
        full = const_expr(slice_vecs % self.num_threads == 0)
        rO = cute.make_rmem_tensor((4, S, nvec), Float32)
        rL = cute.make_rmem_tensor((S, nvec), Float32)
        for k in cutlass.range_constexpr(nvec):
            i = tidx + k * self.num_threads
            e = (sl * slice_vecs + i) * 4
            hh = head_base + e // self.hdimv
            dv = e % self.hdimv
            if const_expr(full) or i < slice_vecs:
                for s in cutlass.range_constexpr(S):
                    rL[s, k] = mLSE[m_idx, hh, s]
                    cute.autovec_copy(
                        cute.local_tile(mO[m_idx, None, hh, s], (4,), (dv // 4,)), rO[None, s, k]
                    )
        for k in cutlass.range_constexpr(nvec):
            i = tidx + k * self.num_threads
            e = (sl * slice_vecs + i) * 4
            hh = head_base + e // self.hdimv
            dv = e % self.hdimv
            if const_expr(full) or i < slice_vecs:
                lse_max = rL[0, k]
                for s in cutlass.range_constexpr(1, S):
                    lse_max = cute.arch.fmax(lse_max, rL[s, k])
                lse_max_cur = lse_max if lse_max != -Float32.inf else Float32(0.0)
                lse_sum = Float32(0.0)
                for s in cutlass.range_constexpr(S):
                    rL[s, k] = cute.math.exp2(rL[s, k] * LOG2_E - lse_max_cur * LOG2_E, fastmath=True)
                    lse_sum += rL[s, k]
                bad = lse_sum == 0.0 or lse_sum != lse_sum
                inv = Float32(0.0) if bad else 1.0 / lse_sum
                # every split partial is finite (an empty row stores O = 0 with lse = -inf, so
                # its weight is exactly 0), so no per-split skip is needed
                acc = cute.make_rmem_tensor(4, Float32)
                acc.fill(0.0)
                for s in cutlass.range_constexpr(S):
                    acc.store(acc.load() + (rL[s, k] * inv) * rO[None, s, k].load())
                gOut = cute.local_tile(mOFinal[m_idx, None, hh], (4,), (dv // 4,))
                gOut.store(acc.load().to(mOFinal.element_type))
                if const_expr(mLSEFinal is not None):
                    if dv == 0:
                        mLSEFinal[m_idx, hh] = cute.math.log(lse_sum, fastmath=True) + lse_max

    @cute.jit
    def _block_range(self, valid_len, split_idx):
        """(first block, block count) of this CTA's top-k window; valid_len None = all valid."""
        if const_expr(valid_len is None):
            n_valid_blocks = Int32(self.n_blocks_full)
        else:
            n_valid_blocks = max(
                Int32(1),
                min(
                    cute.ceil_div(valid_len, self.tile_n),
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
        return n_block_lo, num_n_blocks

    @cute.jit
    def _resolve_batch(self, m_idx, guess, seqlen_spec, mCuSeqlensQ, mCuSeqlensK, mSeqUsedK):
        """fp8_mma batch lookup: the guessed bracket, else m_idx / 2's, else the binary search.
        Only seqlen_k and offset_k of the returned SeqlenInfoQK are meaningful (all this kernel
        reads); on a miss of batch = m_idx they are loaded again for the found batch."""
        g1, g2, lo1, hi1, lo2, hi2 = guess
        batch_idx = g1
        seqlen_k = seqlen_spec.seqlen_k
        offset_k = seqlen_spec.offset_k
        if not (lo1 <= m_idx and m_idx < hi1):
            batch_idx = g2
            if not (lo2 <= m_idx and m_idx < hi2):
                batch_idx = get_batch_from_cu_tensor(m_idx, mCuSeqlensQ)
            found = SeqlenInfoQK.create(
                batch_idx, Int32(1), Int32(1),
                mCuSeqlensQ=None, mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=None, mSeqUsedK=mSeqUsedK,
                tile_m=self.tile_n, tile_n=self.tile_n,
            )
            seqlen_k = found.seqlen_k
            if const_expr(mCuSeqlensK is not None):
                offset_k = found.offset_k
        return batch_idx, dataclasses.replace(seqlen_spec, seqlen_k=seqlen_k, offset_k=offset_k)

    # ================================================================ gather warps
    @cute.jit
    def gather_loop(
        self, mIndexTopk, mK, mV, sLatK, sRope, pl_lat, pl_rope,
        tidx, warp_idx, m_idx, seqlen, batch_idx, head_kv, seqlen_k_limit, dtype,
        n_block_lo, num_n_blocks, idx_first=None,
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

        if const_expr(idx_first is not None):
            # fp8_mma: block 0's index was loaded before the setup barrier
            assert mgr.topk_indices_per_thread == 1
            assert mgr.gmem_threads_per_row == math.gcd(self.hdim_rope, self.dv_chunk, 128) // 16
            mgr.rTopkHalf[0] = idx_first
        for n_block in cutlass.range(num_n_blocks, unroll=1):
            if const_expr(idx_first is None):
                mgr.load_index_topk(n_block_lo + n_block, transpose=False)
            elif n_block > 0:
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

    def _gather_index_row(self, gtid):
        """Tile row whose top-k index gather thread gtid loads (CpasyncGatherKVManager's
        load_index_topk map with one index per thread)."""
        # e4m3 rows: 16 B copies over gcd(64 rope, 128 latent-chunk) elements
        tpr = math.gcd(self.hdim_rope, self.dv_chunk, 128) // 16
        return (gtid % tpr) * (self.num_gather_threads // tpr) + gtid // tpr

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
        softmax_scale_log2, n_block_lo, num_n_blocks, mbar_sum=None,
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
        if const_expr(mbar_sum is not None):
            # the stats pipeline publishes before the row sum, so the epilogue waits on this
            cute.arch.mbarrier_arrive(mbar_sum)
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
            # fp8_mma: S^T stage cs_S sits h TMEM columns per stage on (branch-free select)
            cute.copy(
                thr_ldS,
                cute.make_tensor(tSaccS.iterator + cs_S.index * self.h, tSaccS.layout),
                tSrS,
            )
        cute.arch.fence_view_async_tmem_load()
        fa_sm100_utils.fence_tcgen05_before_thread_sync()
        pl_S.consumer_release(cs_S)
        cs_S.advance()

        for i in cutlass.range_constexpr(h):
            tSrS[i] = tSrS[i] if key_valid else -Float32.inf

        # ---- block max per head across the 128 lanes: warp butterfly + smem -------
        for i in cutlass.range_constexpr(h):
            if const_expr(self.fp8_mma):
                # one redux.sync.max.f32 per value; the same max as the fmax butterfly
                tSrP[i] = cute.arch.warp_redux_sync(tSrS[i], "fmax")
            else:
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
        if const_expr(self.fp8_mma):
            # publish the rescale factor now: the correction's O rescale (which O^T(b) waits
            # for) overlaps this block's exp / P^T / row sum instead of following them
            pl_stats.producer_commit(ps_st)
            rMax = cute.make_rmem_tensor((h,), Float32)
            cute.autovec_copy(sSmMax, rMax)  # LDS.128 instead of h scalar broadcasts

        # ---- P = exp2(S * scale_log2 - m_new * scale_log2) ------------------------
        for i in cutlass.range_constexpr(h):
            # Keep the true running max; only exponentiation needs a finite empty-row bias.
            if const_expr(self.fp8_mma):
                row_max = rMax[cS_t2r[i][1]]
            else:
                row_max = sSmMax[cS_t2r[i][1]]
            safe_max = row_max if row_max != -Float32.inf else Float32(0.0)
            bias = -safe_max * softmax_scale_log2
            if const_expr(self.fp8_mma):
                # fp8 P^T operand: exp2(... + 8) = P * 2^8, exact; the row sum carries the 2^8
                # too, so it cancels in O / rowsum and the LSE subtracts it
                bias = bias + FP8_P_SCALE_LOG2
            tSrP[i] = cute.math.exp2(tSrS[i] * softmax_scale_log2 + bias, fastmath=True)

        if const_expr(self.fp8_mma):
            # P^T goes out before the row-sum shuffles: O^T waits for P^T, not the sum.
            # This key's row of [P_hi | P_lo]^T: 2h contiguous e4m3 (heads, then heads).
            pl_P.producer_acquire(ps_P)
            pt_nd = cute.composition(sPt[None, None, None, ps_P.index], pt_layout)
            rHi = cute.make_rmem_tensor(tSrP.shape, cutlass.Float8E4M3FN)
            rHi.store(tSrP.load().to(cutlass.Float8E4M3FN))
            rRow = cute.make_rmem_tensor((2 * h,), cutlass.Float8E4M3FN)
            rLo = cute.make_rmem_tensor(tSrP.shape, cutlass.Float8E4M3FN)
            rLo.store(((tSrP.load() - rHi.load().to(Float32)) * FP8_P_LO_SCALE).to(
                cutlass.Float8E4M3FN
            ))
            for i in cutlass.range_constexpr(h):
                rRow[cS_t2r[i][1]] = rHi[i]
                rRow[h + cS_t2r[i][1]] = rLo[i]
            cute.autovec_copy(rRow, pt_nd[None, key_row])
            cute.arch.fence_proxy("async.shared", space="cta")
            pl_P.producer_commit(ps_P)
            ps_P.advance()

        # ---- row sum per head across the 128 lanes -------------------------------
        for i in cutlass.range_constexpr(h):
            tSrS[i] = warp_reduce(tSrP[i], operator.add)
        if lane_idx == 0:
            for i in cutlass.range_constexpr(h):
                sRed[warp_idx, i] = tSrS[i]

        if const_expr(not self.fp8_mma):
            # ---- P^T staging: this thread's key, all h heads, MN-major ---------------
            pl_P.producer_acquire(ps_P)
            pt_nd = cute.composition(sPt[None, None, None, ps_P.index], pt_layout)
            for i in cutlass.range_constexpr(h):
                pt_nd[cS_t2r[i][1], key_row] = tSrP[i].to(sPt.element_type)
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
        if const_expr(not self.fp8_mma):
            pl_stats.producer_commit(ps_st)
        ps_st.advance()
        return cs_S, ps_P, ps_st

    @cute.jit
    def _epilogue_fp8_halves(self, mO, ab_dtype, accO, sInv, ctid, m_idx, head_base, split_idx):
        """fp8_mma epilogue, h / 2 accumulator columns per TMEM load: O = O_hi + 2^-4 O_lo,
        times this split's 1 / rowsum, to mO."""
        h2 = self.h // 2
        mma_h = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, OperandMajorMode.MN, OperandMajorMode.K, self.dtype_acc,
            CtaGroup.ONE, (self.dv_chunk, h2),
        )
        thr_h = mma_h.get_slice(0)
        lay_h = thr_h.make_fragment_C(thr_h.partition_shape_C((self.dv_chunk, h2))).layout
        view = lambda j, k: cute.make_tensor(accO[j].iterator + k * h2, lay_h)[(None, None), 0, 0]
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(h2)), Float32
        )
        thr_ld = tcgen05.make_tmem_copy(ld_atom, view(0, 0)).get_slice(ctid)
        c_t2r = thr_ld.partition_D(
            thr_h.partition_C(cute.make_identity_tensor((self.dv_chunk, h2)))[(None, None), 0, 0]
        )
        assert cute.size(c_t2r) == h2
        rHi = cute.make_rmem_tensor(c_t2r.shape, Float32)
        rLo = cute.make_rmem_tensor(c_t2r.shape, Float32)
        # heads outer: only h / 2 heads' output addresses and 1 / rowsum values stay live
        for k in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(self.num_chunks):
                cute.copy(thr_ld, thr_ld.partition_S(view(2 * j, k)), rHi)
                cute.copy(thr_ld, thr_ld.partition_S(view(2 * j + 1, k)), rLo)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(h2):
                    dv = j * self.dv_chunk + c_t2r[i][0]
                    hh = k * h2 + c_t2r[i][1]
                    o = (rHi[i] + rLo[i] * (1.0 / FP8_P_LO_SCALE)) * sInv[hh]
                    if const_expr(self.is_split_kv):
                        mO[m_idx, dv, head_base + hh, split_idx] = o.to(self.dtype_O)
                    else:
                        mO[m_idx, dv, head_base + hh] = o.to(self.dtype_O)

    # =========================================== correction + Q^T load + epilogue
    def _qt_fp8_units(self):
        """e4m3 Q^T is K-major like q in gmem (each head's 576 dims contiguous): it is moved in
        16 B units, 36 per head (32 latent + 4 rope), unit u = ctid + n_corr * it."""
        upr = self.hdim_total // 16
        return upr, (upr * self.h + self.num_corr_threads - 1) // self.num_corr_threads

    @cute.jit
    def _qt_fp8_load(self, mQ, mQv, rQ, ctid, m_idx, head_base):
        """LDG.128 this thread's Q^T units into rQ[16, it]."""
        upr, n_it = self._qt_fp8_units()
        lat_u = self.hdimv // 16
        gQv = mQv[m_idx, None, None]  # (512, h_total)
        gQr = mQ[m_idx, None, None]  # (64,  h_total)
        for it in cutlass.range_constexpr(n_it):
            u = ctid + self.num_corr_threads * it
            if const_expr((it + 1) * self.num_corr_threads <= upr * self.h) or u < upr * self.h:
                n = u // upr
                k16 = u % upr
                addr = utils.elem_pointer(gQr, (16 * (k16 - lat_u), head_base + n)).toint()
                if k16 < lat_u:
                    addr = utils.elem_pointer(gQv, (16 * k16, head_base + n)).toint()
                src = cute.make_tensor(
                    cute.make_ptr(rQ.element_type, addr, cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((16,)),
                )
                cute.autovec_copy(src, rQ[None, it])

    @cute.jit
    def _qt_fp8_store(self, rQ, sQtC, sQtR, qtc_layout, qtr_layout, ctid):
        upr, n_it = self._qt_fp8_units()
        lat_u = self.hdimv // 16
        cu = self.dv_chunk // 16
        for it in cutlass.range_constexpr(n_it):
            u = ctid + self.num_corr_threads * it
            if const_expr((it + 1) * self.num_corr_threads <= upr * self.h) or u < upr * self.h:
                n = u // upr
                k16 = u % upr
                if k16 < lat_u:
                    c = k16 // cu
                    dst = cute.zipped_divide(
                        cute.composition(sQtC[None, None, None, c], qtc_layout), (1, 16)
                    )
                    cute.autovec_copy(rQ[None, it], dst[(0, None), (n, k16 % cu)])
                else:
                    dst = cute.zipped_divide(
                        cute.composition(sQtR[None, None, None, 0], qtr_layout), (1, 16)
                    )
                    cute.autovec_copy(rQ[None, it], dst[(0, None), (n, k16 - lat_u)])

    @cute.jit
    def corr_epilogue_loop(
        self, mQ, mQv, mO, mLSE, mma_O, accO, sQtC, sQtR,
        sScale, sSmMax, sSmSum, sInv, pl_O, pl_stats, mbar_Q,
        tidx, head_kv, m_idx, hg_idx, split_idx,
        softmax_scale_log2, num_n_blocks, mbar_sum=None,
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
        if const_expr(not self.fp8_mma):
            for c in cutlass.range_constexpr(self.num_chunks):
                qt_nd = cute.composition(sQtC[None, None, None, c], qtc_layout)
                for n in cutlass.range_constexpr(h):
                    qt_nd[n, ctid] = gQv[c * self.dv_chunk + ctid, head_base + n]
            qt_rope = cute.composition(sQtR[None, None, None, 0], qtr_layout)
            if ctid < self.hdim_rope:
                for n in cutlass.range_constexpr(h):
                    qt_rope[n, ctid] = gQr[ctid, head_base + n]
        else:
            # e4m3 Q^T: 16 B units, LDG.128 -> STS.128
            rQ = cute.make_rmem_tensor((16, self._qt_fp8_units()[1]), mQv.element_type)
            self._qt_fp8_load(mQ, mQv, rQ, ctid, m_idx, head_base)
            self._qt_fp8_store(rQ, sQtC, sQtR, qtc_layout, qtr_layout, ctid)
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
        if const_expr(mbar_sum is not None):
            cute.arch.mbarrier_wait(mbar_sum, Int32(0))
        if ctid < h:
            rs = sSmSum[ctid]
            bad = rs == 0.0 or rs != rs
            sInv[ctid] = cute.arch.rcp_approx(rs if not bad else Float32(1.0))
            if const_expr(mLSE is not None):
                if const_expr(not self.fp8_mma):
                    lse = (
                        (sSmMax[ctid] * softmax_scale_log2 + cute.math.log2(rs, fastmath=True)) * LN2
                        if not bad
                        else -Float32.inf
                    )
                else:
                    # the row sum is of P * 2^8
                    lse = (
                        (sSmMax[ctid] * softmax_scale_log2 + cute.math.log2(rs, fastmath=True)
                         - FP8_P_SCALE_LOG2) * LN2
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
        if const_expr(self.fp8_mma and h > 16):
            # h = 32: O_hi + O_lo of all h columns plus the h hoisted 1/rowsum values exceed
            # the 128-register budget (STL/LDL); read the accumulators h / 2 columns at a time
            self._epilogue_fp8_halves(mO, mma_O.op.a_dtype, accO, sInv, ctid, m_idx, head_base, split_idx)
        for j in cutlass.range_constexpr(0 if const_expr(self.fp8_mma and h > 16) else self.num_chunks):
            if const_expr(not self.fp8_mma):
                cute.copy(thr_ldO, tOaccO_ld[j], tOrO)
                cute.arch.fence_view_async_tmem_load()
            else:
                # O = O_hi + 2^-4 O_lo; the 2^8 of P cancels against the row sum
                cute.copy(thr_ldO, tOaccO_ld[2 * j], tOrO)
                cute.copy(thr_ldO, tOaccO_ld[2 * j + 1], tOrO_lo)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(h):
                    tOrO[i] = tOrO[i] + tOrO_lo[i] * (1.0 / FP8_P_LO_SCALE)
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
