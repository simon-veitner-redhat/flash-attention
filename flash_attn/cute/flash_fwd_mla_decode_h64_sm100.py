# Copyright (c) 2026, Colfax International.
"""SM100 sparse-MLA (DSA) decode for 64 query heads per KV head, heads on M.

The structure follows FlashMLA's sm100 head64 sparse forward: one CTA per query token, all 64 heads
on the M dimension of `tcgen05.mma.ws` (M=64, TMEM layout E), 64-row KV blocks, Q resident in TMEM.

Grid: one CTA per (token, KV split), split index fastest.  A token's valid blocks
n_valid = min(ceil(valid / 64), topk / 64) are divided evenly: per = ceil(n_valid / S), split s runs
blocks [s * per, s * per + cnt) with cnt = clamp(n_valid - s * per, 0, per), so empty splits are
trailing.  How the S splits of a token meet depends on S (combines_in_kernel):
  S = 8, 16  each split CTA writes O / li in fp32 and the natural-log LSE to the partial tensors
             that flash_fwd_combine reduces.
  S = 2-4    the S CTAs of a token form a cluster (cluster rank == split index; per is at least
             MIN_BLK_PER_SPLIT there) and combine in the kernel: each non-empty split stages
             O / li in fp32 in latent slots 0-1 and its natural-log LSE in the row-max buffer, the
             cluster barrier publishes both, and each of the S_ne = ceil(n_valid / per) non-empty
             CTAs reduces its share of the 64 heads over DSMEM (ld.shared::cluster) into the final
             16-bit O and LSE.  A second cluster barrier keeps every stage alive until all peers
             have read it.  Trailing empty splits exit at once: a cluster barrier waits only for
             non-exited threads (PTX 9.7.14.3), and every CTA derives S_ne from the same valid
             length, so an exited split is never read.  With S_ne == 1 the one non-empty split
             runs the unsplit epilogue (no stage, no cluster barrier); with n_valid == 0, split 0
             writes O = 0 and LSE = -inf as the unsplit kernel does.

Warp roles (384 threads, 1 CTA per SM):
  0-3   softmax: thread t owns head row t % 64 and key columns 32*(t // 64) .. +31 of each block;
        then the epilogue
  4-7   latent producers: Q latent staging into latent slot 1, then the latent gather (cp.async
        16 B), arriving on kv_ready after chunks 0-3 and on kv_ready_hi after all 8
  8     MMA (one elected lane), TMEM alloc and dealloc
  9     validity masks: lanes 0-7, 8 indices each, one 64-bit mask per block into a 3-deep ring
  10-11 rope producers: Q rope staging into the P buffer, then the rope gather (cp.async 16 B)

Per block k (slot = k % 3; each latent slot has two kv_ready barriers, chunks 0-3 and 4-7):
  QK  S = Q K^T, the "dual gemm": .ws N=128 with Q in TMEM.  TMEM lanes 0-63 hold Q latent chunk 2p
      and lanes 64-127 chunk 2p+1 of each chunk pair p, so each MMA computes two independent 64x64
      products into lanes 0-63 / 64-127 of the S columns; S = the sum of the two lane halves.  The
      rope part runs first (2 k-steps over the SW64 re-view of the rope tile) so its rope slot frees
      early; then 16 latent k-steps.  Two rope slots let the rope gather of block k+1 start before
      block k's rope QK, so the rope gather latency is not part of the per-block period.
  PV  O += P V: two .ws SS MMAs N=256 (4 k-steps each).  A = P bf16 from smem (SW128 K-major),
      B = V read MN-major from the same latent slot (V aliases K; the latent is gathered once).

TMEM (512 columns allocated, 464 used): O 0-255, Q latent 256-383, Q rope 384-399, S 400-463.
Smem: 3 latent slots x 64 KiB (SW128 K-major, 8 chunks of 64 rows x 128 B), 2 rope slots x 8 KiB
(SW64, 2 halves of 64 rows x 64 B), P 8 KiB, partial-S exchange 8 KiB (two rounds per block), row
max / li 2 x 512 B, masks, mbarriers.  Q latent is staged in latent slot 1 and Q rope in the P
buffer, both copied to TMEM with tcgen05.cp.128x256b.  The epilogue stages O in latent slot 0
(16-bit, unsplit) or slots 0-1 (fp32, split: stored as the partial, or read by the cluster peers).

Byte address of 16 B unit u of row r (_sw128 and _sw64; the Q stages use the gather's formulas):
  latent chunk c: c * 8192 + r * 128 + ((u ^ (r % 8)) * 16)            SW128, u < 8
  rope half kh:   kh * 4096 + r * 64 + ((u ^ ((r >> 1) % 4)) * 16)     SW64,  u < 4
  P row m:        m * 128 + ((u ^ (m % 8)) * 16)                       SW128, u < 8
Shared-memory descriptors (PTX ISA 9.7.17.4.1): lo = (addr >> 4) | (LBO >> 4) << 16,
hi = SBO >> 4 | 1 << 14 | layout << 29, layout SW128 = 2 / SW64 = 4.  K-major operands use SBO 1024
(SW128) or 512 (SW64); V is read MN-major with LBO 8192 (the chunk stride) and SBO 1024.

Masking: a KV row is valid iff 0 <= idx < seqlen_k.  gather_kv_valid_length only bounds the number
of blocks.  Invalid rows gather with cp.async src-size 0 (zero fill, address of row 0), so V of a
masked row is 0, never stale smem.  A row with no valid key gets O = 0 and LSE = -inf.  A CTA with
no block (valid length 0, or a trailing split) takes the empty exit whatever its indices hold,
before any barrier or TMEM allocation: unsplit it writes O = 0 and LSE = -inf, split only the
partial LSE = -inf (flash_fwd_combine skips a split whose LSE is -inf, so its O partial is never
read), and in a cluster only split 0 of an empty token writes (O = 0, LSE = -inf); the others exit.
"""

from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import tcgen05
from cutlass.base_dsl.typing import Vector

import flash_attn.cute.blackwell_helpers as fa_sm100_utils
import flash_attn.cute.mma_sm100_desc as sm100_desc
from flash_attn.cute.flash_fwd_mla_decode_sm100 import LN2, LOG2_E, SMEM_CAP_BYTES
from flash_attn.cute.mma_sm100_desc import Major
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.utils import get_batch_from_cu_tensor


# ring depths: the latent ring (kv_ready, kv_ready_hi, qk_done, sv_done, masks) has NUM_SLOTS slots,
# the rope ring (rope_ready, qk_rope_done) NUM_ROPE_SLOTS
NUM_SLOTS = 3
NUM_ROPE_SLOTS = 2
# FlashMLA's initial running max: finite, so -inf - m_i never forms a NaN
_MAX_INIT = -1e30
# lazy O rescale: only when the block max exceeds the running max by more than 2^6
_RESCALE_THRESHOLD = 6.0
# in-kernel combine (2-4 splits): at least this many 64-row blocks per non-empty split.  Short
# valid lengths then use fewer splits (256 keys: one, which takes the unsplit epilogue), since a
# peer's stage costs about 1 us per split over DSMEM
MIN_BLK_PER_SPLIT = 4


def _ring(k, depth=NUM_SLOTS):
    """(slot, phase) of block k in a depth-deep ring."""
    return k % depth, (k // depth) & 1


# UMMA shared-memory descriptors (mma_sm100_desc): the constant part from the canonical layout of
# the operand tile in 16 B units, split into the two 32-bit words the kernel adds offsets to.
def _desc_base(shape, stride, swizzle_bits: int, major: Major):
    """(low-word bits, high word) of the descriptor of a tile laid out as shape:stride (16 B
    units) with Swizzle<swizzle_bits, 4, 3>, without the start address."""
    base = sm100_desc.make_smem_desc_base(
        cute.make_layout(shape, stride=stride), cute.make_swizzle(swizzle_bits, 4, 3), major
    )
    return base & 0xFFFFFFFF, base >> 32


def _desc_lo(addr, base_lo: int):
    """Low word of a descriptor: start address (make_smem_desc_start_addr) | the LBO bits."""
    ptr = cute.make_ptr(Uint32, addr, cute.AddressSpace.smem, assumed_align=16)
    return sm100_desc.make_smem_desc_start_addr(ptr) | base_lo


# ------------------------------------------------------------------ addressing
@cute.jit
def _sw128(r, u, row_bytes: cutlass.Constexpr = 128) -> Int32:
    """Byte offset of 16 B unit u of row r, units XOR-swizzled by r % 8 (SW128 for 128 B rows)."""
    return r * row_bytes + ((u ^ (r % 8)) * 16)


@cute.jit
def _sw64(r, u) -> Int32:
    """Byte offset of 16 B unit u (< 4) of row r in a SW64 tile of 64 B rows."""
    return r * 64 + ((u ^ ((r >> 1) % 4)) * 16)


@cute.jit
def _kv_valid(g, seqlen_k) -> cutlass.Boolean:
    return g >= 0 and g < seqlen_k


@cute.jit
def _gather_row(g, seqlen_k):
    """(row, cp.async src-size) of top-k entry g: an invalid entry reads row 0 with size 0."""
    valid = _kv_valid(g, seqlen_k)
    return (g if valid else Int32(0)), (Int32(16) if valid else Int32(0))


# ------------------------------------------------------------------ tcgen05 ops on raw addresses
# The MMAs and the Q copies go to the NVVM dialect ops, as blackwell_helpers' tcgen05 fences do:
# cute has no atom for tcgen05.mma.ws, and the Cp128x256bOp atom wraps every copy in its own
# elect_one (one ELECT and branch per tcgen05.cp, 12 more registers at S = 1).
def _tmem_ptr(taddr, *, loc=None, ip=None):
    """TMEM address (lane in bits 16-31, column in 0-15) as an NVVM TMEM pointer (addrspace 6)."""
    addr = Int32(taddr).ir_value(loc=loc, ip=ip)
    return llvm.inttoptr(llvm.PointerType.get(6), addr, loc=loc, ip=ip)


def _desc(lo, hi: int, *, loc=None, ip=None):
    """64-bit shared-memory descriptor from its runtime low word and constant high word."""
    return (Int64(Uint32(lo)) | Int64(hi << 32)).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _mma_ws_ts(d_tmem, a_tmem, b_lo, b_hi: int, idesc: int, acc, *, loc=None, ip=None):
    """tcgen05.mma.ws.cta_group::1.kind::f16, A from TMEM: D (+)= A B."""
    nvvm.tcgen05_mma_ws(
        nvvm.Tcgen05MMAKind.F16,
        _tmem_ptr(d_tmem, loc=loc, ip=ip),
        _tmem_ptr(a_tmem, loc=loc, ip=ip),
        _desc(b_lo, b_hi, loc=loc, ip=ip),
        Int32(idesc).ir_value(loc=loc, ip=ip),
        (Int32(acc) != 0).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mma_ws_ss(d_tmem, a_lo, a_hi: int, b_lo, b_hi: int, idesc: int, acc, *, loc=None, ip=None):
    """tcgen05.mma.ws.cta_group::1.kind::f16, A and B from smem: D (+)= A B."""
    nvvm.tcgen05_mma_ws(
        nvvm.Tcgen05MMAKind.F16,
        _tmem_ptr(d_tmem, loc=loc, ip=ip),
        _desc(a_lo, a_hi, loc=loc, ip=ip),
        _desc(b_lo, b_hi, loc=loc, ip=ip),
        Int32(idesc).ir_value(loc=loc, ip=ip),
        (Int32(acc) != 0).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _utccp(taddr, s_lo, s_hi: int, *, loc=None, ip=None):
    """tcgen05.cp.cta_group::1.128x256b: 128 rows x 32 B from smem (descriptor) to TMEM."""
    nvvm.tcgen05_cp(
        nvvm.Tcgen05CpShape.SHAPE_128x256b,
        _tmem_ptr(taddr, loc=loc, ip=ip),
        _desc(s_lo, s_hi, loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


# TMEM <-> registers: the 32x32b.x32 atoms on one thread's 32 columns, one tcgen05.ld / st each
# (make_tmem_copy over a whole accumulator split the store into 32 single-column stores).
def _tmem32(taddr):
    """32 fp32 columns of TMEM from taddr (lane in bits 16-31, column in 0-15)."""
    return cute.make_tensor(
        cute.make_ptr(Float32, taddr, cute.AddressSpace.tmem), cute.make_layout(32)
    )


@cute.jit
def _tld32(taddr) -> list:
    atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition.x32), Float32)
    dst = cute.make_rmem_tensor((32,), Float32)
    cute.copy(atom, _tmem32(taddr), dst)
    v = dst.load()
    return [v[i] for i in range(32)]


@cute.jit
def _tst32(taddr, vals) -> None:
    atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition.x32), Float32)
    src = cute.make_rmem_tensor((32,), Float32)
    for i in cutlass.range_constexpr(32):
        src[i] = vals[i]
    cute.copy(atom, src, _tmem32(taddr))


@cute.jit
def _cp_async16(dst_smem, src_gmem, src_size) -> None:
    """cp.async.cg 16 B; src_size 0 zero-fills the 16 destination bytes and reads nothing."""
    cute.arch.cp_async_shared_global(
        _ptr(dst_smem, Uint32, cute.AddressSpace.smem, 16),
        _ptr(src_gmem, Uint32, cute.AddressSpace.gmem, 16),
        16,
        "cg",
        cp_size=src_size,
    )


@cute.jit
def _cvt2(a, b, dtype) -> Int32:
    """a, b -> 16-bit dtype, packed (a in the low half): cvt.rn.{bf16x2,f16x2}.f32."""
    v = Vector.from_elements((Float32(a), Float32(b)), Float32).to(dtype)
    return v.bitcast(Int32).to_elements()[0]


@cute.jit
def _mask_bit(x, bits, i: cutlass.Constexpr) -> Float32:
    """bit i of bits set ? x : -inf"""
    return x if (bits & (1 << i)) != 0 else -Float32.inf


# ------------------------------------------------------------------ plain loads and stores
# The kernel addresses smem and gmem by integer byte address; these wrap cute.arch.load/store.
def _ptr(addr, dtype, space, align):
    return cute.make_ptr(dtype, addr, space, assumed_align=align).llvm_ptr


@cute.jit
def _sts128(addr, a, b, c, d) -> None:
    """16 B smem store of four 32-bit values of one type."""
    dt = type(a)
    vec = Vector.from_elements((a, b, c, d), dt).ir_value()
    cute.arch.store(_ptr(addr, dt, cute.AddressSpace.smem, 16), vec)


@cute.jit
def _stg128(addr, a, b, c, d) -> None:
    """16 B gmem store of four 32-bit values of one type."""
    dt = type(a)
    vec = Vector.from_elements((a, b, c, d), dt).ir_value()
    cute.arch.store(_ptr(addr, dt, cute.AddressSpace.gmem, 16), vec)


@cute.jit
def _lds128(addr, dt) -> list:
    """16 B smem load as four values of type dt."""
    v = cute.arch.load(
        _ptr(addr, dt, cute.AddressSpace.smem, 16), ir.VectorType.get([4], dt.mlir_type)
    )
    return list(Vector(v, dtype=dt).to_elements())


@cute.jit
def _lds32(addr, dt):
    return cute.arch.load(_ptr(addr, dt, cute.AddressSpace.smem, 4), dt)


@cute.jit
def _sts32(addr, v) -> None:
    cute.arch.store(_ptr(addr, type(v), cute.AddressSpace.smem, 4), v)


@cute.jit
def _sts8(addr, v) -> None:
    cute.arch.store(_ptr(addr, cutlass.Uint8, cute.AddressSpace.smem, 1), cutlass.Uint8(v))


@cute.jit
def _stg64(addr, a, b) -> None:
    """8 B gmem store of two 32-bit values of one type."""
    dt = type(a)
    vec = Vector.from_elements((a, b), dt).ir_value()
    cute.arch.store(_ptr(addr, dt, cute.AddressSpace.gmem, 8), vec)


@cute.jit
def _dsmem_base(smem_addr, rank) -> Int32:
    """shared::cluster address of smem_addr in the CTA of cluster rank `rank` (mapa)."""
    p = cute.make_ptr(Float32, smem_addr, cute.AddressSpace.smem, assumed_align=16)
    return cute.arch.map_dsmem_ptr(p, rank).toint()


@cute.jit
def _ldd128(addr, dt) -> list:
    """16 B load from a shared::cluster address as four values of type dt."""
    # the 32-bit shared::cluster address travels in a shared pointer; ss selects .shared::cluster
    v = cute.arch.load(
        _ptr(addr, dt, cute.AddressSpace.smem, 16),
        ir.VectorType.get([4], dt.mlir_type),
        ss="cluster",
    )
    return list(Vector(v, dtype=dt).to_elements())


@cute.jit
def _ldd32(addr, dt):
    return cute.arch.load(_ptr(addr, dt, cute.AddressSpace.smem, 4), dt, ss="cluster")


class FlashAttentionMLADecodeH64Sm100:
    """64 query heads per KV head, heads on M.  Same __call__ signature as
    FlashAttentionMLADecodeSm100, so the interface compile and call sites are shared."""

    def __init__(self, topk_length: int = 2048, num_splits: int = 1):
        self.qhead_per_kvhead = 64
        self.hdimv = 512
        self.tile_n = 64  # KV rows per block
        assert topk_length % self.tile_n == 0, f"topk_length={topk_length} must be a multiple of 64"
        self.n_blocks_full = topk_length // self.tile_n
        # the valid range is split evenly at run time, so S need not divide the block count
        assert 1 <= num_splits <= max(1, self.n_blocks_full // 2), (
            f"num_splits={num_splits}: at most {self.n_blocks_full // 2} (2 blocks per split)"
        )
        self.num_splits = num_splits
        self.is_split_kv = num_splits > 1
        # 2-4 splits: a cluster of S CTAs per token combines in the kernel (final O and LSE, no
        # partials); the O staging and partial LSE are the split ones
        self.cluster_combine = self.combines_in_kernel(num_splits)
        # heads per combine step and thread (peer loads in flight: combine_unroll * S)
        self.combine_unroll = 3

        self.num_threads = 384
        self.num_softmax_threads = 128
        self.num_pair_threads = 64  # softmax warps w and w ^ 2
        self.num_lat_threads = 128
        self.num_rope_threads = 64
        self.num_mask_lanes = 8  # 8 indices each: one 64-bit mask per block
        self.lat_warp_lo = 4
        self.mma_warp_id = 8
        self.mask_warp_id = 9
        self.rope_warp_lo = 10

        # smem plan
        self.buffer_align_bytes = 1024
        self.chunk_bytes = 64 * 128  # 64 rows x 128 B
        self.lat_slot_bytes = 8 * self.chunk_bytes  # 512 latent dims
        self.rope_half_bytes = 64 * 64  # 64 rows x 64 B
        self.p_bytes = 64 * 128  # 64 head rows x 64 keys x 16 bit
        # partial-S exchange in two rounds of 16 fp32 per softmax thread (8 KiB, which pays for the
        # second rope slot)
        self.xbuf_floats = self.num_softmax_threads * 16

        # TMEM columns
        self.tmem_alloc_cols = 512
        self.tmem_off_O = 0
        self.tmem_off_Q = 256
        self.tmem_off_Qr = 384
        self.tmem_off_S = 400

        # named barriers (0 is the CTA barrier): softmax warp pairs (0, 2) and (1, 3), the
        # softmax warpgroup (epilogue staging)
        self.bar_id_pair = 1
        self.bar_id_softmax = 3

        # mbarriers in smem order: (name, arrivals, depth).  A ring has one barrier per slot.
        self.mbarriers = (
            ("q_lat_ready", self.num_lat_threads, 1),
            ("q_rope_ready", self.num_rope_threads, 1),
            ("utccp_lat_done", 1, 1),
            ("rope_ready", self.num_rope_threads, NUM_ROPE_SLOTS),
            ("qk_rope_done", 1, NUM_ROPE_SLOTS),
            ("s_free", self.num_softmax_threads, 1),
            ("p_ready", self.num_softmax_threads, 1),
            # latent chunks 0-3 / 4-7 of a slot: QK on the first half starts while the second lands
            ("kv_ready", self.num_lat_threads, NUM_SLOTS),
            ("kv_ready_hi", self.num_lat_threads, NUM_SLOTS),
            ("qk_done", 1, NUM_SLOTS),
            ("sv_done", 1, NUM_SLOTS),
            ("mask_ready", self.num_mask_lanes, NUM_SLOTS),
            ("mask_free", self.num_softmax_threads, NUM_SLOTS),
        )
        self.mbar_offset = {}
        n = 0
        for name, _, depth in self.mbarriers:
            self.mbar_offset[name] = n
            n += depth
        self.num_mbarriers = n

    @staticmethod
    def combines_in_kernel(num_splits: int) -> bool:
        """True when num_splits splits combine in the kernel over a CTA cluster: the call passes
        the final O and LSE (or None) and runs no flash_fwd_combine.  Otherwise (S > 1) the
        kernel writes fp32 partials for flash_fwd_combine."""
        return 2 <= num_splits <= 4

    def launch_dims(self, total_q, num_kv_heads):
        """(grid, cluster or None) of the launch.  One CTA per (token, KV split), split index
        fastest (split_idx = blockIdx.x % S).  In-kernel combine: clusters of S CTAs along x, so
        cluster rank == blockIdx.x % S == split index, which the combine relies on."""
        grid = (total_q * self.num_splits, num_kv_heads, 1)
        return grid, ((self.num_splits, 1, 1) if self.cluster_combine else None)

    def _get_shared_storage_cls(self):
        align = self.buffer_align_bytes
        lat_words = NUM_SLOTS * self.lat_slot_bytes // 4
        rope_words = NUM_ROPE_SLOTS * 2 * self.rope_half_bytes // 4
        p_words = self.p_bytes // 4
        xbuf_floats = self.xbuf_floats
        row_floats = self.num_softmax_threads
        num_mbarriers = self.num_mbarriers

        @cute.struct
        class SharedStorage:
            lat: cute.struct.Align[cute.struct.MemRange[Uint32, lat_words], align]
            rope: cute.struct.Align[cute.struct.MemRange[Uint32, rope_words], align]
            p: cute.struct.Align[cute.struct.MemRange[Uint32, p_words], align]
            xbuf: cute.struct.Align[cute.struct.MemRange[Float32, xbuf_floats], 128]
            rmax: cute.struct.Align[cute.struct.MemRange[Float32, row_floats], 128]
            libuf: cute.struct.Align[cute.struct.MemRange[Float32, row_floats], 128]
            masks: cute.struct.Align[cute.struct.MemRange[Int64, NUM_SLOTS], 8]
            mbar: cute.struct.MemRange[Int64, num_mbarriers]
            tmem_holding: cute.struct.MemRange[Int32, 1]

        return SharedStorage

    # ------------------------------------------------------------------ host entry
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, h, 64)
        mQv: cute.Tensor,  # (total_q, h, 512)
        mK: cute.Tensor,  # (total_k, h_k, 64)
        mV: cute.Tensor,  # (total_k, h_k, 512)
        mO: cute.Tensor,  # (total_q, h, 512), or (S, total_q, h, 512) fp32 when split
        mLSE: Optional[cute.Tensor],  # (total_q, h), (S, h, total_q) when split, None if unused
        softmax_scale: Float32,
        mCuSeqlensQ: cute.Tensor,
        mIndexTopk: cute.Tensor,  # (total_q, topk)
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mTopkValidLen: Optional[cute.Tensor] = None,  # (total_q,)
        stream: cuda.CUstream = None,
    ):
        dtype = mV.element_type
        assert dtype in (cutlass.Float16, cutlass.BFloat16), "sparse decode requires 16-bit KV"
        if const_expr(self.cluster_combine):
            assert mO.element_type == dtype, "the in-kernel combine writes O in the KV dtype"
        elif const_expr(self.is_split_kv):
            assert mO.element_type == Float32, "split partials are fp32"
            assert mLSE is not None, (
                "split-KV needs the partial LSE: flash_fwd_combine reduces on it"
            )
        else:
            assert mO.element_type == dtype, "the unsplit 64-head decode writes O in the KV dtype"
        self.dtype = dtype
        SharedStorage = self._get_shared_storage_cls()
        smem_bytes = SharedStorage.size_in_bytes()
        assert smem_bytes <= SMEM_CAP_BYTES, f"{smem_bytes} B exceeds the {SMEM_CAP_BYTES} B cap"

        grid_dim, cluster_dim = self.launch_dims(cute.size(mQv.shape[0]), mV.shape[1])
        cluster = {"cluster": cluster_dim} if const_expr(cluster_dim is not None) else {}
        self.kernel(
            mQ,
            mQv,
            mK,
            mV,
            mO,
            mLSE,
            mIndexTopk,
            mTopkValidLen,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedK,
            softmax_scale * LOG2_E,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=(self.num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
            **cluster,
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
        softmax_scale_log2: Float32,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, head_kv, _ = cute.arch.block_idx()

        # x = (token, split), split fastest
        m_idx = bidx
        split_idx = Int32(0)
        if const_expr(self.is_split_kv):
            m_idx = bidx // self.num_splits
            split_idx = bidx % self.num_splits
        head_base = head_kv * self.qhead_per_kvhead
        # the batch lookup (for the KV range) runs in the gather roles, after their Q staging
        seq = (mCuSeqlensQ, mCuSeqlensK, mSeqUsedK)

        if const_expr(mTopkValidLen is None):
            n_valid = Int32(self.n_blocks_full)
        else:
            n_valid = max(
                Int32(0),
                min(cute.ceil_div(mTopkValidLen[m_idx], self.tile_n), Int32(self.n_blocks_full)),
            )
        # this CTA's blocks [blk_lo, blk_lo + num_blocks): the valid range split evenly
        blk_lo = Int32(0)
        num_blocks = n_valid
        s_ne = None
        if const_expr(self.is_split_kv):
            per = cute.ceil_div(n_valid, self.num_splits)
            if const_expr(self.cluster_combine):
                # per >= 1, so the non-empty split count is 0 exactly when n_valid is
                per = max(per, Int32(MIN_BLK_PER_SPLIT))
                s_ne = cute.ceil_div(n_valid, per)
            blk_lo = split_idx * per
            num_blocks = max(Int32(0), min(n_valid - blk_lo, per))

        if num_blocks > 0:
            self.main_body(
                mQ,
                mQv,
                mK,
                mV,
                mO,
                mLSE,
                mIndexTopk,
                softmax_scale_log2,
                SharedStorage,
                tidx,
                warp_idx,
                m_idx,
                head_kv,
                head_base,
                seq,
                split_idx,
                blk_lo,
                num_blocks,
                s_ne=s_ne,
            )
        else:
            if const_expr(self.cluster_combine):
                # an empty token: split 0 writes the final O = 0 / LSE = -inf; trailing empty
                # splits of a non-empty token just exit
                if n_valid == 0 and split_idx == 0:
                    self.empty_body(mO, mLSE, tidx, m_idx, split_idx, head_base)
            else:
                self.empty_body(mO, mLSE, tidx, m_idx, split_idx, head_base)

    # ------------------------------------------------------------ empty CTA (no block)
    @cute.jit
    def empty_body(self, mO, mLSE, tidx, m_idx, split_idx, head_base):
        # no smem, no TMEM, no barriers
        h = self.qhead_per_kvhead
        if const_expr(self.is_split_kv and not self.cluster_combine):
            # LSE partial = -inf only: flash_fwd_combine never reads the O partial of such a split
            if tidx < h:
                mLSE[split_idx, head_base + tidx, m_idx] = -Float32.inf
        else:
            # O = 0 for the 64 x 512 tile, LSE = -inf
            o_base = mO.iterator.toint()
            so0 = Int64(mO.stride[0])
            so1 = Int64(mO.stride[1])
            z = Int32(0)
            units_per_row = self.hdimv * 2 // 16  # 16 B units per 16-bit head row
            for i in cutlass.range(tidx, h * units_per_row, self.num_threads, unroll=1):
                hh = i // units_per_row
                u = i % units_per_row
                addr = (
                    o_base + (Int64(m_idx) * so0 + Int64(head_base + hh) * so1) * 2 + Int64(u * 16)
                )
                _stg128(addr, z, z, z, z)
            if const_expr(mLSE is not None):
                if tidx < h:
                    mLSE[m_idx, head_base + tidx] = -Float32.inf

    # ------------------------------------------------------------ full CTA
    @cute.jit
    def main_body(
        self,
        mQ,
        mQv,
        mK,
        mV,
        mO,
        mLSE,
        mIndexTopk,
        softmax_scale_log2,
        SharedStorage,
        tidx,
        warp_idx,
        m_idx,
        head_kv,
        head_base,
        seq,
        split_idx,
        blk_lo,
        num_blocks,
        s_ne=None,
    ):
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_base = storage.mbar.data_ptr()
        # a ring's pointer is its slot-0 barrier
        bar = {name: mbar_base + off for name, off in self.mbar_offset.items()}

        lat_base = storage.lat.data_ptr().toint()
        rope_base = storage.rope.data_ptr().toint()
        p_base = storage.p.data_ptr().toint()
        # Q staging needs no barrier: issue it first, arrive on q_*_ready after the CTA barrier
        # (cp.async.mbarrier.arrive tracks every earlier cp.async of the thread)
        # the KV-range lookup of the gather roles overlaps the barrier setup, too
        offset_k = Int32(0)
        seqlen_k = Int32(0)
        if warp_idx >= self.lat_warp_lo and warp_idx < self.mma_warp_id:
            self.stage_q_latent(mQv, tidx - self.lat_warp_lo * 32, m_idx, head_base, lat_base)
            offset_k, seqlen_k = self._kv_range(m_idx, seq)
        elif warp_idx >= self.mask_warp_id:
            if warp_idx >= self.rope_warp_lo:
                self.stage_q_rope(mQ, tidx - self.rope_warp_lo * 32, m_idx, head_base, p_base)
            offset_k, seqlen_k = self._kv_range(m_idx, seq)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for name, arrivals, depth in self.mbarriers:
                    for i in cutlass.range_constexpr(depth):
                        cute.arch.mbarrier_init(bar[name] + i, arrivals)
            cute.arch.mbarrier_init_fence()
        if warp_idx == self.mma_warp_id:
            cute.arch.alloc_tmem(self.tmem_alloc_cols, storage.tmem_holding.data_ptr())
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        tbase = storage.tmem_holding.get_tensor(cute.make_layout(1))[0]

        xbuf_base = storage.xbuf.data_ptr().toint()
        rmax_base = storage.rmax.data_ptr().toint()
        li_base = storage.libuf.data_ptr().toint()
        mask_base = storage.masks.data_ptr().toint()

        if warp_idx < self.lat_warp_lo:
            self.softmax_loop(
                mO,
                mLSE,
                softmax_scale_log2,
                tidx,
                warp_idx,
                m_idx,
                split_idx,
                head_base,
                num_blocks,
                tbase,
                p_base,
                xbuf_base,
                rmax_base,
                li_base,
                mask_base,
                lat_base,
                bar["qk_done"],
                bar["mask_ready"],
                bar["mask_free"],
                bar["s_free"],
                bar["sv_done"],
                bar["p_ready"],
                s_ne=s_ne,
            )
        elif warp_idx < self.mma_warp_id:
            self.latent_producer(
                mV,
                mIndexTopk,
                tidx - self.lat_warp_lo * 32,
                m_idx,
                head_kv,
                offset_k,
                seqlen_k,
                blk_lo,
                num_blocks,
                lat_base,
                bar["q_lat_ready"],
                bar["kv_ready"],
                bar["kv_ready_hi"],
                bar["sv_done"],
                bar["utccp_lat_done"],
            )
        elif warp_idx == self.mma_warp_id:
            self.mma_loop(
                num_blocks,
                tbase,
                lat_base,
                rope_base,
                p_base,
                bar["q_lat_ready"],
                bar["q_rope_ready"],
                bar["utccp_lat_done"],
                bar["rope_ready"],
                bar["qk_rope_done"],
                bar["s_free"],
                bar["p_ready"],
                bar["kv_ready"],
                bar["kv_ready_hi"],
                bar["qk_done"],
                bar["sv_done"],
            )
        elif warp_idx == self.mask_warp_id:
            self.mask_loop(
                mIndexTopk,
                m_idx,
                seqlen_k,
                blk_lo,
                num_blocks,
                mask_base,
                bar["mask_ready"],
                bar["mask_free"],
            )
        else:
            self.rope_producer(
                mK,
                mIndexTopk,
                tidx - self.rope_warp_lo * 32,
                m_idx,
                head_kv,
                offset_k,
                seqlen_k,
                blk_lo,
                num_blocks,
                rope_base,
                bar["q_rope_ready"],
                bar["rope_ready"],
                bar["qk_rope_done"],
            )

        # every role is done: the softmax waited the last PV, the MMA warp waited its last commits
        cute.arch.barrier()
        if warp_idx == self.mma_warp_id:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding.data_ptr())
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
        if const_expr(self.cluster_combine):
            if s_ne > 1:
                self.cluster_combine_epilogue(
                    mO,
                    mLSE,
                    tidx,
                    m_idx,
                    split_idx,
                    head_base,
                    s_ne,
                    lat_base,
                    rmax_base,
                    xbuf_base,
                )

    # ================================================================ producers
    @cute.jit
    def _kv_range(self, m_idx, seq):
        """(offset_k, seqlen_k) of token m_idx's request.  Decode has one token per request, so try
        batch = m_idx first (one round trip) before the binary search over cu_seqlens_q."""
        mCuSeqlensQ, mCuSeqlensK, mSeqUsedK = seq
        # clamped candidate (every load stays inside its tensor); its KV range loads alongside
        # the check, so the common case is one round trip
        cand = min(m_idx, cute.size(mCuSeqlensQ) - 2)
        q_lo = mCuSeqlensQ[cand]
        q_hi = mCuSeqlensQ[cand + 1]
        offset_k, seqlen_k = self._kv_range_of(cand, mCuSeqlensK, mSeqUsedK)
        if q_lo > m_idx or q_hi <= m_idx:
            offset_k, seqlen_k = self._kv_range_of(
                get_batch_from_cu_tensor(m_idx, mCuSeqlensQ), mCuSeqlensK, mSeqUsedK
            )
        return offset_k, seqlen_k

    @cute.jit
    def _kv_range_of(self, batch_idx, mCuSeqlensK, mSeqUsedK):
        seqlen = SeqlenInfoQK.create(
            batch_idx,
            Int32(1),
            Int32(1),
            mCuSeqlensQ=None,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=None,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_n,
            tile_n=self.tile_n,
        )
        return seqlen.offset_k, seqlen.seqlen_k

    @cute.jit
    def _load_idx(self, mIndexTopk, m_idx, blk, r0, step: cutlass.Constexpr, n: cutlass.Constexpr):
        """Top-k entries r0 + step * j (j < n) of block blk."""
        base = blk * self.tile_n + r0
        vals = []
        for j in cutlass.range_constexpr(n):
            vals.append(mIndexTopk[m_idx, base + step * j])
        return tuple(vals)

    @cute.jit
    def stage_q_latent(self, mQv, t, m_idx, head_base, lat_base):
        """Q latent -> latent slot 1 (SW128 K-major, heads on rows), gather's thread map."""
        r0 = t // 8
        u = t % 8
        q_base = mQv.iterator.toint()
        sq0 = Int64(mQv.stride[0])
        sq1 = Int64(mQv.stride[1])
        q_tok = q_base + (Int64(m_idx) * sq0 + Int64(head_base) * sq1) * 2 + Int64(u * 16)
        dst0 = lat_base + self.lat_slot_bytes + _sw128(r0, u)
        for j in cutlass.range_constexpr(4):
            # rows r0 + 16 j keep r0's swizzle
            src = q_tok + Int64(r0 + 16 * j) * sq1 * 2
            dst = dst0 + 16 * j * 128
            for c in cutlass.range_constexpr(8):
                _cp_async16(dst + c * self.chunk_bytes, src + c * 128, Int32(16))

    @cute.jit
    def stage_q_rope(self, mQ, t, m_idx, head_base, stage_base):
        """Q rope -> the P buffer (SW64, 8 KiB like the rope slot), same thread map as the rope
        gather.  The rope slot stays free for block 0; the softmax writes P(0) only after
        qk_done(0), whose commit also tracks the Q rope UTCCP."""
        r0 = t // 8
        unit = t % 8
        kh = unit // 4
        uu = unit % 4
        q_base = mQ.iterator.toint()
        sq0 = Int64(mQ.stride[0])
        sq1 = Int64(mQ.stride[1])
        q_tok = q_base + (Int64(m_idx) * sq0 + Int64(head_base) * sq1) * 2 + Int64(unit * 16)
        for j in cutlass.range_constexpr(8):
            r = r0 + 8 * j
            dst = stage_base + kh * self.rope_half_bytes + _sw64(r, uu)
            _cp_async16(dst, q_tok + Int64(r) * sq1 * 2, Int32(16))

    @cute.jit
    def latent_producer(
        self,
        mV,
        mIndexTopk,
        t,
        m_idx,
        head_kv,
        offset_k,
        seqlen_k,
        blk_lo,
        num_blocks,
        lat_base,
        q_lat_ready,
        kv_ready,
        kv_ready_hi,
        sv_done,
        utccp_lat_done,
    ):
        # thread t owns rows t // 8 + 16 j (j < 4) and 16 B unit t % 8 of all 8 chunks
        r0 = t // 8
        u = t % 8
        row0 = _sw128(r0, u)  # rows r0 + 16 j keep r0's swizzle

        cute.arch.cp_async_mbarrier_arrive_noinc(q_lat_ready)
        # later blocks' indices load under the previous block's gather
        g0, g1, g2, g3 = self._load_idx(mIndexTopk, m_idx, blk_lo, r0, 16, 4)

        # ---- latent gather
        v_base = mV.iterator.toint()
        sv0 = Int64(mV.stride[0])
        v_head = v_base + Int64(head_kv) * Int64(mV.stride[1]) * 2 + Int64(u * 16)
        for k in cutlass.range(num_blocks, unroll=1):
            slot, ph = _ring(k)
            gi = [g0, g1, g2, g3]
            if k == 1:
                # slot 1 held Q latent until its UTCCP completed
                cute.arch.mbarrier_wait(utccp_lat_done, 0)
            cute.arch.mbarrier_wait(sv_done + slot, ph ^ 1)
            slot_base = lat_base + slot * self.lat_slot_bytes + row0
            rows = []
            for j in cutlass.range_constexpr(4):
                rows.append(_gather_row(gi[j], seqlen_k))
            for half in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(4):
                    row, size = rows[j]
                    src = v_head + Int64(offset_k + row) * sv0 * 2
                    dst = slot_base + 16 * j * 128
                    for c in cutlass.range_constexpr(4 * half, 4 * half + 4):
                        _cp_async16(dst + c * self.chunk_bytes, src + c * 128, size)
                # tracks every earlier cp.async of the thread: the first arrive completes with
                # chunks 0-3, the second with all 8
                cute.arch.cp_async_mbarrier_arrive_noinc(
                    (kv_ready if half == 0 else kv_ready_hi) + slot
                )
            # next block's indices (the last iteration reloads its own: always in bounds)
            g0, g1, g2, g3 = self._load_idx(
                mIndexTopk, m_idx, blk_lo + min(k + 1, num_blocks - 1), r0, 16, 4
            )

    @cute.jit
    def rope_producer(
        self,
        mK,
        mIndexTopk,
        t,
        m_idx,
        head_kv,
        offset_k,
        seqlen_k,
        blk_lo,
        num_blocks,
        rope_base,
        q_rope_ready,
        rope_ready,
        qk_rope_done,
    ):
        # thread t owns rows t // 8 + 8 j (j < 8) and 16 B unit t % 8 of the 128 B rope row:
        # half kh = unit // 4, SW64 unit ((unit % 4) ^ ((r >> 1) % 4)) of that half
        r0 = t // 8
        unit = t % 8
        kh = unit // 4
        uu = unit % 4

        cute.arch.cp_async_mbarrier_arrive_noinc(q_rope_ready)
        g0, g1, g2, g3, g4, g5, g6, g7 = self._load_idx(mIndexTopk, m_idx, blk_lo, r0, 8, 8)

        k_base = mK.iterator.toint()
        sk0 = Int64(mK.stride[0])
        k_head = k_base + Int64(head_kv) * Int64(mK.stride[1]) * 2 + Int64(unit * 16)
        for k in cutlass.range(num_blocks, unroll=1):
            gi = [g0, g1, g2, g3, g4, g5, g6, g7]
            rs, rph = _ring(k, NUM_ROPE_SLOTS)
            # rope slot rs is free once QK rope(k-2) completed (fresh barriers pass)
            cute.arch.mbarrier_wait(qk_rope_done + rs, rph ^ 1)
            for j in cutlass.range_constexpr(8):
                row, size = _gather_row(gi[j], seqlen_k)
                dst = (
                    rope_base
                    + rs * 2 * self.rope_half_bytes
                    + kh * self.rope_half_bytes
                    + _sw64(r0 + 8 * j, uu)
                )
                _cp_async16(dst, k_head + Int64(offset_k + row) * sk0 * 2, size)
            cute.arch.cp_async_mbarrier_arrive_noinc(rope_ready + rs)
            g0, g1, g2, g3, g4, g5, g6, g7 = self._load_idx(
                mIndexTopk, m_idx, blk_lo + min(k + 1, num_blocks - 1), r0, 8, 8
            )

    @cute.jit
    def mask_loop(
        self, mIndexTopk, m_idx, seqlen_k, blk_lo, num_blocks, mask_base, mask_ready, mask_free
    ):
        lane = cute.arch.lane_idx()
        if lane < self.num_mask_lanes:
            for k in cutlass.range(num_blocks, unroll=1):
                slot, ph = _ring(k)
                bits = Int32(0)
                for b in cutlass.range_constexpr(8):
                    g = mIndexTopk[m_idx, (blk_lo + k) * self.tile_n + lane * 8 + b]
                    if _kv_valid(g, seqlen_k):
                        bits = bits | (1 << b)
                cute.arch.mbarrier_wait(mask_free + slot, ph ^ 1)
                _sts8(mask_base + slot * 8 + lane, bits)
                cute.arch.mbarrier_arrive(mask_ready + slot)

    # ================================================================== MMA warp
    @cute.jit
    def mma_loop(
        self,
        num_blocks,
        tbase,
        lat_base,
        rope_base,
        p_base,
        q_lat_ready,
        q_rope_ready,
        utccp_lat_done,
        rope_ready,
        qk_rope_done,
        s_free,
        p_ready,
        kv_ready,
        kv_ready_hi,
        qk_done,
        sv_done,
    ):
        id_qk = sm100_desc.make_instr_desc(
            self.dtype, self.dtype, Float32, 64, 128, Major.K, Major.K
        )
        id_pv = sm100_desc.make_instr_desc(
            self.dtype, self.dtype, Float32, 64, 256, Major.K, Major.MN
        )
        # K-major SW128: 64 rows x 128 B (latent chunk, Q latent, P); K-major SW64: 64 rows x
        # 64 B (rope half); V MN-major SW128: 256 dims (4 chunks of 64 dims, 8 KiB apart) x 16 rows
        K128, HI128 = _desc_base((64, 8), (8, 1), 3, Major.K)
        K64, HI64 = _desc_base((64, 4), (4, 1), 2, Major.K)
        V128, v_hi = _desc_base(((8, 4), 16), ((1, self.chunk_bytes // 16), 8), 3, Major.MN)
        assert v_hi == HI128  # so HI128 serves as the high word of both PV operands
        rope_lo = _desc_lo(rope_base, K64)
        p_lo = _desc_lo(p_base, K128)
        t_S = tbase + self.tmem_off_S

        # ---- Q -> TMEM by UTCCP (tcgen05.cp -> tcgen05.mma is a tcgen05 pipeline, same thread)
        # Q rope is staged in the P buffer
        cute.arch.mbarrier_wait(q_rope_ready, 0)
        cute.arch.fence_view_async_shared()
        fa_sm100_utils.fence_tcgen05_after_thread_sync()
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(2):
                _utccp(tbase + (self.tmem_off_Qr + 8 * s), p_lo + (s * 32 >> 4), HI64)
        cute.arch.mbarrier_wait(q_lat_ready, 0)
        cute.arch.fence_view_async_shared()
        fa_sm100_utils.fence_tcgen05_after_thread_sync()
        q_lo = _desc_lo(lat_base + self.lat_slot_bytes, K128)
        with cute.arch.elect_one():
            for p in cutlass.range_constexpr(4):
                for s in cutlass.range_constexpr(4):
                    _utccp(
                        tbase + (self.tmem_off_Q + 32 * p + 8 * s),
                        q_lo + ((p * 2 * self.chunk_bytes + s * 32) >> 4),
                        HI128,
                    )
            cute.nvgpu.tcgen05.commit(utccp_lat_done)

        for k in cutlass.range(num_blocks + 1, unroll=1):
            if k < num_blocks:
                slot, ph = _ring(k)
                # S is free once softmax(k-1) loaded it
                cute.arch.mbarrier_wait(s_free, (k & 1) ^ 1)
                rs, rph = _ring(k, NUM_ROPE_SLOTS)
                cute.arch.mbarrier_wait(rope_ready + rs, rph)
                cute.arch.fence_view_async_shared()
                fa_sm100_utils.fence_tcgen05_after_thread_sync()
                with cute.arch.elect_one():
                    for s in cutlass.range_constexpr(2):
                        _mma_ws_ts(
                            t_S,
                            tbase + (self.tmem_off_Qr + 8 * s),
                            rope_lo + ((rs * 2 * self.rope_half_bytes + s * 32) >> 4),
                            HI64,
                            id_qk,
                            Int32(0 if s == 0 else 1),
                        )
                    cute.nvgpu.tcgen05.commit(qk_rope_done + rs)
                slot_lo = _desc_lo(lat_base + slot * self.lat_slot_bytes, K128)
                for half in cutlass.range_constexpr(2):
                    # chunk pairs 2 half, 2 half + 1: latent chunks 4 half .. 4 half + 3
                    cute.arch.mbarrier_wait((kv_ready if half == 0 else kv_ready_hi) + slot, ph)
                    cute.arch.fence_view_async_shared()
                    fa_sm100_utils.fence_tcgen05_after_thread_sync()
                    with cute.arch.elect_one():
                        for j in cutlass.range_constexpr(8 * half, 8 * half + 8):
                            p = j // 4
                            kk = j % 4
                            _mma_ws_ts(
                                t_S,
                                tbase + (self.tmem_off_Q + 8 * j),
                                slot_lo + ((2 * p * self.chunk_bytes + kk * 32) >> 4),
                                HI128,
                                id_qk,
                                Int32(1),
                            )
                with cute.arch.elect_one():
                    cute.nvgpu.tcgen05.commit(qk_done + slot)
            if k > 0:
                j = k - 1
                sj, _ = _ring(j)
                # P(j) stored and O rescaled by softmax(j)
                cute.arch.mbarrier_wait(p_ready, j & 1)
                fa_sm100_utils.fence_tcgen05_after_thread_sync()
                v_lo = _desc_lo(lat_base + sj * self.lat_slot_bytes, V128)
                acc0 = Int32(1)
                if j == 0:
                    acc0 = Int32(0)
                with cute.arch.elect_one():
                    for hh in cutlass.range_constexpr(2):
                        for kk in cutlass.range_constexpr(4):
                            _mma_ws_ss(
                                tbase + (self.tmem_off_O + 128 * hh),
                                p_lo + (kk * 32 >> 4),
                                HI128,
                                v_lo + ((hh * 4 * self.chunk_bytes + kk * 16 * 128) >> 4),
                                HI128,
                                id_pv,
                                acc0 if kk == 0 else Int32(1),
                            )
                    cute.nvgpu.tcgen05.commit(sv_done + sj)

        # every commit must have landed before the CTA can exit (with one block, nobody else waits
        # utccp_lat_done or the last qk_rope_done)
        last = num_blocks - 1
        slot, ph = _ring(last)
        cute.arch.mbarrier_wait(sv_done + slot, ph)
        for b in cutlass.range_constexpr(NUM_ROPE_SLOTS):
            # the last qk_rope_done of each rope slot
            if last - b >= 0:
                rs, rph = _ring(last - b, NUM_ROPE_SLOTS)
                cute.arch.mbarrier_wait(qk_rope_done + rs, rph)
        cute.arch.mbarrier_wait(utccp_lat_done, 0)

    # ================================================================ softmax warps
    @cute.jit
    def softmax_loop(
        self,
        mO,
        mLSE,
        softmax_scale_log2,
        tidx,
        warp_idx,
        m_idx,
        split_idx,
        head_base,
        num_blocks,
        tbase,
        p_base,
        xbuf_base,
        rmax_base,
        li_base,
        mask_base,
        lat_base,
        qk_done,
        mask_ready,
        mask_free,
        s_free,
        sv_done,
        p_ready,
        s_ne=None,
    ):
        lane = cute.arch.lane_idx()
        row = lane + 32 * (warp_idx & 1)  # head row, == tidx % 64
        hf = warp_idx >> 1  # key-column half: columns 32 hf .. 32 hf + 31
        tl = tbase + ((32 * warp_idx) << 16)
        bar_pair = self.bar_id_pair + (warp_idx & 1)
        # thread t pairs with t ^ 64 (warp w ^ 2): same head row, other key-column half
        partner = tidx ^ (self.num_softmax_threads // 2)

        # partial-S exchange: warp w's buffer holds 32 lanes x 16 fp32, lane-interleaved in 16 B
        # units (unit i of every lane, then unit i + 1), so each 16 B STS/LDS is conflict-free
        x_unit = 16
        x_row = 32 * x_unit  # one 16 B unit of every lane
        x_warp = self.xbuf_floats * 4 // (self.num_softmax_threads // 32)
        x_wr = xbuf_base + (warp_idx ^ 2) * x_warp + lane * x_unit
        x_rd = xbuf_base + warp_idx * x_warp + lane * x_unit

        mi = Float32(_MAX_INIT)
        li = Float32(0.0)
        scale = softmax_scale_log2

        for k in cutlass.range(num_blocks, unroll=1):
            slot, ph = _ring(k)
            cute.arch.mbarrier_wait(qk_done + slot, ph)
            cute.arch.mbarrier_wait(mask_ready + slot, ph)
            fa_sm100_utils.fence_tcgen05_after_thread_sync()
            own = _tld32(tl + (self.tmem_off_S + 32 * hf))
            peer = _tld32(tl + (self.tmem_off_S + 32 * (1 - hf)))
            cute.arch.fence_view_async_tmem_load()
            fa_sm100_utils.fence_tcgen05_before_thread_sync()
            cute.arch.mbarrier_arrive(s_free)
            mbits = _lds32(mask_base + slot * 8 + hf * 4, Int32)
            cute.arch.mbarrier_arrive(mask_free + slot)

            # mask before the add: -inf + a finite peer partial stays -inf
            for i in cutlass.range_constexpr(32):
                own[i] = _mask_bit(own[i], mbits, i)
            # partial-S exchange with thread t ^ 64 (warp w ^ 2): same head row, other lane half.
            # Two rounds of 16 fp32.  Round 1 writes the peer's slot (x_wr) and reads the own one
            # (x_rd); round 2 writes the own slot, which this thread has just read, and reads the
            # peer's, so each round needs one pair barrier.
            for rnd in cutlass.range_constexpr(2):
                wr = x_wr if rnd == 0 else x_rd
                rd = x_rd if rnd == 0 else x_wr
                for i in cutlass.range_constexpr(4):
                    c = 16 * rnd + 4 * i
                    _sts128(wr + i * x_row, peer[c], peer[c + 1], peer[c + 2], peer[c + 3])
                cute.arch.barrier(barrier_id=bar_pair, number_of_threads=self.num_pair_threads)
                for i in cutlass.range_constexpr(4):
                    c = 16 * rnd + 4 * i
                    v = _lds128(rd + i * x_row, Float32)
                    for e in cutlass.range_constexpr(4):
                        own[c + e] = own[c + e] + v[e]

            # row max over the 64 keys: own 32, then the partner's through smem
            m0 = own[0]
            m1 = own[16]
            for i in cutlass.range_constexpr(1, 16):
                m0 = cute.arch.fmax(m0, own[i])
                m1 = cute.arch.fmax(m1, own[16 + i])
            cur = cute.arch.fmax(m0, m1) * scale
            _sts32(rmax_base + tidx * 4, cur)
            cute.arch.barrier(barrier_id=bar_pair, number_of_threads=self.num_pair_threads)
            cur = cute.arch.fmax(cur, _lds32(rmax_base + partner * 4, Float32))

            # lazy rescale, warp-uniform (the partner warp holds the same rows and decides the same)
            should_scale = cute.arch.vote_any_sync(cur - mi > _RESCALE_THRESHOLD)
            new_max = mi
            scale_old = Float32(1.0)
            if should_scale:
                new_max = cute.arch.fmax(cur, mi)
                scale_old = cute.math.exp2(mi - new_max, approx=True, ftz=True)
            mi = new_max

            ssum0 = Float32(0.0)
            ssum1 = Float32(0.0)
            pk = []
            for i in cutlass.range_constexpr(16):
                # ex2.approx.ftz (fastmath's exp2 costs registers); a masked key is -inf and gives 0
                e0 = cute.math.exp2(own[2 * i] * scale - new_max, approx=True, ftz=True)
                e1 = cute.math.exp2(own[2 * i + 1] * scale - new_max, approx=True, ftz=True)
                ssum0 = ssum0 + e0
                ssum1 = ssum1 + e1
                pk.append(_cvt2(e0, e1, self.dtype))
            li = li * scale_old + (ssum0 + ssum1)

            if k > 0:
                # PV(k-1) done: P is free and O holds every block before k
                sp, php = _ring(k - 1)
                cute.arch.mbarrier_wait(sv_done + sp, php)
            for i in cutlass.range_constexpr(4):
                _sts128(
                    p_base + _sw128(row, 4 * hf + i),
                    pk[4 * i],
                    pk[4 * i + 1],
                    pk[4 * i + 2],
                    pk[4 * i + 3],
                )
            if k > 0:
                if should_scale:
                    fa_sm100_utils.fence_tcgen05_after_thread_sync()
                    for c in cutlass.range_constexpr(8):
                        o = _tld32(tl + (self.tmem_off_O + 32 * c))
                        cute.arch.fence_view_async_tmem_load()
                        for i in cutlass.range_constexpr(32):
                            o[i] = o[i] * scale_old
                        _tst32(tl + (self.tmem_off_O + 32 * c), o)
                    cute.arch.fence_view_async_tmem_store()
                    fa_sm100_utils.fence_tcgen05_before_thread_sync()
            # P is read by the MMA through the async proxy
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(p_ready)

        # ---- epilogue --------------------------------------------------------------
        # a row whose keys are all masked sums ex2(-inf) = 0; li is NaN only from NaN inputs, which
        # the transposed kernel also writes as an empty row (LSE = -inf)
        _sts32(li_base + tidx * 4, li)
        cute.arch.barrier(barrier_id=bar_pair, number_of_threads=self.num_pair_threads)
        li = li + _lds32(li_base + partner * 4, Float32)
        empty = li == Float32(0.0) or li != li
        args = (
            mO,
            mLSE,
            tidx,
            m_idx,
            split_idx,
            head_base,
            hf,
            row,
            tl,
            li,
            mi,
            empty,
            num_blocks,
            sv_done,
            lat_base,
            rmax_base,
        )
        if const_expr(self.cluster_combine):
            if s_ne == 1:
                # the one non-empty split holds the whole token: the unsplit epilogue (final O
                # and LSE), no stage and no cluster barrier
                self.epilogue_out(*args, split=False, peers=False)
            else:
                self.epilogue_out(*args, split=True, peers=True)
        else:
            self.epilogue_out(*args, split=self.is_split_kv, peers=False)
        # the tcgen05 loads precede the CTA barrier that gates tcgen05.dealloc
        fa_sm100_utils.fence_tcgen05_before_thread_sync()

    @cute.jit
    def epilogue_out(
        self,
        mO,
        mLSE,
        tidx,
        m_idx,
        split_idx,
        head_base,
        hf,
        row,
        tl,
        li,
        mi,
        empty,
        num_blocks,
        sv_done,
        lat_base,
        rmax_base,
        split: cutlass.Constexpr,
        peers: cutlass.Constexpr,
    ):
        """LSE and O / li of the softmax warpgroup.  split: fp32 O staging (else 16-bit final O);
        peers: leave the stage and the partial LSE (rmax[head]) in smem for the cluster combine,
        else store them (the partial tensors when split, the final O and LSE otherwise)."""
        if const_expr(peers):
            # the partial LSE goes to the row-max buffer (free now: the partner read its last
            # value before the li exchange) for the cluster peers
            if hf == 0:
                lse = -Float32.inf
                if not empty:
                    lse = (mi + cute.math.log2(li, fastmath=True)) * LN2
                _sts32(rmax_base + row * 4, lse)
        elif const_expr(mLSE is not None):
            if hf == 0:
                lse = -Float32.inf
                if not empty:
                    lse = (mi + cute.math.log2(li, fastmath=True)) * LN2
                if const_expr(split):
                    mLSE[split_idx, head_base + row, m_idx] = lse
                else:
                    mLSE[m_idx, head_base + row] = lse
        inv = Float32(0.0)
        if not empty:
            inv = cute.arch.rcp_approx(li)

        slot, ph = _ring(num_blocks - 1)
        cute.arch.mbarrier_wait(sv_done + slot, ph)
        fa_sm100_utils.fence_tcgen05_after_thread_sync()
        # O / li -> smem (every latent slot is free once the last PV completed): one row per head,
        # 16 B units XOR-swizzled by row % 8, then coalesced 16 B stores.  Unsplit: 16-bit in the
        # KV dtype, rows of 1024 B in slot 0.  Split: the fp32 partial, rows of 2048 B in slots 0-1.
        ebytes = 4 if split else 2
        row_bytes = self.hdimv * ebytes
        units_per_row = row_bytes // 16
        dims_per_unit = 16 // ebytes
        for hh in cutlass.range_constexpr(2):
            for c2 in cutlass.range_constexpr(2):
                # two loads per wait
                o = _tld32(tl + (self.tmem_off_O + 128 * hh + 64 * c2))
                o = o + _tld32(tl + (self.tmem_off_O + 128 * hh + 64 * c2 + 32))
                cute.arch.fence_view_async_tmem_load()
                w = []
                if const_expr(split):
                    # pairs: a 64-iteration static loop draws the DSL's slow-compile warning
                    for i in cutlass.range_constexpr(32):
                        w.append(o[2 * i] * inv)
                        w.append(o[2 * i + 1] * inv)
                else:
                    for i in cutlass.range_constexpr(32):
                        w.append(_cvt2(o[2 * i] * inv, o[2 * i + 1] * inv, self.dtype))
                # lane half hf of PV half hh holds dims 256 hh + 128 hf + [0, 128)
                u0 = (256 * hh + 128 * hf + 64 * c2) // dims_per_unit
                for q in cutlass.range_constexpr(len(w) // 4):
                    _sts128(
                        lat_base + _sw128(row, u0 + q, row_bytes),
                        w[4 * q],
                        w[4 * q + 1],
                        w[4 * q + 2],
                        w[4 * q + 3],
                    )
        if const_expr(not peers):
            self.store_o(mO, tidx, m_idx, split_idx, head_base, lat_base, split, units_per_row)

    @cute.jit
    def store_o(
        self,
        mO,
        tidx,
        m_idx,
        split_idx,
        head_base,
        lat_base,
        split: cutlass.Constexpr,
        units_per_row,
    ):
        """Staged O (latent slots) -> gmem: coalesced 16 B stores by the softmax warpgroup."""
        ebytes = 4 if split else 2
        row_bytes = units_per_row * 16
        cute.arch.barrier(
            barrier_id=self.bar_id_softmax, number_of_threads=self.num_softmax_threads
        )
        o_base = mO.iterator.toint()
        if const_expr(split):
            # (S, total_q, h, dv) fp32
            o_tok = (
                o_base
                + (Int64(split_idx) * Int64(mO.stride[0]) + Int64(m_idx) * Int64(mO.stride[1]))
                * ebytes
            )
            so_head = Int64(mO.stride[2])
        else:
            o_tok = o_base + Int64(m_idx) * Int64(mO.stride[0]) * ebytes
            so_head = Int64(mO.stride[1])
        o_tok = o_tok + Int64(head_base) * so_head * ebytes
        units = self.qhead_per_kvhead * units_per_row
        for i in cutlass.range_constexpr(units // self.num_softmax_threads):
            f = i * self.num_softmax_threads + tidx
            r = f // units_per_row
            u = f % units_per_row
            v = _lds128(lat_base + _sw128(r, u, row_bytes), Uint32)
            _stg128(o_tok + Int64(r) * so_head * ebytes + Int64(u * 16), v[0], v[1], v[2], v[3])

    # ================================================================ in-kernel combine
    @cute.jit
    def cluster_combine_epilogue(
        self, mO, mLSE, tidx, m_idx, split_idx, head_base, s_ne, lat_base, rmax_base, xbuf_base
    ):
        """All 384 threads of a non-empty split, S_ne > 1, after TMEM dealloc.  Every non-empty
        split of the token has staged O / li (fp32, SW128 rows of 2048 B in latent slots 0-1) and
        its partial LSE (rmax[head]).  This CTA (rank split_idx of S_ne) reduces heads
        [64 rank / S_ne, 64 (rank + 1) / S_ne) over DSMEM into the final O and LSE."""
        S = self.num_splits
        h = self.qhead_per_kvhead
        # a row empty in this split (partial LSE -inf) stages 0, not 0 * O (NaN when O is): its
        # combine weight is 0, and a NaN split then adds nothing, as flash_fwd_combine skips it.
        # Rare, so a branch: softmax thread t overwrites the units it staged (row t % 64, lane
        # half t // 64); the CTA barrier before the TMEM dealloc ordered its stores and the LSE
        if tidx < self.num_softmax_threads:
            half = self.num_softmax_threads // 2
            row = tidx % half
            if _lds32(rmax_base + row * 4, Float32) == -Float32.inf:
                z = Float32(0.0)
                row_bytes = self.hdimv * 4
                for hh in cutlass.range_constexpr(2):
                    # fp32 dims 256 hh + 128 hf + [0, 128): 16 B units (64 hh + 32 hf) + [0, 32)
                    for q in cutlass.range_constexpr(32):
                        u = 64 * hh + 32 * (tidx // half) + q
                        _sts128(lat_base + _sw128(row, u, row_bytes), z, z, z, z)
        # publish this CTA's stage and LSEs, see every peer's (release / acquire)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        h_lo = (h * split_idx) // s_ne
        nh = (h * (split_idx + 1)) // s_ne - h_lo

        # combine weights of this CTA's heads -> xbuf[hl][rank] (the exchange buffer is free)
        if tidx < nh:
            hh = h_lo + tidx
            lses = []
            m = -Float32.inf
            for r in cutlass.range_constexpr(S):
                v = -Float32.inf
                if r < s_ne:
                    v = _ldd32(_dsmem_base(rmax_base, Int32(r)) + hh * 4, Float32)
                lses.append(v)
                m = cute.arch.fmax(m, v)
            # a head empty in every split: sum 0, LSE -inf, weights 0 (so O = 0), as in
            # flash_fwd_combine; the finite stand-ins keep -inf - -inf out of the exponents
            live = m != -Float32.inf
            m_s = m if live else Float32(0.0)
            ssum = Float32(0.0)
            for r in cutlass.range_constexpr(S):
                ssum = ssum + cute.math.exp2((lses[r] - m_s) * LOG2_E, fastmath=True)
            lse = m_s + cute.math.log2(ssum, fastmath=True) * LN2
            lse = lse if live else -Float32.inf
            lse_s = lse if live else Float32(0.0)
            for r in cutlass.range_constexpr(S):
                w = cute.math.exp2((lses[r] - lse_s) * LOG2_E, fastmath=True)
                _sts32(xbuf_base + (tidx * S + r) * 4, w)
            if const_expr(mLSE is not None):
                mLSE[m_idx, head_base + hh] = lse
        cute.arch.barrier()

        # O, one instance per non-empty split count (the peers read are exactly the live ones)
        for sn in cutlass.range_constexpr(2, S + 1):
            if s_ne == sn:
                self.combine_heads(
                    mO, sn, tidx, m_idx, split_idx, head_base, h_lo, nh, lat_base, xbuf_base
                )
        # keep this CTA's smem alive until every peer has read its stage.  Relaxed is enough: this
        # barrier orders no writes (no peer reads anything written after the first barrier), and
        # each thread's DSMEM loads have returned before it arrives, since their values feed the
        # O stores above
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    @cute.jit
    def combine_heads(
        self,
        mO,
        sn: cutlass.Constexpr,
        tidx,
        m_idx,
        split_idx,
        head_base,
        h_lo,
        nh,
        lat_base,
        xbuf_base,
    ):
        """O of heads h_lo + [0, nh) over the sn non-empty splits: peer k reads rank
        (split_idx + k) % sn, k = 0 (this CTA) from local smem, the others over DSMEM.  Thread t
        owns 16 B fp32 unit u = t % 128 (4 dims) of heads t // 128 + 3 j, j < n_j; combine_unroll
        heads per step with every load issued first (peer loads predicated on j < n_j), then 8 B
        stores in the KV dtype."""
        S = self.num_splits
        row_bytes = self.hdimv * 4
        units_per_row = row_bytes // 16
        n_grp = self.num_threads // units_per_row
        u = tidx % units_per_row
        g0 = tidx // units_per_row
        ranks = [split_idx]
        bases = [lat_base]
        for k in cutlass.range_constexpr(1, sn):
            rk = split_idx + k
            rk = rk - sn if rk >= sn else rk
            ranks.append(rk)
            bases.append(_dsmem_base(lat_base, rk))
        so_head = Int64(mO.stride[1]) * 2
        o_col = (
            mO.iterator.toint()
            + (Int64(m_idx) * Int64(mO.stride[0]) + Int64(head_base + h_lo) * Int64(mO.stride[1]))
            * 2
            + Int64(u * 8)
        )
        n_j = cute.ceil_div(nh - g0, n_grp)
        unr = self.combine_unroll
        for jb in cutlass.range(0, n_j, unr, unroll=1):
            hls = []
            vals = []
            for jj in cutlass.range_constexpr(unr):
                # past n_j the head is clamped to the last one: its local load is harmless, its
                # peer loads are skipped (no DSMEM traffic) and so is its store
                hl = min(g0 + n_grp * (jb + jj), nh - 1)
                hls.append(hl)
                off = _sw128(h_lo + hl, u, row_bytes)
                vals.append(_lds128(lat_base + off, Float32))
                for k in cutlass.range_constexpr(1, sn):
                    if const_expr(jj == 0):  # jb < n_j
                        vals.append(_ldd128(bases[k] + off, Float32))
                    else:
                        v0, v1, v2, v3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                        if jb + jj < n_j:
                            v0, v1, v2, v3 = _ldd128(bases[k] + off, Float32)
                        vals.append([v0, v1, v2, v3])
            for jj in cutlass.range_constexpr(unr):
                acc = [Float32(0.0)] * 4
                for k in cutlass.range_constexpr(sn):
                    w = _lds32(xbuf_base + (hls[jj] * S + ranks[k]) * 4, Float32)
                    v = vals[jj * sn + k]
                    for e in cutlass.range_constexpr(4):
                        acc[e] = acc[e] + w * v[e]
                if jb + jj < n_j:
                    _stg64(
                        o_col + Int64(hls[jj]) * so_head,
                        _cvt2(acc[0], acc[1], self.dtype),
                        _cvt2(acc[2], acc[3], self.dtype),
                    )
