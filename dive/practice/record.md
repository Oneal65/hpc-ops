# CuTe Practice Record — hpc-ops

| Item | Value |
|---|---|
| Date of scan | 2026-10-26 |
| Repository | `/root/hpc-ops`, branch `main`, commit `2a2e265` |
| CuTe version | CUTLASS 4.4.2, headers only, in `3rd/cutlass/include` (commit `da5e086`) |
| Target GPU | 8 × NVIDIA H20 (sm90, compute capability 9.0) |
| Toolchain | nvcc 12.8.93, cmake 4.2.0, ninja, torch 2.8.0+cu128 |
| Writing style | Approximately 80 % ASD-STE100 (Simplified Technical English) |

---

## 1. Scope

### 1.1 Purpose

This document has three parts:

1. A map of the hpc-ops repository, with a focus on CuTe code.
2. A method to practice CuTe in this repository.
3. A roadmap, from layout algebra to full Hopper kernels.

### 1.2 "CuTe DSL": two meanings

The term "CuTe DSL" has two meanings. Know which one you study.

| Name | Language | Package | In this repository? |
|---|---|---|---|
| CuTe (C++) | C++ templates | `cute/*.hpp` in CUTLASS 3.x / 4.x | **Yes.** All kernels use it. |
| CuTe DSL (Python) | Python, JIT to PTX | `nvidia-cutlass-dsl`, `import cutlass.cute` | **No.** It is not vendored. |

This repository teaches CuTe C++. The concepts are the same in both
languages: Layout, Tensor, Atom, TiledCopy, TiledMMA, partition. If you
learn them here, you can move to the Python DSL quickly. Stage 10 of the
roadmap (section 6) gives an optional bridge to the Python DSL.

### 1.3 Terms

| Term | Meaning in this document |
|---|---|
| CTA | One thread block. |
| Warp | 32 threads. |
| Warpgroup (WG) | 4 consecutive warps = 128 threads. Hopper WGMMA runs on one WG. |
| gmem / smem / rmem / tmem | Global memory / shared memory / registers / tensor memory (sm100 only). |
| TMA | Tensor Memory Accelerator. A hardware copy engine for gmem ↔ smem (sm90+). |
| mbarrier | A shared-memory barrier object with a phase bit and a transaction count. |
| WGMMA | Warpgroup MMA instruction (sm90). The accumulator is in registers. |
| UMMA / tcgen05 | The sm100 MMA instruction. One thread issues it. The accumulator is in TMEM. |
| Atom | The smallest hardware operation that CuTe wraps (one MMA or one copy instruction). |
| Tiled\* | An atom that is repeated over threads and values to cover a tile. |
| Stage | One slot in a circular buffer of smem tiles. |

---

## 2. Repository structure

### 2.1 Top-level map

```text
hpc-ops/
├── hpc/                 Python wrappers. One file can expose several operators.
│   ├── __init__.py      Detects the GPU arch and loads _C_sm<arch>.abi3.so.
│   └── <op>.py          User API + torch.library.register_fake.
├── src/
│   ├── <op>/entry.cc    Validation, allocation, launch config, HPC_ARCH_DISPATCH.
│   ├── <op>/<op>.h      The *_async contract that all architectures share.
│   ├── <op>/sm90/       Hopper kernels (.cu / .cuh).
│   ├── <op>/sm100/      Blackwell kernels.
│   ├── utils/           Shared device helpers (utils.cuh, tma.cuh) + standalone CuTe tests.
│   ├── C/               Python extension glue.
│   └── communicator/    Host IPC. No kernels.
├── tests/test_<op>.py   pytest correctness tests.
├── benchmark/           Performance scripts.
├── 3rd/cutlass/include  Vendored CUTLASS/CuTe headers (no docs, no examples).
└── dive/                Your notes. CMake does not compile this folder.
```

### 2.2 Execution path of one operator

```text
 Python                 C++ host (every arch)                  Device (one arch)
┌──────────────┐  ┌──────────────────────────────────┐  ┌────────────────────────────┐
│ hpc/gemm.py  │─▶│ torch.ops.hpc.gemm_bf16xfp32     │  │                            │
│  (API, fake) │  │   │                              │  │                            │
└──────────────┘  │   ▼                              │  │                            │
                  │ src/gemm/entry.cc                │  │                            │
                  │   TORCH_CHECK(...)  validation   │  │                            │
                  │   torch::empty(...) allocation   │  │                            │
                  │   pick tile / splitK             │  │                            │
                  │   HPC_ARCH_DISPATCH(             │  │                            │
                  │     90, gemm_bf16xfp32_async(..))│─▶│ src/gemm/sm90/gemm.cu      │
                  └──────────────────────────────────┘  │  host: make_tma_copy,      │
                                                        │        tile_to_shape, ...  │
                                                        │  kernel<<<grid,block>>>    │
                                                        │  device: CuTe code         │
                                                        └────────────────────────────┘
```

CuTe code is in two places:

* **Host side of the `.cu` file.** It makes gmem tensors, smem layouts, TMA descriptors and the TiledMMA type. The kernel gets them as template parameters.
* **Device side (`__global__` and `.cuh`).** It partitions tensors and calls `cute::copy` and `cute::gemm`.

### 2.3 Operator catalog and CuTe density

"CuTe lines" is an approximate count of lines that use `cute::`, `make_tensor`,
`make_layout`, `local_tile` or `partition_`.

| Operator dir | Arch | Main file(s) | Lines | CuTe lines | CuTe pattern | Learning value |
|---|---|---|---|---|---|---|
| `gemm` | sm90 | `sm90/gemm.cu` | 557 | 43 | TMA + WGMMA, warp-specialized, persistent, split-K | ★★★★★ best first full kernel |
| `gemm` | sm100 | `sm100/gated_mla_gemm.cuh` | 519 | 37 | TMA + UMMA + TMEM, 2-SM cluster, CLC scheduler | ★★★★ (compile only on H20) |
| `group_gemm` | sm90 | `sm90/kernels.cuh`, `sm90/config.h` | 900 + 172 | 93 | TMA + WGMMA, FP8, per-group TMA descriptor update | ★★★★★ |
| `group_gemm` | sm90 | `sm90/cp_async/*` | ~1100 | 86 | cp.async multistage + WGMMA, predication | ★★★★★ classic pipeline |
| `attention/prefill` | sm90 | `sm90/kernels.cuh` | 3798 | 383 | Flash attention, two GEMMs, online softmax, paged KV | ★★★★★ advanced |
| `attention/decode` | sm90 | `sm90/util_kernels.cuh`, `static/`, `dynamic/` | ~5000 | ~300 | Flash decoding, split-K, swap-AB, dynamic schedule | ★★★★ |
| `stem` | sm90 | `sm90/stem_kernels.cuh` | 1246 | 28 | Warp-specialized GEMM (OAM GEMM) | ★★★ clean second GEMM |
| `fuse_moe` | sm90 | `sm90/count_and_gather*.cu` | ~1400 | 42 | TMA descriptor build only (no MMA) | ★★ |
| `utils` | all | `utils.cuh`, `tma.cuh`, `test_*.cu` | ~1100 | 60 | Helpers, standalone CuTe programs | ★★★★★ sandbox |
| `activation`, `rope`, `normalization`, `sampler`, `topk`, `iHC`, `allreduce` | shared | — | — | 0 | Plain CUDA (vec_t, warp shuffles) | ☆ not CuTe |

Result: the CuTe code is in **gemm, group_gemm, attention, stem**, plus
helpers in **utils**. The other operators are plain CUDA.

### 2.4 Build facts for a learner

| Fact | Detail |
|---|---|
| Source discovery | CMake globs `src/**/*.cu` and `src/**/*.cc` (`CMakeLists.txt:59`). You do not edit a source list. |
| Excluded sources | Any path that contains `test` is excluded (`CMakeLists.txt:61`). This is why `src/utils/test_*.cu` are not in the module. |
| Arch selection | A path without `sm<N>/` is shared. `sm<TARGET>/` is compiled. Other `sm<N>/` are not compiled. |
| Flags | `sm_90a`, C++20, `-Werror=all-warnings`, `-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED`, `-Wl,--no-undefined`. |
| Build sm90 only | `SM_ARCH=90 make` (the default on this machine). |
| One test | `python3 -m pytest -v tests/test_<op>.py`. Build first, because the tests import from `build/lib.*`. |
| Standalone CuTe | `src/utils/Makefile` builds each `.cu` with plain `nvcc -arch=sm_90a -I../../ -I../../3rd/cutlass/include`. |

### 2.5 Environment status (updated 2026-10-26, see `stage0.md`)

| Item | Status | Action |
|---|---|---|
| GPU | H20 × 8, sm90 | Use for sm90 labs. sm100 code compiles, but cannot run. |
| `build/` | **Built.** `build/sm90/_C_sm90.abi3.so`, 0 warnings, 2 min 13 s | Rebuild with `SM_ARCH=90 make`. |
| GEMM correctness | **Pass.** 288/288 cases via `labs/stage0_gemm_smoke.py` | Use pytest when it is installed. |
| `pytest`, `black`, `cpplint` | Not installed | Run `pip install -r requirements-dev.txt`. |
| Old `hpc` in site-packages | `hpc-ops 1.1.0`, old layout | Import from `build/lib.*` (the test files do this). |
| Standalone nvcc build | **Tested. It works.** `labs/` builds 6 labs in 10 s. | Use this method for the labs. |

---

## 3. CuTe concept map

### 3.1 Concept layers

```text
 Layer 5  Algorithms       cute::copy   cute::gemm   copy_if   clear   axpby
             ▲
 Layer 4  Partitioning     local_tile  local_partition  thr.partition_S/D/A/B/C
             ▲             partition_fragment_C  make_fragment_A/B  retile_S/D
             ▲
 Layer 3  Tiled ops        TiledCopy (make_tiled_copy, make_tiled_copy_C, make_tma_copy)
             ▲             TiledMMA  (make_tiled_mma)
             ▲
 Layer 2  Atoms            Copy_Atom<SM80_CP_ASYNC.., SM90_TMA_LOAD, SM90_U32x4_STSM_N, ..>
             ▲             MMA atom  <SM90_64x64x16_F32BF16BF16_SS, SM100_MMA_F16BF16_SS, ..>
             ▲
 Layer 1  Tensor           Tensor = Engine (pointer: gmem/smem/rmem/tmem) + Layout
             ▲
 Layer 0  Layout algebra   Shape, Stride, Layout, IntTuple, Int<N>
                           composition, complement, logical_divide, zipped_divide,
                           logical_product, raked_product, coalesce, left/right_inverse,
                           Swizzle, ComposedLayout, tile_to_shape
```

Learn the layers from the bottom up. Each layer uses only the layers below it.

### 3.2 Where each concept appears in this repository

| Concept | Representative location |
|---|---|
| `make_layout`, `Int<N>`, static/dynamic strides | `src/gemm/sm90/gemm.cu:422-429` |
| `tile_to_shape` + GMMA swizzle atom | `src/gemm/sm90/gemm.cu:431-437`, `src/group_gemm/sm90/config.h:14-41` |
| `cosize` (smem size) | `src/gemm/sm90/gemm.cu:439-440` |
| `take<0,2>` (drop the stage mode) | `src/gemm/sm90/gemm.cu:443-446` |
| `make_tma_copy` (load and store) | `src/gemm/sm90/gemm.cu:443-446` |
| `get_tma_tensor`, `get_slice`, `partition_S/D` | `src/gemm/sm90/gemm.cu:120-138` |
| `tma.with(barrier)` | `src/gemm/sm90/gemm.cu:194` |
| TMA descriptor update in the kernel | `src/utils/tma.cuh`, `src/utils/test_tma.cu:20-39` |
| mbarrier ring (`initialize_barrier`, `wait_barrier`, `arrive_barrier`, `set_barrier_transaction_bytes`) | `src/gemm/sm90/gemm.cu:106-220` |
| `warpgroup_reg_alloc/dealloc` | `src/gemm/sm90/gemm.cu:166, 226` |
| `make_tiled_mma`, `partition_A/B`, `make_fragment_A/B`, `partition_fragment_C` | `src/gemm/sm90/gemm.cu:233-243, 452` |
| WGMMA sequence (`warpgroup_arrive`, `gemm`, `commit_batch`, `wait`) | `src/gemm/sm90/gemm.cu:264-300` |
| `make_tiled_copy_C` + STSM + `retile_S` | `src/gemm/sm90/gemm.cu:322-338` |
| TMA store (`tma_store_fence/arrive/wait`) | `src/gemm/sm90/gemm.cu:335-364` |
| `Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL>`, `make_tiled_copy(atom, thr, val)` | `src/group_gemm/sm90/cp_async/config.h:61-75` |
| `local_tile` | `src/group_gemm/sm90/cp_async/group_gemm_fp8.cu:101`, `src/utils/test_trans_0189.cu:57` |
| `make_identity_tensor` + `copy_if` (predication) | `src/group_gemm/sm90/cp_async/group_gemm_fp8.cu:136-157` |
| `retile_fragment` (accumulator as a (row, col) view) | `src/utils/utils.cuh:433` |
| `left_inverse(C).compose(A)` (reuse the accumulator as an A operand) | `src/attention/prefill/sm90/kernels.cuh:411` |
| Identity tensor for masks | `src/attention/prefill/sm90/kernels.cuh:413` |
| Custom `make_tiled_copy` with `raked_product` + `right_inverse` | `src/utils/utils.cuh:705-717` |
| TMEM allocation, `make_tmem_copy`, UMMA | `src/gemm/sm100/gated_mla_gemm.cuh:184, 282, 362-371` |

---

## 4. Key concepts, with diagrams

Each subsection explains one concept that a learner must know before the
related roadmap stage. The examples use output from a test program that was
compiled and run on this machine (section 5.3).

### 4.1 A Layout is a function

A layout maps a logical coordinate to a linear index. It has two parts:
a **Shape** and a **Stride**. The notation is `Shape:Stride`.

```text
 row_major = (4,8):(8,1)          index = row*8 + col*1

        col→  0    1    2    3    4    5    6    7
 row 0   |    0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
 row 1   |    8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
 row 2   |   16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
 row 3   |   24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |

 col_major = (4,8):(1,4)          index = row*1 + col*4   (CuTe default, "LayoutLeft")
```

* `_8` means a static value (`Int<8>{}`). The compiler knows it.
* `8` means a dynamic value (a run-time `int`).
* Use static values when you can. The compiler can then unroll loops and remove index math.

### 4.2 Hierarchical layouts

A mode can itself be a layout. This lets one layout describe a "tile of tiles".

```text
 h = ((2,2),8):((1,16),2)

 mode 0 has shape (2,2) and stride (1,16)   → 4 rows, but in two levels
 mode 1 has shape 8     and stride 2

 h(3,1):  row coord 3 → (3%2, 3/2) = (1,1) → 1*1 + 1*16 = 17
          col coord 1 →                          1*2  =  2
          index = 19          (the program printed: h(3,1) = 19)
```

All WGMMA fragments and TMA partitions in this repository are hierarchical
layouts of this type. Practice how to read them until it is easy.

### 4.3 Tensor = pointer + layout

```text
             Tensor
   ┌────────────┴────────────┐
 Engine (iterator)          Layout
 make_gmem_ptr(p)  → gmem   (M,K):(K,1)
 make_smem_ptr(p)  → smem   Swizzle o (8,64):(64,1)
 register array    → rmem   ((2,2,8),1,1):((1,2,4),0,0)
 make_tmem_ptr(a)  → tmem   (sm100 only)
```

`local_tile` cuts one tile from a big tensor:

```text
 A = (32,64):(64,1)            local_tile(A, (8,16), coord(1,2))
 ┌────┬────┬────┬────┐
 │0,0 │0,1 │0,2 │0,3 │         result: (8,16):(64,1)
 ├────┼────┼────┼────┤         first element = A(8, 32) = 8*64 + 32 = 544
 │1,0 │1,1 │1,2█│1,3 │  ◀── tile (1,2)
 ├────┼────┼────┼────┤         (the program printed: first=544)
 │2,0 │2,1 │2,2 │2,3 │
 ├────┼────┼────┼────┤
 │3,0 │3,1 │3,2 │3,3 │
 └────┴────┴────┴────┘
```

If you use `make_coord(_, _)`, you keep all tiles. The result is then a
rank-4 tensor `(8,16,4,4)`. `src/utils/test_trans_0189.cu:57` shows this.

### 4.4 Swizzle (bank-conflict removal)

All smem tiles in the TMA + WGMMA kernels use a GMMA swizzle atom:

```text
 GMMA::Layout_K_SW128_Atom<bf16>  prints as  Sw<3,4,3> o smem_ptr[16b] o (_8,_64):(_64,_1)
```

`Sw<B,M,S>` = `Swizzle<3,4,3>`. CuTe applies it to the **byte address**
(`3rd/cutlass/include/cute/pointer_swizzle.hpp:87`):

* Bits [4,7) select the 16-byte chunk in a 128-byte row.
* Bits [7,10) are the row number mod 8.
* new_chunk = chunk XOR (row mod 8).

```text
            16-byte chunk position in the 128-byte row
 row   c0  c1  c2  c3  c4  c5  c6  c7      ← logical chunk (8 bf16 each)
  0     0   1   2   3   4   5   6   7      ← physical chunk
  1     1   0   3   2   5   4   7   6
  2     2   3   0   1   6   7   4   5
  3     3   2   1   0   7   6   5   4
  4     4   5   6   7   0   1   2   3
  5     5   4   7   6   1   0   3   2
  6     6   7   4   5   2   3   0   1
  7     7   6   5   4   3   2   1   0

 A column of 8 logical chunks (same cN, rows 0..7) now goes to 8 different
 physical chunks → 8 different bank groups → no bank conflict.
```

Rules:

1. The smem base address must be 1024-byte aligned. The swizzle pattern repeats every 8 × 128 B.
2. The TMA box and the WGMMA descriptor must use the **same** swizzle. `make_tma_copy(..., take<0,2>(slayout))` and `make_fragment_A(partition_A(sX))` both read it from the same layout, so they agree.
3. `tile_to_shape(atom, (M,K,Stage))` repeats the atom to fill the tile. Example from the test program:

```text
 tile_to_shape(SW128<bf16>, (64,64,3))
 = Sw<3,4,3> o ((_8,_8),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_4096))
 cosize = 12288 elements = 24 KiB
```

### 4.5 TMA copy

TMA moves a whole box between gmem and smem. One thread issues it.

```text
 HOST (gemm.cu:443)                               DEVICE (gemm.cu:120-196)
 ┌───────────────────────────────────┐            ┌────────────────────────────────────────┐
 │ X  = make_tensor(gmem_ptr,(m,k))  │            │ gX  = tma_x.get_tma_tensor((m,k))      │
 │ box= take<0,2>(slayout_x)         │            │       (a coordinate tensor, no pointer)│
 │ tma_x = make_tma_copy(            │  __grid_   │ tXg = tma_x.get_slice(0).partition_S(gX)│
 │   SM90_TMA_LOAD{}, X, box)        │  constant__│       (TMA, TILE_M, TILE_K)            │
 │   └─ builds CUtensorMap (128 B)   │ ─────────▶ │ tXs = ...partition_D(sX)               │
 └───────────────────────────────────┘            │       (TMA, 1, 1, kStage)              │
                                                  │                                        │
                                                  │ one thread:                            │
                                                  │  copy(tma_x.with(readable[s]),         │
                                                  │       tXg(_, itile_m, itile_k),        │
                                                  │       tXs(_, 0, 0, s))                 │
                                                  │  set_barrier_transaction_bytes(        │
                                                  │       readable[s], bytes_of_box)       │
                                                  └────────────────────────────────────────┘
                     gmem ══TMA engine══▶ smem stage s
                                              │ when all bytes arrive
                                              ▼
                                     mbarrier readable[s] flips phase
```

Points to remember:

* `get_tma_tensor` returns coordinates, not addresses. The TMA engine does the address math.
* The transaction byte count must equal the **full** box size. TMA counts the zero-filled out-of-bounds part too.
* To change the base address or shape of a tensor at run time (variable-length batches, MoE groups), copy the descriptor to smem, patch it, and publish it. See `src/utils/tma.cuh` and `src/utils/test_tma.cu`.

### 4.6 mbarrier pipeline (producer / consumer ring)

All Hopper kernels here use raw mbarriers, not `cutlass::Pipeline*` classes.
Each stage has two barriers: `readable[s]` and `writable[s]`.

```text
 Producer (1 thread)                     smem ring (kStage = 3)          Consumers (math WGs)
 phase = 1  (start: all slots free)      ┌────┬────┬────┐                phase = 0
                                         │ s0 │ s1 │ s2 │
 loop k:                                 └────┴────┴────┘                loop k:
   wait(writable[s], phase) ───────────────────────────────────────────▶   (slot s is free)
   TMA copy → s, arrive+expect_tx(readable[s]) ─────▶ data lands ──▶       wait(readable[s], phase)
                                                                            WGMMA on slot s
                                                       ◀──────────────     arrive(writable[s])
   s++ ; if s == kStage { s = 0; phase ^= 1 }                              s++ ; if s == kStage { s = 0; phase ^= 1 }

 Time ─────────────────────────────────────────────────────────────────────────────▶
 Producer: [load s0][load s1][load s2][wait w0][load s0'][wait w1][load s1'] ...
 Consumer:          [ mma s0 ][ mma s1 ][ mma s2 ][ mma s0' ][ mma s1' ] ...
                    ▲ overlap: loads for k+1, k+2 run during MMA of k
```

The phase bit lets a barrier be reused. A wait on phase `p` returns when
the barrier has completed phase `p`. The producer starts with `phase = 1`,
so its first `kStage` waits return at once (`gemm.cu:175`).

### 4.7 Warp specialization

```text
 CTA = 384 threads  (__launch_bounds__(384,1))
 ┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
 │ WG0  threads   0..127    │ WG1  threads 128..255    │ WG2  threads 256..383    │
 │ MATH (consumer)          │ MATH (consumer)          │ LOAD (producer)          │
 │ warpgroup_reg_alloc<168> │ warpgroup_reg_alloc<168> │ warpgroup_reg_dealloc<24>│
 │ WGMMA + epilogue         │ WGMMA + epilogue         │ 1 elected thread → TMA   │
 └──────────────────────────┴──────────────────────────┴──────────────────────────┘
  Register budget: 2×128×168 + 128×24 = 46080 ≤ 65536 per SM
```

* `setmaxnreg` (wrapped by `warpgroup_reg_alloc/dealloc`) moves registers from the producer to the consumers.
* `src/gemm/sm90/gemm.cu` uses `kWarpGroupN` math WGs (1 or 2) + 1 load WG.
* `group_gemm`, `attention/prefill` warp-spec kernels and the `stem` OAM GEMM use 2 + 1.

### 4.8 WGMMA and the accumulator layout

```text
 TiledMMA = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<K,K>{})   → 128 threads (1 WG)

  A (64×16) in smem ──descriptor──┐
                                  ├──▶ WGMMA ──▶ C (64×64) fp32 in registers of 128 threads
  B (64×16) in smem ──descriptor──┘

  "SS" = A and B both from smem.     "RS" = A from registers, B from smem.
  make_fragment_A/B(...)  → GMMA descriptors, NOT register copies.
  partition_fragment_C(.) → real registers.
```

The program printed the accumulator layout of thread 0:

```text
 tCr = ((_2,_2,_8),_1,_1):((_1,_2,_4),_0,_0)       32 fp32 values per thread
          │  │  └─ 8 column blocks of width 8
          │  └──── 2 row halves (+0 / +8)
          └─────── 2 adjacent columns

 For warp w (0..3), lane l (0..31) in the WG:
   row = 16*w + l/4 + 8*i          i ∈ {0,1}
   col = 8*b + 2*(l%4) + j         j ∈ {0,1}, b ∈ {0..7}
   register index = j + 2*i + 4*b

        cols 0 1 | 2 3 | 4 5 | 6 7 | 8 9 ...
 row  0   l0     | l1  | l2  | l3  | l0  ...     (warp 0)
 row  1   l4     | l5  | l6  | l7  | l4  ...
 ...
 row  7   l28    | l29 | l30 | l31 |
 row  8   l0 (i=1) ...
 ...
 row 16   warp 1 starts
```

Consequence: the 4 lanes `l, l^1, l^2, l^3` own the same row. A row
reduction (softmax max or sum) is two `shfl_xor` steps (1, 2). See
`warp_4lane_reduce_max_xor` in `src/utils/utils.cuh:398`.

The WGMMA call sequence (`gemm.cu:264-281`):

```text
 tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;   // first k-block overwrites C
 warpgroup_fence_operand(tCr);                   // keep registers in order
 warpgroup_arrive();                             // wgmma.fence
 for ik: cute::gemm(tiled_mma, tA(_,_,ik,s), tB(_,_,ik,s), tCr);
         tiled_mma.accumulate_ = GMMA::ScaleOut::One;
 warpgroup_commit_batch();                       // wgmma.commit_group
 warpgroup_wait<0>();                            // wgmma.wait_group 0
 warpgroup_fence_operand(tCr);
```

### 4.9 Swap-AB

WGMMA M is fixed at 64. For small token counts (decode, small-batch GEMM)
the repository swaps the operands:

```text
 Wanted:   Y[m,n] = X[m,k] · W[n,k]ᵀ        m is small (1..64), n is large

 Swap-AB:  Yᵀ[n,m] = W[n,k] · X[m,k]ᵀ
           MMA "A" = W  (n on the 64-row M slot)
           MMA "B" = X  (m on the N slot: N can be 8,16,...,256)
           Accumulator is (n, m)  →  rows = n, cols = tokens

 gemm.cu:236  tWs4r = thr_mma.partition_A(sW);   // A = weights
 gemm.cu:237  tXs4r = thr_mma.partition_B(sX);   // B = activations
 gemm.cu:429  Y viewed as (n, m, splitK) with stride (1, n, n*m)
```

Swap-AB is in `gemm/sm90`, `group_gemm`, `attention/decode` and
`gemm/sm100`. When you read those kernels, translate "M" and "N" with care.

### 4.10 Epilogue: registers → smem → gmem

```text
 tCr (fp32) ──convert──▶ tYrh (bf16)
                           │  make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T,bf16>, tiled_mma)
                           │  retile_S(tYrh) → partition_D(sY)
                           ▼
                 STSM (stmatrix): 1 instruction writes 8×8 tiles from 32 lanes
                           ▼
                        sY (swizzled smem)
                           │  tma_store_wait<0>()  (previous store has read sY)
                           │  syncwarpgroup        (all STSM done)
                           │  tma_store_fence()    (make smem visible to the async proxy)
                           ▼
                 one thread: copy(tma_y, tYs, tYg); tma_store_arrive()
                           ▼
                         gmem Y
```

### 4.11 Flash attention in CuTe (prefill)

```text
 For one Q tile (128 rows = 2 math WGs × 64), loop over KV tiles j:

  sQ ──┐
       ├─ GEMM-1 (SS): S = Q·Kᵀ  ──▶ tAttr  fp32 accumulator (64 × 64 per WG)
  sK ──┘                                │
                                        │ retile_fragment → (rows, cols) view
                                        │ identity tensor → (row, col) per element → mask
                                        │ online softmax: m_new = max(m, rowmax)
                                        │                 P = exp2(S·scale - m_new)
                                        │                 O *= exp2(m - m_new)
                                        ▼
                                  tAttA = make_tensor(tAttr.data(),
                                          left_inverse(layout_asC).compose(layout_asA))
                                        │   same registers, new A-operand view (no data move)
                                        │   convert fp32 → bf16
  sV ──────────────────────────────┐    ▼
                                   ├─ GEMM-2 (RS): O += P·V   (A = P from registers)
                                   ┘
 After the last j: O /= rowsum → bf16 → STSM → sY → TMA store
```

Why the `left_inverse(...).compose(...)` trick works:

```text
 layout_asC : (thread, value) → (row, col)   how the accumulator owns P
 layout_asA : (thread, value) → (row, col)   how an RS A-operand must own P

 left_inverse(asC) : (row, col) → (thread, value)
 left_inverse(asC) ∘ asA : A-slot → C-slot in the SAME thread

 For bf16 the thread part is the identity, so this is only a register re-index.
 For fp8 the A-operand layout is different, so the kernel needs byte_perm
 shuffles and a transposed V (smem_trans_and_interleave0189_*, utils.cuh:461-532).
```

### 4.12 Blackwell (sm100) differences

```text
 sm90 (Hopper)                                sm100 (Blackwell)  gemm/sm100/gated_mla_gemm.cuh
 ───────────────────────────────              ────────────────────────────────────────────────
 128 threads issue WGMMA together            1 thread issues UMMA (tcgen05.mma)
 accumulator in registers                    accumulator in TMEM (tensor memory)
 epilogue reads registers                    epilogue: make_tmem_copy(SM100_TMEM_LOAD_..) → registers
 static persistent grid                      Cluster Launch Control (CLC) work stealing
 TMA multicast for clusters                  2-SM MMA (SM100_MMA_F16BF16_2x1SM_SS) + SM100_TMA_2SM_LOAD

 Warp roles in the sm100 kernel (256 threads):
 ┌────────┬────────┬────────┬────────┬───────────────────────────────┐
 │ warp 0 │ warp 1 │ warp 2 │ warp 3 │ warps 4..7 (1 WG)             │
 │ CLC    │ TMA    │ UMMA   │ idle   │ epilogue: TMEM→reg, sigmoid,  │
 │ sched  │ load   │ + TMEM │        │ ×x2, STSM, TMA store          │
 └────────┴────────┴────────┴────────┴───────────────────────────────┘
```

The H20 is sm90. You can compile this kernel with `SM_ARCH=100 make`, but
you cannot run it on this machine.

---

## 5. Practice method

### 5.1 The practice loop

Do each lab in this loop. Do not skip the "print" step. Most CuTe errors
are layout errors, and a print shows them at once.

```text
        ┌────────────────────────────────────────────────────────────┐
        ▼                                                            │
  1. READ ──▶ 2. PRINT ──▶ 3. PREDICT ──▶ 4. CHANGE ──▶ 5. BUILD ──▶ 6. VERIFY ──▶ 7. MEASURE ──▶ 8. RECORD
  repo code   print(layout)  write the     one small     nvcc or    compare with   ncu / nsys /   notes in
  + docs      print_tensor   expected      change        make       torch / CPU    torch timer    dive/practice
              on host or     result
              thread 0       first
```

Rules:

* Make one change at a time.
* Write the expected result before you run the program. Then compare.
* Keep each lab as a small file that you can run alone.
* Use `static_assert` and `CUTE_STATIC_ASSERT_V` to check shapes at compile time.

### 5.2 Workspace

Put all practice code under `dive/practice/`. CMake globs only `src/`, so
this code does not go into the `hpc` module and cannot break the build.

```text
dive/practice/
├── record.md            this document
├── stage0.md            environment record (2026-10-26)
├── stage1.md            layout algebra lab record (2026-10-26)
├── logs/                build and run logs
└── labs/                Makefile, check.hpp and one .cu per lab
```

Recommended `dive/practice/labs/Makefile`:

```make
ROOT  := ../../..
NVCC  := nvcc
FLAGS := -O2 -std=c++17 --expt-relaxed-constexpr -lineinfo \
         -arch=sm_90a -DHPC_TARGET_ARCH=90 \
         -I$(ROOT) -I$(ROOT)/3rd/cutlass/include -lcuda

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)
%: %.cu
	$(NVCC) -o $@ $< $(FLAGS)
clean:
	rm -f $(BINS)
```

* `-DHPC_TARGET_ARCH=90` is necessary if you include `src/utils/utils.cuh`.
* `-I$(ROOT)` lets you include repository helpers, for example `#include "src/utils/utils.cuh"`.
* One lab takes approximately 15–30 s to compile.

When a lab is a full operator (Stage 8), move it to `src/<op>/` and follow
the operator rules in `CLAUDE.md`.

### 5.3 Starter lab (tested on this machine)

This program compiled and ran on 2026-10-26. The outputs in section 4 come
from it.

```cpp
// dive/practice/labs/lab01_layout.cu
#include <cstdio>
#include "cute/tensor.hpp"
using namespace cute;

int main() {
  // 1. A layout is a function: coord -> index
  auto row_major = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
  auto col_major = make_layout(make_shape(Int<4>{}, Int<8>{}));  // default: LayoutLeft
  print("row_major: "); print(row_major); print("\n");
  print("col_major: "); print(col_major); print("\n");
  print_layout(row_major);

  // 2. Hierarchical layout
  auto h = make_layout(make_shape(make_shape(Int<2>{}, Int<2>{}), Int<8>{}),
                       make_stride(make_stride(Int<1>{}, Int<16>{}), Int<2>{}));
  print("hier: "); print(h); print("  h(3,1) = %d\n", int(h(3, 1)));

  // 3. Tiling a tensor
  float buf[32 * 64];
  for (int i = 0; i < 32 * 64; ++i) buf[i] = i;
  auto A = make_tensor(&buf[0], make_shape(32, 64), make_stride(64, Int<1>{}));
  auto tA = local_tile(A, make_tile(Int<8>{}, Int<16>{}), make_coord(1, 2));
  print("tile(1,2): "); print(tA.layout()); print("  first=%f\n", tA(0, 0));

  // 4. Swizzle used by GMMA K-major SW128 for bf16
  auto sw = GMMA::Layout_K_SW128_Atom<bfloat16_t>{};
  print("SW128 atom: "); print(sw); print("\n");
  auto sl = tile_to_shape(sw, make_shape(Int<64>{}, Int<64>{}, Int<3>{}));
  print("smem (64,64,3): "); print(sl); print("  cosize=%d\n", int(cosize(sl)));

  // 5. TiledMMA thread/value mapping
  auto mma = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{});
  print("tiled_mma size = %d threads\n", int(size(mma)));
  auto thr = mma.get_slice(0);
  auto gC = make_tensor(make_gmem_ptr((float*)nullptr), make_shape(Int<64>{}, Int<64>{}),
                        make_stride(Int<64>{}, Int<1>{}));
  auto tCr = thr.partition_fragment_C(gC);
  print("thread0 accumulator: "); print(tCr.layout()); print("\n");
  return 0;
}
```

Verified output (abbreviated):

```text
row_major: (_4,_8):(_8,_1)
col_major: (_4,_8):(_1,_4)
hier: ((_2,_2),_8):((_1,_16),_2)  h(3,1) = 19
tile(1,2): (_8,_16):(64,_1)  first=544.000000
SW128 atom: Sw<3,4,3> o smem_ptr[16b](unset) o (_8,_64):(_64,_1)
smem (64,64,3): Sw<3,4,3> o smem_ptr[16b](unset) o ((_8,_8),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_4096))  cosize=12288
tiled_mma size = 128 threads
thread0 accumulator: ((_2,_2,_8),_1,_1):((_1,_2,_4),_0,_0)
```

### 5.4 Debug and analysis tools

| Tool | Use |
|---|---|
| `print(x)` | Shows a layout, tensor, TiledCopy or TiledMMA. Works on the host and on the device. |
| `print_layout(l)` | Shows a rank-2 layout as a table. |
| `print_tensor(t)` | Shows the values of a tensor. Use it in one thread only: `if (thread0()) print_tensor(t);`. |
| `print_latex(tiled_mma)` | Writes a LaTeX picture of the thread/value assignment. Compile it with `pdflatex`. |
| `CUTE_STATIC_ASSERT_V(size(a) == size(b))` | Checks shapes at compile time. |
| `compute-sanitizer --tool racecheck/synccheck/memcheck` | Finds races, barrier errors and out-of-bounds access. `make sanitizer` runs the test suite with it. |
| `cuobjdump -sass <bin>` | Shows the SASS. Search for `HGMMA`, `UTMALDG`, `STSM`, `LDSM`, `SYNCS`. |
| `ncu --set full` | Measures one kernel: tensor-core use, smem bank conflicts, stall reasons. |
| `nsys profile` | Shows the timeline of a whole run. |

### 5.5 Record template for each lab

Copy this block into `progress.md` for each lab:

```markdown
### Lab NN — <title>
- Date:
- Goal: <one sentence>
- Repo code read: <file:line>
- Prediction: <what I expect to see>
- Result: <what I saw; paste print output>
- Difference and reason:
- Performance (if applicable): <time, TFLOPS, % of torch>
- Open questions:
```

---

## 6. Roadmap

### 6.1 Overview

```text
 Stage 0  Setup ────────────────────────────────────────────────┐ 1 day
    │                                                           │
 Stage 1  Layout algebra (host only)                            │ 1 week
    │                                                           │ FOUNDATION
 Stage 2  Tensors, partitioning, TiledCopy on GPU               │ 1 week
    │                                                           │
 Stage 3  Classic pipeline: cp.async + MMA                      ┘ 1 week
    │
 Stage 4  Hopper TMA + mbarrier ────────────────────────────────┐ 1 week
    │                                                           │ HOPPER
 Stage 5  WGMMA + warp-specialized persistent GEMM  ★ core      │ 2 weeks
    │                                                           ┘
    ├────────────────────┬──────────────────────┐
 Stage 6  Fused kernels:  Stage 7  FP8 and       │                ┐ 2–3 weeks
 flash attention          quantized GEMM         │                │ APPLICATION
    │                     │                      │                ┘
    └─────────┬───────────┘                      │
 Stage 8  Contribute to hpc-ops (operator / fix / perf)            1–2 weeks
    │
 Stage 9  (optional) Blackwell sm100: read and compile only
 Stage 10 (optional) Python CuTe DSL bridge
```

The times are estimates for part-time study. Each stage has **exit
criteria**. Do not go to the next stage until you meet them.

### Stage 0 — Setup

**Goal:** Make a working build, a working test run and a working lab build.

Procedure:

1. Install the developer tools: `pip install -r requirements-dev.txt`.
2. Build the sm90 module: `SM_ARCH=90 make`.
3. Run one CuTe-heavy test: `python3 -m pytest -v tests/test_gemm_bf16xfp32.py`.
4. Make `dive/practice/labs/` and the Makefile from section 5.2.
5. Build and run `lab01_layout.cu` from section 5.3.
6. Build and run `src/utils/test_tma.cu` and `src/utils/test_trans_0189.cu` (output to `/tmp` or `labs/`).

**Exit criteria:**

* The test passes.
* `python3 -c "import hpc; print(hpc.__loaded_arch__)"` prints `90`.
* `lab01` prints the output in section 5.3.

### Stage 1 — Layout algebra (host only)

**Goal:** Read and compute any CuTe layout by hand.

Read:

* CuTe docs in the CUTLASS GitHub repository: `media/docs/cpp/cute/01_layout.md`, `02_layout_algebra.md`.
* `src/utils/utils.cuh:433` (`retile_fragment`): it uses `flatten`, `select`, `take`, `append`, `coalesce`, `make_layout`.
* `src/utils/utils.cuh:705` (custom `make_tiled_copy`): it uses `raked_product` and `right_inverse`.

Labs (all host-only programs; each compiles in seconds):

| Lab | Task | Check |
|---|---|---|
| 1.1 | Print 10 layouts. Compute 3 coordinates by hand for each. | Values match `layout(coord)`. |
| 1.2 | `coalesce`, `composition`, `complement` on small layouts. | Draw each result as a table. |
| 1.3 | `logical_divide`, `zipped_divide`, `tiled_divide` of `(32,64)` by `(8,16)`. | Explain each mode of the result. |
| 1.4 | `logical_product`, `blocked_product`, `raked_product`. | Draw thread ownership for a 4×4 thread layout and a 2×2 value layout. |
| 1.5 | Print `GMMA::Layout_{K,MN}_SW{32,64,128}_Atom<bf16>` and `<fp8>`. Fill a table like section 4.4 for SW64. | Your table matches the bits. |
| 1.6 | Include `src/utils/utils.cuh` (`-DHPC_TARGET_ARCH=90`). Apply `hpc::retile_fragment` to the `tCr` layout from lab01. | Result is `(_2,(_2,_8)):(_2,(_1,_4))` = 2 rows × 16 columns per thread (tested). Explain it with section 4.8. |

**Exit criteria:** You can explain the output of `tile_to_shape(SW128, (64,64,3))` mode by mode, without a computer.

### Stage 2 — Tensors, partitioning and TiledCopy on the GPU

**Goal:** Write correct copy kernels with `make_tiled_copy`, `local_tile`,
`partition_S/D` and predication.

Read:

* `src/group_gemm/sm90/cp_async/config.h:55-89` (TiledCopy definitions).
* `src/group_gemm/sm90/cp_async/group_gemm_fp8.cu:95-160` (`local_tile`, identity tensor, `copy_if`).
* `src/attention/decode/sm90/splitk_combine_kernels.cuh` (330 lines, plain CuTe copies, no TMA/MMA).

Labs:

| Lab | Task | Check |
|---|---|---|
| 2.1 | Kernel: copy a `(M,N)` bf16 matrix with `make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>,bf16>, thr(16,8), val(1,8))`. | Equal to the input. SASS shows `LDG.E.128`. |
| 2.2 | Add predication with `make_identity_tensor` + `copy_if` for M, N not divisible by the tile. | Correct for M=1000, N=1000. |
| 2.3 | Transpose through smem. First a plain smem layout, then a swizzled one. | `ncu` shows bank conflicts in the first version and not in the second. |
| 2.4 | Run `test_trans_0189.cu`. Explain the "0189" interleave pattern from the printed table. | Write the pattern as a CuTe layout. |
| 2.5 | Write a torch reference for the split-K LSE merge. Compare it with the decode combine kernel (through `tests/test_attention_decode_*`). | Same result within 1e-3. |

**Exit criteria:** You can predict `size()` and `shape()` of every
`partition_S/D` result in `group_gemm_fp8.cu` before you print it.

### Stage 3 — Classic pipeline: cp.async + MMA

**Goal:** Understand a multistage software pipeline before you meet TMA.

Read:

* `src/group_gemm/sm90/cp_async/group_gemm_fp8.cu:20-230`. This kernel uses `cp.async` (Ampere style) to load and WGMMA to compute. It has no warp specialization.
* CUTLASS GitHub `examples/cute/tutorial/` (the sgemm and sm80 tutorials).

```text
 cp.async multistage (kStage = 3), 1 warpgroup does everything:

 prologue:  issue loads k0, k1         cp_async_fence after each
 loop k:    cp_async_wait<kStage-2>    (tile k is in smem)
            __syncthreads()
            issue load k+2  → slot (k+2)%3
            WGMMA on slot k%3
```

Labs:

| Lab | Task | Check |
|---|---|---|
| 3.1 | Single-tile bf16 GEMM: one CTA, `SM80_16x8x16_F32BF16BF16F32_TN` TiledMMA, gmem → smem with `cp.async`, smem → rmem with `SM75_U32x4_LDSM_N`. | Equal to `A @ B.T` (CPU) within bf16 tolerance. |
| 3.2 | Extend to full M, N, K with a 2-stage pipeline. Then 3 stages. | Measure TFLOPS for 4096³. Compare with `torch.matmul`. |
| 3.3 | In the repo cp.async path: change kStage 2 → 3 for `k%128==0`. Rebuild. Run `tests/test_group_gemm_cp_async.py`. | Test passes. Record time. |

**Exit criteria:** Your 3-stage SM80-style GEMM reaches ≥ 50 % of
`torch.matmul` on 4096³ bf16 on the H20.

### Stage 4 — Hopper TMA and mbarrier

**Goal:** Load and store tiles with TMA. Synchronize with mbarriers.

Read:

* `src/utils/test_tma.cu` (complete: descriptor build, device update, TMA load, print).
* `src/utils/tma.cuh` (descriptor patch helpers).
* `src/gemm/sm90/gemm.cu:106-222` (barrier ring, producer loop).
* `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp` (barrier functions).

Labs:

| Lab | Task | Check |
|---|---|---|
| 4.1 | Run `test_tma.cu`. Change `m_view` and the box. Predict which rows are zero-filled. | Printed smem matches the prediction. |
| 4.2 | Kernel: TMA-load a `(128,64)` bf16 tile with an SW128 smem layout, then TMA-store it to another buffer. | Output equals input. |
| 4.3 | Add a 4-stage ring with `readable/writable` barriers. One producer thread, 128 consumer threads that sum the tile. | Correct sum. `compute-sanitizer --tool synccheck` is clean. |
| 4.4 | Break it on purpose: use a wrong transaction byte count. Use the wrong start phase. | Record the symptom (hang, wrong data). This helps you find real bugs later. |

**Exit criteria:** You can draw section 4.6 from memory and explain why the
producer starts with `phase = 1`.

### Stage 5 — WGMMA and a warp-specialized persistent GEMM (core)

**Goal:** Write your own Hopper GEMM. Understand every line of
`src/gemm/sm90/gemm.cu`.

Read, in this order:

1. `src/gemm/sm90/gemm.cu:411-475` (host: layouts, TMA, TiledMMA, launch).
2. `src/gemm/sm90/gemm.cu:83-366` (kernel: roles, ring, WGMMA, epilogue).
3. `src/stem/sm90/stem_kernels.cuh:701` (OAM GEMM: 2 math WGs + 1 producer, the same pattern).
4. `src/group_gemm/sm90/kernels.cuh:218` (grouped version with descriptor update and tile schedulers).

Build your own GEMM in versions. Keep each version as a separate file.

| Version | Add | Check |
|---|---|---|
| v1 | 1 WG, 1 stage, TMA load, `SM90_64x64x16_F32BF16BF16_SS`, `warpgroup_wait<0>`, plain gmem store. | Correct for 64×64×K. |
| v2 | kStage = 4 ring with mbarriers (no warp roles yet; thread 0 issues TMA). | Correct for any M, N, K multiple of the tile. |
| v3 | Warp specialization: 2 math WGs + 1 producer WG, `setmaxnreg`. | TFLOPS increase vs v2. |
| v4 | Persistent grid (`grid = #SM`) + tile scheduler with block swizzle (copy `get_next_tile`, `gemm.cu:20`). | TFLOPS increase for large shapes. |
| v5 | Epilogue: `make_tiled_copy_C` + STSM + TMA store. | Fewer gmem store instructions in SASS. |
| v6 | `warpgroup_wait<1>` to overlap consecutive k-blocks. | Record the effect. |

Exercises in the repository code (see section 7 for risks):

* Print `SLayoutX`, `TiledMma`, `tXs`, `tWr`, `tYr_low` inside `gemm.cu` (thread 0 only). Explain every mode.
* In `group_gemm` per-tensor: apply the scale once after the K loop, not after every K-tile. Measure accuracy and time.
* Replace the raw mbarriers in one kernel with `cutlass::PipelineTmaAsync`. Compare the code size and the speed.

**Exit criteria:** Your v5 GEMM reaches ≥ 70 % of `torch.matmul` on
4096³ bf16, and you can explain all of `gemm.cu` to another engineer.

### Stage 6 — Fused kernels: flash attention

**Goal:** Combine two GEMMs, softmax, masks and paged memory in one kernel.

Read, in this order:

1. `src/attention/prefill/sm90/config.h:44-95` + `multi_stage_dim128.cu`.
2. `src/attention/prefill/sm90/kernels.cuh:3157` (multi-stage kernel: ~300 lines, no warp roles). **Read this first.**
3. `src/attention/prefill/sm90/kernels.cuh:219` (bf16 warp-specialized kernel).
4. `src/attention/prefill/sm90/kernels.cuh:573` (paged KV cache: the block id is the 5th TMA coordinate).
5. `src/attention/decode/sm90/util_kernels.cuh` (decode helpers: `load_paged_kv`, `online_softmax`, `advance_stage`).

Labs:

| Lab | Task | Check |
|---|---|---|
| 6.1 | Print `layout_asC`, `layout_asA`, and `left_inverse(asC).compose(asA)` for the 64×64 SS / 64×128 RS pair. | Show that the thread part is the identity. |
| 6.2 | Print `retile_fragment(tAttr)` and the partitioned identity tensor. Make a table: thread → (rows, cols). | Matches section 4.8. |
| 6.3 | Write your own flash-attention forward from your Stage 5 GEMM: one Q tile, causal, head dim 128, no paging. | Matches `torch.nn.functional.scaled_dot_product_attention`. |
| 6.4 | In the repository: change the multi-stage kernel `kStage` from 1 to 2. | `tests/test_attention_prefill_bf16.py` passes. Record time. |
| 6.5 | Add a sliding-window mask to the identity-tensor mask loop. | New test against a torch reference. |
| 6.6 | Add FA3-style overlap: start QK of tile j+1 before softmax of tile j. | Record speedup. |

**Exit criteria:** Your 6.3 kernel is correct for causal and non-causal
cases, and you can explain how the P registers become the A operand.

### Stage 7 — FP8 and quantized GEMM

**Goal:** Handle FP8 operands, scales and the FP8 operand-layout rules.

Read:

* `src/group_gemm/sm90/kernels.cuh:451-477` (per-tensor scale per K-tile).
* `src/group_gemm/sm90/kernels.cuh:720-840` (blockwise scale via TMA and identity tensor).
* `src/attention/prefill/sm90/kernels.cuh:1340-1405` (P × 256, `__byte_perm`, transposed V, two-level accumulation).
* `src/utils/utils.cuh:461-540` (`smem_trans_and_interleave0189_*`).

```text
 Why FP8 needs extra work in attention:

 bf16:  P (C layout, fp32) ──re-index──▶ A layout (bf16)          V used MN-major
 fp8 :  P (C layout, fp32) ──×256, cvt──▶ fp8x4 ──byte_perm──▶ A layout (fp8)
        fp8 WGMMA needs K-major B  ⇒  V must be transposed in smem (LDSM_T → byte_perm → STSM)
        the k order of P changes   ⇒  V rows get the same "0189" permutation
```

Labs:

| Lab | Task | Check |
|---|---|---|
| 7.1 | Stage 5 GEMM with `SM90_64x64x32_F32E4M3E4M3_SS_TN` and a per-tensor scale. | Matches torch fp8 reference (dequantize, then matmul). |
| 7.2 | Add 1×128 / 128×128 blockwise scales. | Matches the reference. |
| 7.3 | Derive the 8-bit A-fragment layout by printing it. Explain the `0x5410` / `0x7632` byte_perm selectors. | Written explanation + diagram. |

**Exit criteria:** You can explain why the fp8 attention kernel cannot use
the `left_inverse ∘ compose` trick that the bf16 kernel uses.

### Stage 8 — Contribute to hpc-ops

**Goal:** Use the full repository workflow: wrapper, `entry.cc`,
`HPC_ARCH_DISPATCH`, kernel, test and format.

Procedure (from `CLAUDE.md`, "Add an operator"):

1. Add `hpc/<op>.py` with `torch.library.register_fake`.
2. Add `src/<op>/<op>.h` with the `*_async` declaration.
3. Add `src/<op>/entry.cc` with validation, allocation, `HPC_ARCH_DISPATCH("<op>", 90, ...)` and `TORCH_LIBRARY_FRAGMENT`.
4. Move your Stage 5 or 6 kernel into `src/<op>/sm90/`.
5. Add `tests/test_<op>.py`.
6. Run `SM_ARCH=90 make`, the focused test, `make format-check` and `make sanitizer`.

Candidate topics that the scan found (verify each one first; section 7):

* `src/attention/prefill/sm90/kernels.cuh`: no `tma_store_wait` call in the file. Check for a race on the `sY` staging buffer in the persistent kernels.
* The "multi_stage" attention config uses `kStage = 1`. Test 2 stages.
* Attention and GEMM kernels use `warpgroup_wait<0>` after each GEMM. Test overlap with `warpgroup_wait<1>`.
* `syncwarpgroup(iwarpgroup)` uses named barrier id 0 for WG0. `__syncthreads` also uses id 0 (`src/utils/utils.cuh:605`).

**Exit criteria:** One change passes the tests, `make format-check` and
`make sanitizer`, and is ready for review.

### Stage 9 — (optional) Blackwell sm100

**Goal:** Read a tcgen05 / TMEM kernel. Know the differences from sm90.

* Read `src/gemm/sm100/gated_mla_gemm.cu` and `.cuh` with section 4.12.
* Compile with `SM_ARCH=100 make` to check your changes. You cannot run sm100 code on the H20.
* Compare the warp roles, the barrier rings and the epilogue with `src/gemm/sm90/gemm.cu`.

### Stage 10 — (optional) Python CuTe DSL bridge

**Goal:** Use the same concepts in the Python DSL.

* Install `nvidia-cutlass-dsl` in a separate virtual environment. Do not change the repository environment.
* Port lab 1.x (layouts) and your Stage 5 v1 GEMM to `cutlass.cute`.
* Map the names: `make_layout`, `make_tensor`, `local_tile`, `make_tiled_copy`, `make_tiled_mma` and `cute.gemm` have the same meaning.
* Compare compile time and generated code with the C++ version.

---

## 7. Cautions and known pitfalls

The scan found these items by reading. Most were **not** checked with a
run. Verify each item before you act on it.

| # | Item | Location | Status |
|---|---|---|---|
| 1 | Swap-AB: MMA "A" is the weight, accumulator is (N, M). Easy to misread. | gemm/sm90, group_gemm, decode, gemm/sm100 | Design fact |
| 2 | TMA transaction bytes must equal the full box, also for tail tiles. | all TMA kernels | Design fact |
| 3 | Named barrier id 0 is shared by `__syncthreads` and `syncwarpgroup(0)`. | `src/utils/utils.cuh:605` | Read, not tested |
| 4 | No `tma_store_wait` in prefill kernels. Possible `sY` reuse race. | `src/attention/prefill/sm90/kernels.cuh` | grep confirmed absence; race not tested |
| 5 | `online_softmax` has no guard for a row that is all `-inf`. Possible NaN in block-sparse paths. | prefill `kernels.cuh:63-94` | Read, not tested |
| 6 | A caller-supplied `split_flag` must be zero before each decode call. | `src/attention/entry.cc:158-161` | Read, not tested |
| 7 | cp.async group GEMM: if `k % 64 != 0`, nothing launches and the output stays uninitialized. | `group_gemm_fp8.cu:331` | Read, not tested |
| 8 | Any path that contains `test` is excluded from the CMake build. | `CMakeLists.txt:61` | Confirmed |
| 9 | KV-cache tail slots must be zero. TMA loads whole pages. | `hpc/attention.py` docstring | Documented |
| 10 | Six warp-spec attention kernels are near-copies. A fix in one must go to all. | prefill `kernels.cuh` | Read |

---

## 8. Reading order (summary)

| Order | File | Lines | Focus | Stage |
|---|---|---|---|---|
| 1 | `src/utils/test_trans_0189.cu` | 80 | smem tensors, `local_tile`, print | 1–2 |
| 2 | `src/utils/test_tma.cu` | 110 | TMA build, update, load | 4 |
| 3 | `src/utils/utils.cuh` | 400–720 | reductions, `retile_fragment`, barriers, custom TiledCopy | 1–6 |
| 4 | `src/utils/tma.cuh` | 71 | descriptor patch | 4 |
| 5 | `src/group_gemm/sm90/cp_async/config.h` + `group_gemm_fp8.cu` | 450 | TiledCopy, cp.async pipeline, predication | 2–3 |
| 6 | `src/gemm/sm90/gemm.cu` | 557 | **complete Hopper GEMM** | 5 |
| 7 | `src/stem/sm90/stem_kernels.cuh` | 698–1030 | second WS GEMM | 5 |
| 8 | `src/group_gemm/sm90/config.h` + `kernels.cuh` | 1070 | grouped FP8 GEMM, schedulers | 5, 7 |
| 9 | `src/attention/prefill/sm90/kernels.cuh` | 3157–3454 | multi-stage flash attention | 6 |
| 10 | `src/attention/prefill/sm90/kernels.cuh` | 219–570 | warp-spec flash attention | 6 |
| 11 | `src/attention/decode/sm90/util_kernels.cuh` | 830 | decode helpers, paged KV | 6 |
| 12 | `src/attention/prefill/sm90/kernels.cuh` | 969–1460 | FP8 attention | 7 |
| 13 | `src/gemm/sm100/gated_mla_gemm.cuh` | 519 | tcgen05, TMEM, CLC | 9 |

---

## 9. Progress tracker

```text
[x] Stage 0  Setup: build, test, lab01                     → stage0.md
[x] Stage 1  Layout algebra        labs 1.1–1.6            → stage1.md
[ ] Stage 2  Tensors + TiledCopy   labs 2.1–2.5
[ ] Stage 3  cp.async + MMA        labs 3.1–3.3
[ ] Stage 4  TMA + mbarrier        labs 4.1–4.4
[ ] Stage 5  WS persistent GEMM    v1–v6
[ ] Stage 6  Flash attention       labs 6.1–6.6
[ ] Stage 7  FP8                   labs 7.1–7.3
[ ] Stage 8  Contribution          1 merged-ready change
[ ] Stage 9  sm100 (optional)
[ ] Stage 10 Python DSL (optional)
```

---

## 10. References

| Reference | Location |
|---|---|
| Repository development guide | `CLAUDE.md` (build, layout, dispatch rules) |
| CuTe headers (this version) | `3rd/cutlass/include/cute/` |
| CuTe documentation | GitHub `NVIDIA/cutlass`, folder `media/docs/cpp/cute/` (layout, layout algebra, tensor, algorithms, MMA atom, GEMM tutorial) |
| CuTe examples | GitHub `NVIDIA/cutlass`, folder `examples/cute/tutorial/` |
| Vendored commit | `da5e086` (see `3rd/update-cutlass.sh`). Use docs from the same version when you can. |
| PTX ISA | NVIDIA PTX ISA: sections on `wgmma`, `cp.async.bulk.tensor`, `mbarrier`, `stmatrix`, `tcgen05` |
| Python CuTe DSL | PyPI `nvidia-cutlass-dsl`; GitHub `NVIDIA/cutlass`, folder `python/CuTeDSL/` and `examples/python/CuTeDSL/` |
