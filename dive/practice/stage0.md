# Stage 0 — Environment record

| Item | Value |
|---|---|
| Date | 2026-10-26 |
| Repository | `/root/hpc-ops`, commit `2a2e265` |
| GPU | NVIDIA H20, compute capability 9.0, driver 535.161.08 |
| CUDA / nvcc | 12.8.93 |
| cmake | 4.2.0 |
| torch | 2.8.0+cu128 |
| Result | **Stage 0 passed.** The module builds, loads and passes the GEMM check. |

---

## 1. What Stage 0 must prove

| # | Item | Exit check |
|---|---|---|
| 1 | The repository builds for sm90 | `SM_ARCH=90 make` exits 0 |
| 2 | A CuTe-heavy operator is correct | `tests/test_gemm_bf16xfp32.py`, all cases |
| 3 | `import hpc` loads the sm90 module | `hpc.__loaded_arch__ == 90` |
| 4 | A standalone CuTe program builds and runs | `labs/lab01_layout` |
| 5 | The practice workspace works | `labs/Makefile` builds a lab in ≤ 30 s |

All five items pass. See section 6 for the one deviation (pytest).

---

## 2. Build the module

```bash
cd /root/hpc-ops
SM_ARCH=90 make          # 2 min 13 s, 64 parallel jobs
```

Output:

```text
build/sm90/_C_sm90.abi3.so                          146 MB
build/lib.linux-x86_64-cpython-312/hpc/_C_sm90.abi3.so   (copy that the tests import)
```

Build facts from the log:

| Fact | Value |
|---|---|
| Gencode | `sm_90a` |
| Compiler warnings | 0 |
| Build time | 2 min 13 s wall, 28 min 40 s CPU (≈ 13× parallelism) |
| Build command | `python3 setup.py build` → cmake + ninja per arch in `build/sm90` |
| Log | `dive/practice/logs/stage0_build.log` |

Check the loaded module:

```python
>>> import hpc
>>> hpc.__loaded_arch__
90
>>> hpc.__built_json__
{'version': '0.0.1.dev0+g2a2e265.sm90', 'git-hash': '2a2e265', 'sm-arch': 'sm90',
 'g++': 'g++-11.4.0', 'nvcc': 'nvcc-12.8.93', 'cutlass': '4.4.2',
 'cub': '2.7.0', 'stdc++': '202002.', 'glibc': '2.35', ...}
```

Note: import `hpc` from `build/lib.linux-x86_64-cpython-312`, not from the repository
root. The repository copy has no `.so` yet, so it raises. The test files do this for
you: `sys.path.insert(0, "../build/lib.*/")`.

---

## 3. Correctness check for the CuTe GEMM

The GEMM test has 18 × 4 × 1 × 2 × 2 = 288 cases (m × n × k × output dtype ×
split-flag mode).

**Deviation:** `pytest` is not installed in this environment (see section 6). To
complete the exit check, `dive/practice/labs/stage0_gemm_smoke.py` repeats the
exact body of `tests/test_gemm_bf16xfp32.py`, with the same seed, shapes,
reference and tolerance, in a plain loop.

```bash
python3 dive/practice/labs/stage0_gemm_smoke.py
```

```text
loaded arch: 90 device: NVIDIA H20
288 passed, 0 failed
```

The script checks more than the pass/fail of one call:

* The reference is `x.float() @ w.t()` in fp32, tolerance `rtol=0.08, atol=0.01`.
* When `split_flag` is used, it checks that the kernel resets every flag to 0.
  If the kernel does not reset them, the second use of the same workspace is wrong.
* The shapes include m = 1, 6 (tile M smaller than the MMA M) and m = 12303, 32768
  (odd and very large row counts), so the tile scheduler and the tail tiles are covered.

This operator is a good first read because it uses most CuTe surfaces at once:
`make_tensor` + `make_gmem_ptr`, `tile_to_shape` + GMMA swizzle, `make_tma_copy`,
`partition_S/D`, `make_tiled_mma`, `partition_A/B`, `make_fragment_A/B`,
`partition_fragment_C`, `warpgroup_*`, `make_tiled_copy_C`, `retile_S`, TMA store,
and a split-K reduce. Stage 5 reads it in detail.

---

## 4. Standalone CuTe programs

| Program | Build | Run | Result |
|---|---|---|---|
| `src/utils/test_trans_0189.cu` | 18 s | ok | Prints the 64×64 smem transpose table |
| `src/utils/test_tma.cu` | 25 s | ok | `err = 0`; smem rows 0–1 filled, rows 2–7 zero |
| `src/utils/test_vec_t.cu` | 15 s | ok | `err = 0`; prints the `vec_t` value pattern |
| `dive/practice/labs/lab01_layout.cu` | 12 s | ok | Same output as `record.md` section 5.3 |

`test_tma.cu` demonstrates an important behaviour. The host builds the descriptor
for `(m=5, k=16)` but the device patches it to `m_view = 3` before the copy. The TMA
load then zero-fills rows 3 and 4 of the 8-row box:

```text
box (8,16) from a 3-row view:  rows 0,1,2 = data, rows 3..7 = 0
```

This "box is always full, out-of-bounds part is zero-filled" rule is why the
transaction byte count in the kernels is the full box size (see `record.md`
section 4.5).

`test_tma.cu` also shows the descriptor-update pattern that the variable-length
and MoE kernels use: copy the template descriptor to smem, patch it
(`hpc::update_tma_gtensor`), then `tma_descriptor_cp_fence_release` to gmem.

---

## 5. Practice workspace

```text
dive/practice/
├── record.md              the roadmap
├── stage0.md              this file
├── stage1.md              lab 1.1–1.6 record
├── logs/                  build and run logs
└── labs/
    ├── Makefile           nvcc, -arch=sm_90a, -DHPC_TARGET_ARCH=90
    ├── .gitignore         bin/
    ├── check.hpp          CHECK_EQ / CHECK_LAYOUT / CHECK_SAME_FN / SHOW
    ├── lab01_layout.cu    starter (record.md section 5.3)
    ├── lab11_eval.cu      …lab16_retile.cu  (Stage 1)
    └── stage0_gemm_smoke.py
```

Commands:

```bash
cd dive/practice/labs
make             # build every lab -> bin/
make run         # build and run
./bin/lab01_layout
make clean
```

Why this location:

* CMake globs only `src/**/*.cu`, so `dive/` files never enter the module.
* The include path `-I$(ROOT)` lets a lab use `src/utils/utils.cuh` (lab 1.6 does).
* `-DHPC_TARGET_ARCH=90` is required by `src/utils/utils.h` when a lab includes
  the repository helpers.

---

## 6. Deviations and open items

| Item | Status | Work-around used | Suggested fix |
|---|---|---|---|
| `pytest` not installed | Open | `stage0_gemm_smoke.py` | `pip install pytest==8.4.1` |
| `black`, `cpplint`, `clang-format` not installed | Open | not needed for Stage 0/1 | `pip install black==25.1.0 cpplint==2.0.2 clang-format==20.1.8` |
| `pip install` was blocked in this session | Open | none | run it in a shell with network access, or `!pip install ...` |
| Old `hpc-ops 1.1.0` in site-packages | Known | run tests from `build/lib.*` | none needed; the test files pick the build directory first |
| `requirements-dev.txt` pins `torch==2.7.0`, machine has 2.8.0 | Known | no rebuild | do not downgrade torch to run the tests; 2.8.0 passed all 288 cases |

`black`/`cpplint` are only needed for `make format` / `make format-check`, which
Stage 8 needs. Stages 1–7 do not.

---

## 7. Summary

| Exit check | Result |
|---|---|
| `SM_ARCH=90 make` | Pass, 0 warnings, 2 min 13 s |
| `hpc.__loaded_arch__` | 90 |
| GEMM correctness (288 cases) | 288 passed, 0 failed |
| `test_tma.cu`, `test_trans_0189.cu`, `test_vec_t.cu` | Run clean, `err = 0` |
| `labs/lab01_layout` | Builds in 12 s, output as documented |
| `labs/Makefile` | 7 labs build in 10 s with `-j8` |

Stage 0 is complete. Go to Stage 1 (`stage1.md`).
