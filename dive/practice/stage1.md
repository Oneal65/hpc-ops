# Stage 1 — Layout algebra: lab record

| Item | Value |
|---|---|
| Date | 2026-10-26 |
| Labs | `dive/practice/labs/lab11_eval.cu` … `lab16_retile.cu` |
| Build | `make -j8` → 6 labs in 10 s, 0 warnings |
| Result | **All labs pass.** 4881 checks, 0 failures. |
| Method | Write the predicted value in the source first. The program compares it with CuTe. |

```text
lab11_eval     42 passed, 0 failed     layout evaluation, 11 layouts x coords/size/cosize
lab12_algebra  26 passed, 0 failed     coalesce, composition, complement
lab13_divide   17 passed, 0 failed     logical / zipped / tiled / flat divide, local_tile
lab14_product  84 passed, 0 failed     logical / blocked / raked product, TV layout
lab15_swizzle 549 passed, 0 failed     GMMA swizzle atoms, chunk tables, tile_to_shape
lab16_retile 4163 passed, 0 failed     retile_fragment, WGMMA accumulator owner map
```

Every `CHECK_*` in these labs was written **before** the first run, from the
definitions. Section 6 lists the two predictions that were wrong, and why.

---

## 1. Lab 1.1 — A layout is a function

File: `labs/lab11_eval.cu` → `bin/lab11_eval`

### Method

1. Read the layout `shape:stride`.
2. Compute `index = Σ coord_i * stride_i` by hand.
3. For a 1-D index into a multi-mode layout, split it colexicographically first:
   `i → (i % s0, i / s0)` for rank 2, left mode fastest.
4. Write the number in `CHECK_EQ(layout(coord), number)`.
5. Run the program. A `FAIL` line shows the CuTe value next to the prediction.

### Results

| Layout | Printed | Probes (predicted = actual) |
|---|---|---|
| L1 `(4,8):(8,1)` row-major | `(_4,_8):(_8,_1)` | `(1,2)=10`, `(3,7)=31`, `(2,0)=16`, `1-D (10)=18`, `size=32`, `cosize=32` |
| L2 `(4,8):(1,4)` default | `(_4,_8):(_1,_4)` | `(1,2)=9`, `(3,7)=31`, `(2,0)=2`, `1-D (10)=10` |
| L3 `(4,8):(1,5)` padded | `(_4,_8):(_1,_5)` | `(1,2)=11`, `(3,7)=38`, `(0,1)=5`, `size=32`, `cosize=39` |
| L4 `8:2` | `_8:_2` | `(3)=6`, `(7)=14`, `cosize=15` |
| L5 `((2,2),8):((1,16),2)` | `((_2,_2),_8):((_1,_16),_2)` | `(3,1)=19`, `(1,7)=15`, `(2,3)=22`, natural coord `((1,1),1)=19` |
| L6 `(4,(2,4)):(2,(1,8))` | `(_4,(_2,_4)):(_2,(_1,_8))` | `(1,(1,2))=19`, `(3,(0,3))=30`, `(2,5)=21`, `cosize=32` |
| L7 `(4,8):(0,1)` broadcast | `(_4,_8):(_0,_1)` | `(3,5)=5`, `(0,7)=7`, `size=32`, `cosize=8` |
| L8 `(2,3,4):(12,4,1)` | `(_2,_3,_4):(_12,_4,_1)` | `(1,2,3)=23`, `(0,1,0)=4`, `(1,0,2)=14` |
| L9 `(2,3,4)` default | `(_2,_3,_4):(_1,_2,_6)` | `stride<2>=6`, `(1,2,3)=23`, `(0,1,0)=2`, `(1,0,2)=13` |
| L10 dynamic `(6,10):(10,1)` | `(6,10):(10,_1)` | `(5,9)=59`, `(1,0)=10`, `(2,3)=23`, `1-D (7)=11` |
| L11 `LayoutRight` | `(_2,_3,_4):(_12,_4,_1)` | equal to L8 |

### What the numbers teach

**1. `size` is the product of the shape; `cosize` is the extent of the image.**

```text
 L3 = (4,8):(1,5)     size = 4*8 = 32     cosize = 39
 ┌────┬────┬────┬────┬────┬────┬────┬────┐
 │ 0  │ 1  │ 2  │ 3  │    │ 5  │ 6  │ 7  │ 8
 ├────┼────┼────┼────┼────┼────┼────┼────┼────
 │    │ .. │    │    │    │    │    │    │
 └────┴────┴────┴────┴────┴────┴────┴────┘
  col 0 uses 0..3, col 1 uses 5..8, ...   slot 4, 9, ... are unused

 L7 = (4,8):(0,1)     size = 32           cosize = 8
 4 rows all alias the same 8 elements (broadcast) — the memory is 4x smaller than the shape
```

`cosize` decides shared-memory size in every kernel (`gemm.cu:439`). If you size
with `size`, a padded or broadcast layout over- or under-allocates.

**2. CuTe's default is column-major.** `make_layout(shape)` gives `LayoutLeft`,
the opposite of a C array. L2, L9 and L10(1-D) show it. In a GEMM kernel you
normally write the row-major strides yourself, as `gemm.cu:422` does:
`make_stride(k, Int<1>{})`.

**3. Static and dynamic print differently.** `_8` is a compile-time `Int<8>{}`;
`8` is a run-time `int`. The compiler can unroll and constant-fold static modes
only. `L10 = (6,10):(10,_1)` has one dynamic mode (m) and one static (the inner
stride), which is the usual mix: shapes dynamic, the innermost stride static.

**4. A 1-D index is not a "flattened coords" index.** `L1(10) = 18` because 10
splits into `(10 % 4, 10 / 4) = (2,2)`. For a column-major layout the 1-D index
equals the offset (`L2(10) = 10`). This matters when you write
`tensor(i)` on a multi-mode tensor.

### Reproduce

```bash
cd dive/practice/labs && make bin/lab11_eval && ./bin/lab11_eval
```

---

## 2. Lab 1.2 — coalesce, composition, complement

File: `labs/lab12_algebra.cu` → `bin/lab12_algebra` (26 checks)

### coalesce: remove redundancy

Rule: modes `i` and `i+1` merge when `stride[i+1] == shape[i] * stride[i]`. A mode
of shape 1 always drops.

| Input | Output | Why |
|---|---|---|
| `(_2,(_1,_6)):(_1,(_6,_2))` | `_12:_1` | shape-1 mode drops; then `(_2,_6):(_1,_2)` merges → `12:1` |
| `(_4,_8):(_8,_1)` row-major | `(_4,_8):(_8,_1)` | `1 ≠ 4*8`, no merge |
| `(_4,_8):(_1,_4)` column-major | `_32:_1` | `4 == 4*1`, merge |
| `((2,4),(3,5)):((1,2),(8,24))` by-mode | `(_8,_15):(_1,_8)` | mode 0 → `8:1`, mode 1 → `15:8` |

`gemm.cu` uses coalesce on purpose: the TMA box `take<0,2>(slayout)` keeps the
stage mode out, and `Coalesce`-style calls in `retile_fragment`
(`utils.cuh:433`) merge the accumulator modes into `(rows, cols)`.

### composition: `R(i) = A(B(i))`

| A | B | R | Meaning |
|---|---|---|---|
| `(6,2):(8,2)` | `(4,3):(3,1)` | `((2,2),3):((24,2),8)` | B's rows 0–1 and 2–3 do not land on A's rows contiguously, so the mode splits |
| `20:2` | `(5,4):(4,1)` | `(5,4):(8,2)` | same rank; the stride of A multiplies into every stride of B |
| `(6,8):(8,1)` | `(3,4):(8,1)` (a tiler) | `(3,4):(8,1)` | take a sub-block of an existing layout |

The test also checks `R1(i) == A1(B1(i))` for every `i`. That is the definition:
composition is functional composition, not a layout rewrite.

The third row is the "tiler" use: `composition(big, tile)` gives the layout of a
tile **inside** `big`, with `big`'s strides. This is what `local_tile` builds
(lab 1.3).

### complement: fill the holes

`complement(A, M)` returns a layout `C` such that `(A, C)` is a bijection onto
`[0, M)`. Here `M = 24`.

| A | complement(A, 24) | Why |
|---|---|---|
| `4:1` | `6:4` | A covers 0–3, C steps by 4, six times |
| `6:4` | `4:1` | A covers multiples of 4, C fills the gaps with stride 1 |
| `4:2` | `(2,3):(1,8)` | A covers even offsets; C needs stride 1 (2 values) then stride 8 |
| `(2,2):(1,6)` | `(3,2):(2,12)` | A covers `{0,1,6,7}`; C covers `{2..5, 8..}` in two modes |

The lab checks the property directly: `(A, complement(A))` visits all 24 offsets
exactly once.

Complement is the hidden half of `logical_divide` and `logical_product` — see
labs 1.3 and 1.4. If you understand complement, the divide and product results
stop looking arbitrary.

### Reproduce

```bash
make bin/lab12_algebra && ./bin/lab12_algebra
```

---

## 3. Lab 1.3 — The four divides

File: `labs/lab13_divide.cu` → `bin/lab13_divide` (17 checks)

Setup: `A = (32,64):(64,1)` (row-major), tiler `T = (8,16)`.

```text
 A: 32 rows x 64 cols          T: 8 x 16 tile
 ┌────┬────┬────┬────┐   the tile grid is 4 x 4 = 16 tiles
 │T00 │T01 │T02 │T03 │   tile (1,2) starts at row 8, col 32 -> offset 8*64+32 = 544
 ├────┼────┼────┼────┤
 │T10 │T11 │T12█│T13 │  ◀── tile (1,2)
 ├────┼────┼────┼────┤
 │T20 │T21 │T22 │T23 │
 ├────┼────┼────┼────┤
 │T30 │T31 │T32 │T33 │
 └────┴────┴────┴────┘
```

| Operation | Result | Mode grouping |
|---|---|---|
| `logical_divide(A, T)` | `((_8,_4),(_16,_4)):((_64,_512),(_1,_16))` | `(tile, rest)` per mode — rest = **which** tile |
| `zipped_divide(A, T)` | `((_8,_16),(_4,_4)):((_64,_1),(_512,_16))` | `(tile-as-a-whole, rest)` |
| `tiled_divide(A, T)` | `((_8,_16),_4,_4):((_64,_1),_512,_16)` | zipped, with the rest mode split into separate modes |
| `flat_divide(A, T)` | `(_8,_16,_4,_4):(_64,_1,_512,_16)` | tiled, fully flattened |

1-D test: `logical_divide(24:1, 4:2)` = `(_4,(_2,_3)):(_2,(_1,_8))`. In words: step
by 2, four times; then two interleaved streams 1 apart; then advance 8 at a time.

### The four are the same function, grouped differently

The lab proves this by evaluating all 2048 coordinates:

```text
 ld((r, tm), (c, tn))  ==  zd((r, c), (tm, tn))     for all r<8, c<16, tm<4, tn<4
```

But as **1-D** functions they differ, because the 1-D index walks the modes in a
different order:

```text
 ld(1) = 64    zd(1) = 1      ld(8) = 512   zd(8) = 1
 └ ld's first mode is the 4 tile-rows (stride 512), zd's first mode is the 16 tile-cols
```

Rule of thumb: use `zipped_divide` when you want "one tile as a tensor"; use
`logical_divide` when you want "tile and rest stay paired per mode"; use
`flat_divide` when you address coordinates `(r, c, tm, tn)` by hand.

### local_tile is zipped_divide + a slice

```text
 t12 = local_tile(T, (8,16), (1,2))
 t12.layout() = (_8,_16):(_64,_1)        t12(0,0) = 544        t12(7,15) = 544 + 7*64 + 15
 tall = local_tile(T, (8,16), (_, _))
 tall.layout() = (_8,_16,_4,_4):(_64,_1,_512,_16)   ← identical to flat_divide
```

That second line is the useful identity: **`local_tile(..., (_, _))` == `flat_divide`**.
The GEMM kernels use the first form (`local_tile(A, make_tile(...), make_coord(itile_m, _))`)
and the TMA partitioners produce the second form (`(TMA, TILE_M, TILE_K)`).

The last check uses the `Step<_1,_1,X>` argument with a rank-3 tiler. That is the
form in the paged-KV code: `local_tile(gK, tile(block, dim, head), coord(_, _, ih), Step<_1,_1,X>{})`
drops the third coordinate mode and keeps the tile layout.

### Reproduce

```bash
make bin/lab13_divide && ./bin/lab13_divide
```

---

## 4. Lab 1.4 — Product, and the thread-value layout

File: `labs/lab14_product.cu` → `bin/lab14_product` (84 checks)

Setup: `thr = (4,4):(1,4)` (16 threads, column-major), `val = (2,2):(1,2)`
(4 values per thread). The layout maps an `(8,8)` coordinate to
`thread_index + 16 * value_index`.

### blocked vs raked

```text
 blocked_product(thr, val) = ((_4,_2),(_4,_2)):((_1,_16),(_4,_32))
   the TILE of threads is the inner block: thread ids repeat every 4x4 block

       0    1    2    3    4    5    6    7
     +----+----+----+----+----+----+----+----+
  0  |  0 |  4 |  8 | 12 | 32 | 36 | 40 | 44 |
  1  |  1 |  5 |  9 | 13 | 33 | 37 | 41 | 45 |
  2  |  2 |  6 | 10 | 14 | 34 | 38 | 42 | 46 |
  3  |  3 |  7 | 11 | 15 | 35 | 39 | 43 | 47 |
  4  | 16 | 20 | 24 | 28 | 48 | 52 | 56 | 60 |
  ...                                          ← thread 0 owns (0,0),(0,1),(1,0),(1,1): a 2x2
                                                 block, then jumps 16

 raked_product(thr, val) = ((_2,_4),(_2,_4)):((_16,_1),(_32,_4))
   the VALUE block is inner: each thread owns one contiguous 2x2 block

       0    1    2    3    4    5    6    7
     +----+----+----+----+----+----+----+----+
  0  |  0 | 32 |  4 | 36 |  8 | 40 | 12 | 44 |
  1  | 16 | 48 | 20 | 52 | 24 | 56 | 28 | 60 |
  2  |  1 | 33 |  5 | 37 |  9 | 41 | 13 | 45 |
  3  | 17 | 49 | 21 | 53 | 25 | 57 | 29 | 61 |
  ...                                          ← thread 0 = offsets 0..3 = a 2x2 block
```

Thread 5 = `(1,1)` in `thr`. In `raked_product` its 4 values are
`(2,2), (3,2), (2,3), (3,3)` — the 2×2 block at rows 2–3, cols 2–3.

**Why it matters:** loads from global memory want *each thread to read 16
contiguous bytes*. `raked_product` builds that pattern (one value block per
thread). The copy atoms in `group_gemm/sm90/cp_async/config.h` and the TMA boxes
use this shape. `blocked_product` gives the "tile of threads" pattern used by
`make_tiled_mma` (a warp owns the tile, values are spread).

### TV layout

`right_inverse(raked_product)` with shape `(threads, values)` gives the layout
CuTe stores inside a `TiledCopy`:

```text
 tv = ((_4,_4),(_2,_2)):((_2,_16),(_1,_8))     (thr, val) -> offset in the (8,8) tile

 thread 5, values 0..3 -> offsets 18, 19, 26, 27      (the 2x2 block at rows 2-3, cols 2-3)
```

The lab checks that `raked_product(tv(t, v)) == t + 16*v` for all 64 pairs — the
two are exact inverses.

### The same thing as a real Copy atom

`make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, bf16>{}, thr=(16,8), val=(1,8))`
prints:

```text
TiledCopy
  Tiler_MN:       (_16,_64)          ← 16 rows x (8 threads x 8 values = 64) cols
  TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)
Copy_Atom
  ValLayoutSrc:   (_1,_8):(_0,_1)    ← 8 bf16 = 16 bytes per thread per instruction
```

`size(tiled_copy) = 128` threads. The tile is `16 × 64` elements. This is the
`local_partition`/`partition_S` machinery in one line: a TiledCopy is a Tiler
plus a TV layout plus a Copy atom.

### Reproduce

```bash
make bin/lab14_product && ./bin/lab14_product
```

---

## 5. Lab 1.5 — Swizzle

File: `labs/lab15_swizzle.cu` → `bin/lab15_swizzle` (549 checks)

### Atoms

| Atom | Printed | Rows × cols | Bytes per row |
|---|---|---|---|
| `Layout_K_INTER_Atom<bf16>` | `Sw<0,4,3> o ... o (_8,_8):(_8,_1)` | 8 × 8 | 16 |
| `Layout_K_SW32_Atom<bf16>` | `Sw<1,4,3> o ... o (_8,_16):(_16,_1)` | 8 × 16 | 32 |
| `Layout_K_SW64_Atom<bf16>` | `Sw<2,4,3> o ... o (_8,_32):(_32,_1)` | 8 × 32 | 64 |
| `Layout_K_SW128_Atom<bf16>` | `Sw<3,4,3> o ... o (_8,_64):(_64,_1)` | 8 × 64 | **128** |
| `Layout_K_SW128_Atom<fp8>` | `Sw<3,4,3> o smem_ptr[8b] o (_8,_128):(_128,_1)` | 8 × 128 | 128 |
| `Layout_MN_SW128_Atom<bf16>` | `Sw<3,4,3> o ... o (_64,_8):(_1,_64)` | 64 × 8 | 16 (M is contiguous) |
| `Layout_MN_SW64_Atom<bf16>` | `Sw<2,4,3> o ... o (_32,_8):(_1,_32)` | 32 × 8 | 32 |

Rules read off the table:

1. **One atom row is always 8 rows.** The swizzle period is 8 rows.
2. **Row width in bytes = swizzle width** (32 / 64 / 128 B). So the column count
   is `width / sizeof(T)`: bf16 SW128 → 64 columns; fp8 SW128 → 128 columns.
3. `Layout_K_*` is K-major (row stride 1 in elements, rows are 128 B apart).
   `Layout_MN_*` is M-major (M stride 1), used when M is contiguous in gmem —
   `attention/prefill/sm90/config.h:61` uses it for V.
4. `Sw<0,4,3>` is the identity: `INTER` means "no swizzle", 16 B rows.

### The physical chunk table (the core of swizzle)

`Swizzle<B,M,S>` acts on the **byte address** (`cute/pointer_swizzle.hpp:87`), and
XORs address bits `[4, 4+B)` with bits `[7, 7+B)`. The lab builds the real
addresses of a 1024-byte aligned buffer and checks a predictor for every
combination:

```text
 K_SW128<bf16>:  row = 128 B, bits[7,10) = row      physical chunk = c ^ row
 row | c0 c1 c2 c3 c4 c5 c6 c7
  0  |  0  1  2  3  4  5  6  7
  1  |  1  0  3  2  5  4  7  6
  2  |  2  3  0  1  6  7  4  5
  3  |  3  2  1  0  7  6  5  4
  4  |  4  5  6  7  0  1  2  3
  5  |  5  4  7  6  1  0  3  2
  6  |  6  7  4  5  2  3  0  1
  7  |  7  6  5  4  3  2  1  0

 K_SW64<bf16>:   row = 64 B,  bits[7,9) = row/2      chunk = c ^ ((row>>1) & 3)
  0  |  0  1  2  3      4  |  2  3  0  1
  1  |  0  1  2  3      5  |  2  3  0  1
  2  |  1  0  3  2      6  |  3  2  1  0
  3  |  1  0  3  2      7  |  3  2  1  0

 K_SW32<bf16>:   row = 32 B,  bit 7 = row/4          chunk = c ^ ((row>>2) & 1)
  0..3 | 0  1      4..7 | 1  0
```

The `fp8` table is identical to the `bf16` SW128 table: the swizzle works in
bytes, so the width in *bytes* is what decides, not the element type.

For each of the 549 checks the lab also proves two invariants:

* The swizzle **never moves data to another row** (`physical row == logical row`).
* The bytes **inside** a 16-byte chunk stay contiguous.

So swizzle is a pure permutation of 16-byte chunks inside each row.

### Why: bank conflicts

```text
 Without swizzle, a column access (K-major, 8 rows of one logical chunk):
   row r, chunk c  ->  byte address = r*128 + c*16      r = 0..7
   bank of the first word = (address / 4) mod 32 = (32r + 4c) mod 32 = 4c  <- SAME for all r
   ⇒ 8 requests hit the same 4 banks ⇒ 8-way conflict

 With SW128, chunk c of row r lands in physical chunk c^r:
   address = r*128 + (c^r)*16
   bank = (32r + 4(c^r)) mod 32 = 4(c^r)
   ⇒ c^0..c^7 = 8 distinct values ⇒ 8 distinct bank groups ⇒ no conflict
```

The lab measures this directly: for every logical chunk, it collects the 8
physical chunk positions over 8 rows and checks the mask is `0xFF` (all 8 groups
used).

### tile_to_shape — line by line

```text
 tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, (64,64,3))

 Sw<3,4,3> o ((_8,_8),(_64,_1),(_1,_3)):((_64,_512),(_1,_0),(_0,_4096))
              └──┬──┘  └──┬──┘  └──┬──┘   └───┬───┘  └─┬─┘  └─┬──┘
                 │        │        │          │        │      └ stage s: +4096 elements (8 KiB)
                 │        │        │          │        └ row mode: stride 1 (K-major)
                 │        │        │          └ atom-internal row stride 64
                 │        │        └ atom column mode, 64 columns
                 │        └ atom row mode: 8 rows, one atom = 8 rows
                 └ 8 atoms along M (64 / 8), stride 512 = 8 atoms × 512... see below
```

How to read it:

1. The atom covers `8 × 64` elements = one 8-row block.
2. 64 rows need 8 atoms along M: shape `_8` (atom index), stride `8 × 64 = 512`.
   In mode position 0 the first sub-mode (`_8`) is the row inside the atom
   (stride 64) and the second (`_8`) is the atom index (stride 512). `(9, 3, 2)` =
   `1*64 + 1*512 + 3 + 2*4096` — the lab checks this exactly.
3. 64 columns fit in one atom (the SW128 atom is 64 columns wide for bf16), so
   the inner column mode is `(_64,_1)` and the tiler adds a size-1 mode with
   stride 0.
4. The stage mode adds `_1` (size) with stride 0 (because `(_64,_1)`: the size-1
   mode has stride 0 in `tile_to_shape`'s appended tiler) and `_3` with stride 4096.
5. `cosize = 64*64*3 = 12288` elements = 24 KiB. Stage `s` starts at
   `4096 * 2 B = 8192 B`, a multiple of 1024, so **every stage has the same
   swizzle phase**. If the per-stage size were not a multiple of 1024 B, stage 0
   and stage 1 would swizzle differently and the TMA box would mismatch.

The second `tile_to_shape` (SW64, `64×64`) shows the default order fills M first:

```text
 Sw<2,4,3> o ((_8,_8),(_32,_2)):((_32,_256),(_1,_2048))
             8 atoms along M (stride 256 = 8*32), 2 atoms along K (stride 2048 = 8*256)
```

### Where this is used

| Code | What it does |
|---|---|
| `gemm.cu:431` | `tile_to_shape(K_SW128, (kTileM, kTileK, kStage))` for X |
| `gemm.cu:433` | `tile_to_shape(K_SW128, (kTileN, kTileK, kWarpGroupN, 2, kStage))` for W |
| `gemm.cu:436` | `tile_to_shape(MN_SW128, (kTileN, kTileM, kWarpGroupN))` for the epilogue smem Y |
| `gemm.cu:443` | `make_tma_copy(..., take<0,2>(slayout))` — the TMA box is the atom grid with the stage mode removed |

The WGMMA `make_fragment_A/B` reads the **same** layout and builds the
descriptor's swizzle bits from it. That is why the TMA box and the MMA descriptor
cannot disagree: they are two readings of one layout.

### Reproduce

```bash
make bin/lab15_swizzle && ./bin/lab15_swizzle
```

---

## 6. Lab 1.6 — retile_fragment and the accumulator owner map

File: `labs/lab16_retile.cu` → `bin/lab16_retile` (4163 checks)

### Part 1: retile_fragment

```text
 tCr               = ((_2,_2,_8),_1,_1):((_1,_2,_4),_0,_0)     32 registers/thread
 retile_fragment   = (_2,(_2,_8)):(_2,(_1,_4))
                     └┬┘ └──┬──┘  └┬┘ └──┬──┘
                      │     │      │     └ 8 column blocks of width 8 -> stride 4
                      │     │      └ inner column: 2 values, stride 1
                      │     └ 2 x 8 = 16 columns per thread
                      └ 2 rows per thread, register stride 2
```

`retile_fragment` reads the layout, keeps the "tile" mode (`take<1,R>`) as the
row mode, and flattens the "thread/value" mode (`flatten(select<0>)`) into the
column mode. The prediction in the lab (written from `utils.cuh:433`) matched.

This is what makes online softmax readable: instead of
`((_2,_2,_8),1,1)`, you get `(rows, cols)` and can write
`for (i) { rowmax over k }` — exactly as `attention/prefill/sm90/kernels.cuh`
does.

### Part 2: who owns which accumulator element

The lab partitions an identity tensor like C for all 128 threads and checks the
formula from `record.md` section 4.8 for **all 4096 elements**:

```text
 row = 16*w + l/4 + 8*i     w = warp 0..3, l = lane 0..31, i = 0,1
 col = 8*b + 2*(l%4) + j    b = 0..7, j = 0,1
 register = j + 2*i + 4*b
```

Measured owner map (C rows 0–9, cols 0–15):

```text
        c0  c1  c2  c3  c4  c5  c6  c7  c8  c9 c10 c11 c12 c13 c14 c15
 r0  |   0   0   1   1   2   2   3   3   0   0   1   1   2   2   3   3
 r1  |   4   4   5   5   6   6   7   7   4   4   5   5   6   6   7   7
 r2  |   8   8   9   9  10  10  11  11   8   8   9   9  10  10  11  11
 r3  |  12  12  13  13  14  14  15  15  12  12  13  13  14  14  15  15
 r4  |  16  16  17  17  18  18  19  19  16  16  17  17  18  18  19  19
 ...
 r8  |   0   0   1   1   2   2   3   3   0   0   1   1   2   2   3   3   ← i = 1: same lanes, +8 rows
 ...
 r16 |  32  32  33  33  34  34  35  35                               ← warp 1
```

Read the structure:

1. **A thread owns 2 columns per 8-column block** (`c0,c1` for thread 0), then
   reappears at `c8,c9`. Column stride 8 across blocks.
2. **A thread owns rows 8 apart** (`+8` from `i`), then jumps 16 at the warp
   boundary.
3. **Four consecutive lanes share a row**: `l, l^1, l^2, l^3` (same `l/4`). The
   lab checks this for all 64 rows. Hence a **row reduction is two `shfl_xor`
   steps with masks 1 and 2** — exactly `warp_4lane_reduce_max_xor`
   (`utils.cuh:398`) and `warp_4lane_reduce_sum_xor` (`utils.cuh:405`).
4. **A warp owns 16 rows** (2 × 8), so a 64-row tile is 4 warps = 1 warpgroup.

This is why flash attention in this repository reduces over 4 lanes and then
across warps through smem, and why `retile_fragment` gives each thread exactly 2
rows to softmax.

### Part 3: retile of the identity view

The lab also checks that after `retile_fragment`, each thread's 2 rows are
constant per row (`get<0>` equal for all 16 columns) and the 16 column indices
increase monotonically. That is the property the attention kernels rely on when
they do `tI(0, k)` to read "the column of my element" for the causal mask.

### Two predictions that were wrong

| Where | I predicted | Actual | Why |
|---|---|---|---|
| lab 1.1 `L10(7)` | 10 | **11** | 7 splits into `(7%6, 7/6) = (1,1)` → `10*1 + 1 = 11`. I wrote the comment "10 + 1" but the number 10. |
| lab 1.3 `ld` vs `zd` as 1-D functions | equal | **not equal** (`ld(8)=512`, `zd(8)=1`) | The four divides group the modes differently, and a 1-D index walks modes in order. Only the *coordinate* evaluation is equal. |

The second one is the useful lesson: **compare layouts by coordinates, not by
1-D index.** `CHECK_SAME_FN` (same function on 1-D indices) is the wrong check for
`ld` vs `zd`; the lab now checks coordinate equality over all 2048 elements
instead.

---

## 7. Reproduce everything

```bash
cd /root/hpc-ops/dive/practice/labs
make -j8            # 6 labs, ~10 s
make run            # run all, stop on first failure
```

Expected final lines:

```text
[lab11_eval] 42 passed, 0 failed
[lab12_algebra] 26 passed, 0 failed
[lab13_divide] 17 passed, 0 failed
[lab14_product] 84 passed, 0 failed
[lab15_swizzle] 549 passed, 0 failed
[lab16_retile] 4163 passed, 0 failed
```

---

## 8. Exit criteria for Stage 1

| Criterion | Status |
|---|---|
| Explain any layout's printed form | Pass — lab 1.1, 11 layouts |
| Compute coords, `size`, `cosize` by hand | Pass — L3 (`cosize 39`) and L7 (broadcast) |
| State what coalesce / composition / complement do | Pass — lab 1.2 |
| Produce the four divides from a tiler | Pass — lab 1.3 |
| Explain blocked vs raked product and the TV layout | Pass — lab 1.4 |
| Build the swizzle chunk table for SW32/64/128 and say why it removes conflicts | Pass — lab 1.5 |
| Read `tile_to_shape(atom, (M,N,Stage))` mode by mode | Pass — lab 1.5 |
| Explain `retile_fragment` and the accumulator owner map from memory | Pass — lab 1.6 |

The next step is Stage 2: the same concepts on the GPU with real tensors,
`make_tiled_copy`, predication and shared memory.
