// Lab 1.5 — GMMA swizzle atoms, the physical-chunk table, and tile_to_shape mode by mode.
// The swizzle acts on the BYTE address (cute/pointer_swizzle.hpp:87), so the table is built
// from real addresses of a 1024-byte aligned buffer.
#include <cstdint>

#include "check.hpp"
using namespace cute;  // NOLINT

alignas(1024) static uint8_t g_smem[64 * 1024];

// For row r and logical 16-byte chunk c of a K-major atom: physical chunk and physical row.
template <class T, class Atom>
void chunk_table(const char *name, Atom atom, int (*predict)(int r, int c)) {
  auto t = make_tensor(make_smem_ptr(reinterpret_cast<T *>(g_smem)), atom);
  constexpr int kElemPerChunk = 16 / sizeof(T);
  const int cols = size<1>(atom);
  const int row_bytes = cols * sizeof(T);
  const int chunks = row_bytes / 16;
  printf("%s  (row = %d B, %d chunks)\n row |", name, row_bytes, chunks);
  for (int c = 0; c < chunks; ++c) printf(" c%d", c);
  printf("\n");
  for (int r = 0; r < 8; ++r) {
    printf("  %d  |", r);
    for (int c = 0; c < chunks; ++c) {
      auto off = reinterpret_cast<uint8_t *>(&t(r, c * kElemPerChunk)) - g_smem;
      int prow = int(off / row_bytes), pchunk = int(off % row_bytes) / 16;
      printf("  %d", pchunk);
      CHECK_EQ(prow, r);                // swizzle never moves data to another row
      CHECK_EQ(pchunk, predict(r, c));  // the XOR rule
      // the 8 (or 16) elements inside one chunk stay contiguous
      auto off_last = reinterpret_cast<uint8_t *>(&t(r, c * kElemPerChunk + kElemPerChunk - 1)) - g_smem;
      CHECK_EQ(off_last - off, 16 - sizeof(T));
    }
    printf("\n");
  }
}

int main() {
  using _1 = Int<1>; using _3 = Int<3>; using _8 = Int<8>; using _64 = Int<64>;
  using bf16 = bfloat16_t;
  using fp8 = float_e4m3_t;

  print("---- atoms: Sw<B,M,S> o smem_ptr o (rows, cols):(stride)\n");
  SHOW(GMMA::Layout_K_INTER_Atom<bf16>{});
  SHOW(GMMA::Layout_K_SW32_Atom<bf16>{});
  SHOW(GMMA::Layout_K_SW64_Atom<bf16>{});
  SHOW(GMMA::Layout_K_SW128_Atom<bf16>{});
  SHOW(GMMA::Layout_K_SW128_Atom<fp8>{});
  SHOW(GMMA::Layout_MN_SW128_Atom<bf16>{});
  SHOW(GMMA::Layout_MN_SW64_Atom<bf16>{});

  // prediction: one atom row is always 8 rows; its width in bytes = swizzle width
  CHECK_EQ(size<0>(GMMA::Layout_K_SW128_Atom<bf16>{}), 8);
  CHECK_EQ(size<1>(GMMA::Layout_K_SW128_Atom<bf16>{}), 64);   // 128 B / 2 B
  CHECK_EQ(size<1>(GMMA::Layout_K_SW64_Atom<bf16>{}), 32);    //  64 B / 2 B
  CHECK_EQ(size<1>(GMMA::Layout_K_SW32_Atom<bf16>{}), 16);    //  32 B / 2 B
  CHECK_EQ(size<1>(GMMA::Layout_K_INTER_Atom<bf16>{}), 8);    //  16 B / 2 B
  CHECK_EQ(size<1>(GMMA::Layout_K_SW128_Atom<fp8>{}), 128);   // 128 B / 1 B
  CHECK_EQ(size<0>(GMMA::Layout_MN_SW128_Atom<bf16>{}), 64);  // MN-major: contiguous along M
  CHECK_EQ(size<1>(GMMA::Layout_MN_SW128_Atom<bf16>{}), 8);

  print("---- physical 16-byte chunk for (row, logical chunk)\n");
  // Swizzle<B,4,3>: XOR address bits [4,4+B) with bits [7,7+B).
  // SW128: row = 128 B, bits[7,10) = r         -> c ^ r
  // SW64 : row =  64 B, bits[7,9)  = r/2 % 4   -> c ^ ((r>>1)&3)
  // SW32 : row =  32 B, bit 7      = r/4 % 2   -> c ^ ((r>>2)&1)
  chunk_table<bf16>("K_SW128<bf16>", GMMA::Layout_K_SW128_Atom<bf16>{},
                    [](int r, int c) { return c ^ r; });
  chunk_table<fp8>("K_SW128<fp8>", GMMA::Layout_K_SW128_Atom<fp8>{},
                   [](int r, int c) { return c ^ r; });
  chunk_table<bf16>("K_SW64<bf16>", GMMA::Layout_K_SW64_Atom<bf16>{},
                    [](int r, int c) { return c ^ ((r >> 1) & 3); });
  chunk_table<bf16>("K_SW32<bf16>", GMMA::Layout_K_SW32_Atom<bf16>{},
                    [](int r, int c) { return c ^ ((r >> 2) & 1); });

  print("---- bank check: one logical column (chunk c) over 8 rows\n");
  // SW128: 8 rows of one logical chunk -> 8 distinct physical chunks -> 8 distinct
  // 16 B bank groups (one 128 B row = all 32 banks).
  {
    auto t = make_tensor(make_smem_ptr(reinterpret_cast<bf16 *>(g_smem)),
                         GMMA::Layout_K_SW128_Atom<bf16>{});
    for (int c = 0; c < 8; ++c) {
      int mask = 0;
      for (int r = 0; r < 8; ++r) {
        auto off = reinterpret_cast<uint8_t *>(&t(r, c * 8)) - g_smem;
        mask |= 1 << int(off % 128 / 16);
      }
      CHECK_EQ(mask, 0xFF);
    }
  }

  print("---- tile_to_shape(K_SW128<bf16>, (64,64,3)) — the shape used for 3 stages of a 64x64 tile\n");
  auto sl = tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, make_shape(_64{}, _64{}, _3{}));
  SHOW(sl);
  auto inner = sl.layout_b();  // the layout before the swizzle
  SHOW(inner);
  CHECK_LAYOUT(inner, make_layout(make_shape(make_shape(_8{}, _8{}), make_shape(_64{}, _1{}),
                                             make_shape(_1{}, _3{})),
                                  make_stride(make_stride(_64{}, Int<512>{}),
                                              make_stride(_1{}, Int<0>{}),
                                              make_stride(Int<0>{}, Int<4096>{}))));
  CHECK_EQ(cosize(sl), 64 * 64 * 3);
  CHECK_EQ(inner(9, 3, 2), 1 * 64 + 1 * 512 + 3 + 2 * 4096);  // row 9 = (1, atom 1)
  // stage s starts 4096 elements = 8 KiB later, a multiple of 1024 B -> swizzle phase is the same
  CHECK_EQ(inner(0, 0, 1) * sizeof(bf16) % 1024, 0);

  // SW64 atom (8,32) -> 8 atoms along M x 2 atoms along K; default order fills M first,
  // so the next K atom starts after 8 atoms = 8*256 = 2048 elements
  auto sl2 = tile_to_shape(GMMA::Layout_K_SW64_Atom<bf16>{}, make_shape(_64{}, _64{}));
  SHOW(sl2);
  CHECK_LAYOUT(sl2.layout_b(),
               make_layout(make_shape(make_shape(_8{}, _8{}), make_shape(Int<32>{}, Int<2>{})),
                           make_stride(make_stride(Int<32>{}, Int<256>{}),
                                       make_stride(_1{}, Int<2048>{}))));
  return lab::report("lab15_swizzle");
}
