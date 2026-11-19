// Stage 0 starter lab (same program as record.md section 5.3).
#include <cstdio>

#include "cute/tensor.hpp"
using namespace cute;  // NOLINT

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
  auto gC = make_tensor(make_gmem_ptr((float *)nullptr), make_shape(Int<64>{}, Int<64>{}),
                        make_stride(Int<64>{}, Int<1>{}));
  auto tCr = thr.partition_fragment_C(gC);
  print("thread0 accumulator: "); print(tCr.layout()); print("\n");
  return 0;
}
