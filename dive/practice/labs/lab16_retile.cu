// Lab 1.6 — retile_fragment (src/utils/utils.cuh:433) and the WGMMA accumulator mapping.
// Also checks the formula in record.md section 4.8 for all 128 threads.
#include "check.hpp"
#include "src/utils/utils.cuh"
using namespace cute;  // NOLINT

int main() {
  using _64 = Int<64>;
  auto mma = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto gC = make_tensor(make_gmem_ptr((float *)nullptr), make_shape(_64{}, _64{}),
                        make_stride(_64{}, Int<1>{}));

  auto tCr = mma.get_slice(0).partition_fragment_C(gC);
  SHOW(tCr.layout());
  CHECK_EQ(size(tCr), 32);  // 64*64 / 128 threads

  auto r = hpc::retile_fragment(tCr);
  SHOW(r.layout());
  // Prediction: rows = the 2 "i" values (register stride 2); cols = (j,b) with strides (1,4)
  CHECK_LAYOUT(r.layout(), make_layout(make_shape(Int<2>{}, make_shape(Int<2>{}, Int<8>{})),
                                       make_stride(Int<2>{}, make_stride(Int<1>{}, Int<4>{}))));

  print("---- accumulator owner map: identity tensor partitioned like C\n");
  auto cI = make_identity_tensor(make_shape(_64{}, _64{}));
  int bad = 0;
  int owner[64][64];
  for (int t = 0; t < 128; ++t) {
    auto tI = mma.get_slice(t).partition_C(cI);  // ((2,2,8),1,1) of (row, col)
    int w = t / 32, l = t % 32;
    for (int b = 0; b < 8; ++b)
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
          int v = j + 2 * i + 4 * b;  // register index
          auto rc = tI(v);
          int row = get<0>(rc), col = get<1>(rc);
          owner[row][col] = t;
          bad += row != 16 * w + l / 4 + 8 * i;
          bad += col != 8 * b + 2 * (l % 4) + j;
        }
    // retile_fragment of the identity view: one row index per "i", 16 increasing columns
    auto rI = hpc::retile_fragment(tI);
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 16; ++k) {
        bad += get<0>(rI(i, k)) != 16 * w + l / 4 + 8 * i;
        if (k > 0) bad += get<1>(rI(i, k)) <= get<1>(rI(i, k - 1));
      }
  }
  CHECK_EQ(bad, 0);

  // The 4 lanes that share a row: l, l^1, l^2, l^3 (same l/4) -> 2 shfl_xor steps (1, 2)
  for (int row = 0; row < 64; ++row) {
    int lanes_mask = 0;
    for (int col = 0; col < 64; ++col) lanes_mask |= 1 << (owner[row][col] % 4);
    CHECK_EQ(lanes_mask, 0xF);
    for (int col = 0; col < 64; ++col) CHECK_EQ(owner[row][col] / 4, owner[row][0] / 4);
  }

  print("owner thread of C rows 0..9, cols 0..15:\n");
  for (int row = 0; row < 10; ++row) {
    for (int col = 0; col < 16; ++col) printf("%4d", owner[row][col]);
    printf("\n");
  }
  print("row 16 (warp 1): ");
  for (int col = 0; col < 8; ++col) printf("%4d", owner[16][col]);
  printf("\n");
  return lab::report("lab16_retile");
}
