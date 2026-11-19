// Lab 1.4 — logical / blocked / raked product, and the thread-value (TV) layout that
// make_tiled_copy builds from them (src/utils/utils.cuh:705-717).
// thr = (4,4) column-major (16 threads), val = (2,2) column-major (4 values per thread).
#include "check.hpp"
using namespace cute;  // NOLINT

int main() {
  using _1 = Int<1>; using _2 = Int<2>; using _3 = Int<3>; using _4 = Int<4>;
  using _6 = Int<6>; using _8 = Int<8>; using _16 = Int<16>; using _32 = Int<32>;

  print("---- logical_product(A, B) = (A, compose(complement(A, size(A)*cosize(B)), B))\n");
  auto lp0 = logical_product(make_layout(_4{}, _1{}), make_layout(_3{}, _1{}));
  SHOW(lp0);
  CHECK_LAYOUT(lp0, make_layout(make_shape(_4{}, _3{}), make_stride(_1{}, _4{})));

  auto lpA = make_layout(make_shape(_2{}, _2{}), make_stride(_4{}, _1{}));
  auto lp1 = logical_product(lpA, make_layout(_6{}, _1{}));
  SHOW(lp1);
  CHECK_LAYOUT(lp1, make_layout(make_shape(make_shape(_2{}, _2{}), make_shape(_2{}, _3{})),
                                make_stride(make_stride(_4{}, _1{}), make_stride(_2{}, _8{}))));

  print("---- 2-D products: (M,N) coordinate -> thr_idx + 16 * val_idx\n");
  auto thr = make_layout(make_shape(_4{}, _4{}));  // (4,4):(1,4)
  auto val = make_layout(make_shape(_2{}, _2{}));  // (2,2):(1,2)

  // blocked: the first layout is the inner block -> each thread id appears once per 4x4 block
  auto bp = blocked_product(thr, val);
  SHOW(bp);
  CHECK_LAYOUT(bp, make_layout(make_shape(make_shape(_4{}, _2{}), make_shape(_4{}, _2{})),
                               make_stride(make_stride(_1{}, _16{}), make_stride(_4{}, _32{}))));
  print_layout(bp);

  // raked: the second layout is the inner block -> each thread owns a contiguous 2x2 block
  auto rp = raked_product(thr, val);
  SHOW(rp);
  CHECK_LAYOUT(rp, make_layout(make_shape(make_shape(_2{}, _4{}), make_shape(_2{}, _4{})),
                               make_stride(make_stride(_16{}, _1{}), make_stride(_32{}, _4{}))));
  print_layout(rp);

  // thread 5 = (1,1) in thr; its 4 values in rp sit at rows 2..3, cols 2..3
  CHECK_EQ(rp(2, 2), 5 + 16 * 0);
  CHECK_EQ(rp(3, 2), 5 + 16 * 1);
  CHECK_EQ(rp(2, 3), 5 + 16 * 2);
  CHECK_EQ(rp(3, 3), 5 + 16 * 3);
  // in bp the same thread is spread out by 4
  CHECK_EQ(bp(1, 1), 5);
  CHECK_EQ(bp(5, 1), 5 + 16);
  CHECK_EQ(bp(1, 5), 5 + 32);

  print("---- TV layout: (thr, val) -> offset of the (8,8) column-major tile\n");
  auto tv = right_inverse(rp).with_shape(make_shape(size(thr), size(val)));
  SHOW(tv);
  CHECK_EQ(size<0>(tv), 16);
  CHECK_EQ(size<1>(tv), 4);
  CHECK_EQ(tv(5, 0), 2 + 8 * 2);  // (2,2)
  CHECK_EQ(tv(5, 1), 3 + 8 * 2);  // (3,2)
  CHECK_EQ(tv(5, 2), 2 + 8 * 3);  // (2,3)
  CHECK_EQ(tv(5, 3), 3 + 8 * 3);  // (3,3)
  // tv is the inverse of rp on the whole tile
  for (int t = 0; t < 16; ++t)
    for (int v = 0; v < 4; ++v) CHECK_EQ(rp(tv(t, v)), t + 16 * v);

  // the same thing that cute::make_tiled_copy builds (8 bf16 per thread = one 128-bit copy)
  auto copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, bfloat16_t>{},
                              make_layout(make_shape(_16{}, _8{}), make_stride(_8{}, _1{})),
                              make_layout(make_shape(_1{}, _8{})));
  print("tiled_copy 16x8 threads x 1x8 values:\n");
  print(copy);
  print("\n");
  CHECK_EQ(size(copy), 128);                     // threads
  CHECK_EQ(size<0>(typename decltype(copy)::Tiler_MN{}), 16);  // tile rows
  CHECK_EQ(size<1>(typename decltype(copy)::Tiler_MN{}), 64);  // tile cols = 8 thr x 8 val

  return lab::report("lab14_product");
}
