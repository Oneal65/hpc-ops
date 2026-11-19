// Lab 1.1 — Evaluate 10 layouts by hand, then let CuTe check the prediction.
// Rule: index = sum(coord_i * stride_i). A 1-D index into a multi-mode layout is first
// split colexicographically (left mode fastest): i -> (i % s0, i / s0).
#include "check.hpp"
using namespace cute;  // NOLINT

int main() {
  using _1 = Int<1>; using _2 = Int<2>; using _3 = Int<3>; using _4 = Int<4>;
  using _5 = Int<5>; using _6 = Int<6>; using _8 = Int<8>; using _12 = Int<12>;
  using _16 = Int<16>;

  // L1 row-major (4,8):(8,1)
  auto L1 = make_layout(make_shape(_4{}, _8{}), make_stride(_8{}, _1{}));
  SHOW(L1);
  CHECK_EQ(L1(1, 2), 10);
  CHECK_EQ(L1(3, 7), 31);
  CHECK_EQ(L1(2, 0), 16);
  CHECK_EQ(L1(10), 18);  // 10 -> (10%4, 10/4) = (2,2) -> 16+2
  CHECK_EQ(size(L1), 32);
  CHECK_EQ(cosize(L1), 32);

  // L2 default (column-major, LayoutLeft) (4,8):(1,4)
  auto L2 = make_layout(make_shape(_4{}, _8{}));
  SHOW(L2);
  CHECK_EQ(L2(1, 2), 9);
  CHECK_EQ(L2(3, 7), 31);
  CHECK_EQ(L2(2, 0), 2);
  CHECK_EQ(L2(10), 10);  // column-major: 1-D index == offset

  // L3 padded column-major (4,8):(1,5) — one gap element per column
  auto L3 = make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _5{}));
  SHOW(L3);
  CHECK_EQ(L3(1, 2), 11);
  CHECK_EQ(L3(3, 7), 38);
  CHECK_EQ(L3(0, 1), 5);
  CHECK_EQ(size(L3), 32);
  CHECK_EQ(cosize(L3), 39);  // last offset 38, +1

  // L4 1-D strided 8:2
  auto L4 = make_layout(_8{}, _2{});
  SHOW(L4);
  CHECK_EQ(L4(3), 6);
  CHECK_EQ(L4(7), 14);
  CHECK_EQ(cosize(L4), 15);

  // L5 hierarchical ((2,2),8):((1,16),2)
  auto L5 = make_layout(make_shape(make_shape(_2{}, _2{}), _8{}),
                        make_stride(make_stride(_1{}, _16{}), _2{}));
  SHOW(L5);
  CHECK_EQ(L5(3, 1), 19);  // 3 -> (1,1) -> 1+16, col 1 -> 2
  CHECK_EQ(L5(1, 7), 15);  // 1 -> (1,0) -> 1,    col 7 -> 14
  CHECK_EQ(L5(2, 3), 22);  // 2 -> (0,1) -> 16,   col 3 -> 6
  CHECK_EQ(L5(make_coord(make_coord(1, 1), 1)), 19);  // same element, natural coord

  // L6 hierarchical in the second mode (4,(2,4)):(2,(1,8))
  auto L6 = make_layout(make_shape(_4{}, make_shape(_2{}, _4{})),
                        make_stride(_2{}, make_stride(_1{}, _8{})));
  SHOW(L6);
  CHECK_EQ(L6(1, make_coord(1, 2)), 19);  // 2 + 1 + 16
  CHECK_EQ(L6(3, make_coord(0, 3)), 30);  // 6 + 24
  CHECK_EQ(L6(2, 5), 21);                 // col 5 -> (1,2) -> 17; row 2 -> 4
  CHECK_EQ(cosize(L6), 32);               // 6 + 1 + 24 + 1

  // L7 broadcast (stride 0) (4,8):(0,1) — every row reads the same 8 elements
  auto L7 = make_layout(make_shape(_4{}, _8{}), make_stride(Int<0>{}, _1{}));
  SHOW(L7);
  CHECK_EQ(L7(3, 5), 5);
  CHECK_EQ(L7(0, 7), 7);
  CHECK_EQ(size(L7), 32);
  CHECK_EQ(cosize(L7), 8);

  // L8 3-D row-major (2,3,4):(12,4,1)
  auto L8 = make_layout(make_shape(_2{}, _3{}, _4{}), make_stride(_12{}, _4{}, _1{}));
  SHOW(L8);
  CHECK_EQ(L8(1, 2, 3), 23);
  CHECK_EQ(L8(0, 1, 0), 4);
  CHECK_EQ(L8(1, 0, 2), 14);

  // L9 3-D default (column-major) (2,3,4):(1,2,6)
  auto L9 = make_layout(make_shape(_2{}, _3{}, _4{}));
  SHOW(L9);
  CHECK_EQ(stride<2>(L9), 6);
  CHECK_EQ(L9(1, 2, 3), 23);
  CHECK_EQ(L9(0, 1, 0), 2);
  CHECK_EQ(L9(1, 0, 2), 13);

  // L10 dynamic row-major (m,n):(n,1), m = 6, n = 10 (run-time values print without '_')
  int m = 6, n = 10;
  auto L10 = make_layout(make_shape(m, n), make_stride(n, _1{}));
  SHOW(L10);
  CHECK_EQ(L10(5, 9), 59);
  CHECK_EQ(L10(1, 0), 10);
  CHECK_EQ(L10(2, 3), 23);
  CHECK_EQ(L10(7), 11);  // 7 -> (7%6, 7/6) = (1,1) -> 10 + 1

  // LayoutRight helper gives row-major strides
  auto L11 = make_layout(make_shape(_2{}, _3{}, _4{}), LayoutRight{});
  SHOW(L11);
  CHECK_LAYOUT(L11, L8);

  // Print one table to see the function as a picture
  print_layout(L6);
  return lab::report("lab11_eval");
}
