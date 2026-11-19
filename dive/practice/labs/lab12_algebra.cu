// Lab 1.2 — coalesce, composition, complement.
// Examples from the CuTe layout-algebra doc, plus functional checks.
#include "check.hpp"
using namespace cute;  // NOLINT

int main() {
  using _1 = Int<1>; using _2 = Int<2>; using _3 = Int<3>; using _4 = Int<4>;
  using _5 = Int<5>; using _6 = Int<6>; using _8 = Int<8>; using _12 = Int<12>;
  using _16 = Int<16>; using _20 = Int<20>; using _24 = Int<24>; using _32 = Int<32>;

  print("---- coalesce: merge mode i and i+1 when stride[i+1] == shape[i]*stride[i]\n");
  auto c0 = make_layout(make_shape(_2{}, make_shape(_1{}, _6{})),
                        make_stride(_1{}, make_stride(_6{}, _2{})));
  SHOW(c0);
  SHOW(coalesce(c0));
  CHECK_LAYOUT(coalesce(c0), make_layout(_12{}, _1{}));  // size-1 mode drops, 2:1 + 6:2 -> 12:1
  CHECK_SAME_FN(coalesce(c0), c0);

  auto rm = make_layout(make_shape(_4{}, _8{}), make_stride(_8{}, _1{}));
  SHOW(coalesce(rm));
  CHECK_LAYOUT(coalesce(rm), rm);  // 1 != 4*8: cannot merge

  auto cm = make_layout(make_shape(_4{}, _8{}));
  SHOW(coalesce(cm));
  CHECK_LAYOUT(coalesce(cm), make_layout(_32{}, _1{}));

  // by-mode coalesce keeps the rank (used in retile_fragment, utils.cuh:433)
  auto bm = make_layout(make_shape(make_shape(_2{}, _4{}), make_shape(_3{}, _5{})),
                        make_stride(make_stride(_1{}, _2{}), make_stride(_8{}, _24{})));
  SHOW(coalesce(bm, Step<_1, _1>{}));
  CHECK_LAYOUT(coalesce(bm, Step<_1, _1>{}),
               make_layout(make_shape(_8{}, Int<15>{}), make_stride(_1{}, _8{})));

  print("---- composition: R(i) = A(B(i))\n");
  auto A1 = make_layout(make_shape(_6{}, _2{}), make_stride(_8{}, _2{}));
  auto B1 = make_layout(make_shape(_4{}, _3{}), make_stride(_3{}, _1{}));
  auto R1 = composition(A1, B1);
  SHOW(R1);
  CHECK_LAYOUT(R1, make_layout(make_shape(make_shape(_2{}, _2{}), _3{}),
                               make_stride(make_stride(Int<24>{}, _2{}), _8{})));
  for (int i = 0; i < size(B1); ++i) CHECK_EQ(R1(i), A1(B1(i)));

  auto A2 = make_layout(_20{}, _2{});
  auto B2 = make_layout(make_shape(_5{}, _4{}), make_stride(_4{}, _1{}));
  auto R2 = composition(A2, B2);
  SHOW(R2);
  CHECK_LAYOUT(R2, make_layout(make_shape(_5{}, _4{}), make_stride(_8{}, _2{})));

  // composition with a tiler: take a (3,4) sub-block of a (6,8) row-major layout
  auto big = make_layout(make_shape(_6{}, _8{}), make_stride(_8{}, _1{}));
  auto sub = composition(big, make_tile(_3{}, _4{}));
  SHOW(sub);
  CHECK_LAYOUT(sub, make_layout(make_shape(_3{}, _4{}), make_stride(_8{}, _1{})));

  print("---- complement(A, M): the layout that fills the holes of A up to M\n");
  auto k1 = complement(make_layout(_4{}, _1{}), _24{});
  SHOW(k1);
  CHECK_LAYOUT(k1, make_layout(_6{}, _4{}));

  auto k2 = complement(make_layout(_6{}, _4{}), _24{});
  SHOW(k2);
  CHECK_LAYOUT(k2, make_layout(_4{}, _1{}));

  auto a3 = make_layout(_4{}, _2{});
  auto k3 = complement(a3, _24{});
  SHOW(k3);
  CHECK_LAYOUT(k3, make_layout(make_shape(_2{}, _3{}), make_stride(_1{}, _8{})));

  auto a4 = make_layout(make_shape(_2{}, _2{}), make_stride(_1{}, _6{}));
  auto k4 = complement(a4, _24{});
  SHOW(k4);
  CHECK_LAYOUT(k4, make_layout(make_shape(_3{}, _2{}), make_stride(_2{}, _12{})));

  // Property: (A, complement(A)) is a bijection onto [0, 24)
  auto full = make_layout(a4, k4);
  bool seen[24] = {};
  for (int i = 0; i < 24; ++i) seen[full(i)] = true;
  int covered = 0;
  for (bool b : seen) covered += b;
  CHECK_EQ(size(full), 24);
  CHECK_EQ(covered, 24);

  return lab::report("lab12_algebra");
}
