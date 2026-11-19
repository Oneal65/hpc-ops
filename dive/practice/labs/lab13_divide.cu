// Lab 1.3 — logical / zipped / tiled / flat divide of a (32,64) row-major matrix by an (8,16) tile.
// Shape after the divide: mode "tile" = inside one tile, mode "rest" = which tile.
#include "check.hpp"
using namespace cute;  // NOLINT

int main() {
  using _1 = Int<1>; using _2 = Int<2>; using _3 = Int<3>; using _4 = Int<4>;
  using _8 = Int<8>; using _16 = Int<16>; using _24 = Int<24>; using _32 = Int<32>;
  using _64 = Int<64>; using _512 = Int<512>;

  print("---- 1-D: logical_divide(24:1, 4:2) = compose(A, (B, complement(B, 24)))\n");
  auto d1 = logical_divide(make_layout(_24{}, _1{}), make_layout(_4{}, _2{}));
  SHOW(d1);
  CHECK_LAYOUT(d1, make_layout(make_shape(_4{}, make_shape(_2{}, _3{})),
                               make_stride(_2{}, make_stride(_1{}, Int<8>{}))));

  print("---- 2-D: A = (32,64):(64,1), tiler = (8,16)\n");
  auto A = make_layout(make_shape(_32{}, _64{}), make_stride(_64{}, _1{}));
  auto tiler = make_tile(_8{}, _16{});

  auto ld = logical_divide(A, tiler);
  SHOW(ld);
  CHECK_LAYOUT(ld, make_layout(make_shape(make_shape(_8{}, _4{}), make_shape(_16{}, _4{})),
                               make_stride(make_stride(_64{}, _512{}), make_stride(_1{}, _16{}))));

  auto zd = zipped_divide(A, tiler);
  SHOW(zd);
  CHECK_LAYOUT(zd, make_layout(make_shape(make_shape(_8{}, _16{}), make_shape(_4{}, _4{})),
                               make_stride(make_stride(_64{}, _1{}), make_stride(_512{}, _16{}))));

  auto td = tiled_divide(A, tiler);
  SHOW(td);
  CHECK_LAYOUT(td, make_layout(make_shape(make_shape(_8{}, _16{}), _4{}, _4{}),
                               make_stride(make_stride(_64{}, _1{}), _512{}, _16{})));

  auto fd = flat_divide(A, tiler);
  SHOW(fd);
  CHECK_LAYOUT(fd, make_layout(make_shape(_8{}, _16{}, _4{}, _4{}),
                               make_stride(_64{}, _1{}, _512{}, _16{})));

  // Same element under the two groupings: ld((r,tm),(c,tn)) == zd((r,c),(tm,tn)).
  // (As 1-D functions they differ: the 1-D index walks the modes in a different order.)
  int same = 0;
  for (int tm = 0; tm < 4; ++tm)
    for (int tn = 0; tn < 4; ++tn)
      for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 16; ++c)
          same += ld(make_coord(make_coord(r, tm), make_coord(c, tn))) ==
                  zd(make_coord(make_coord(r, c), make_coord(tm, tn)));
  CHECK_EQ(same, 2048);
  CHECK_EQ(ld(1) == zd(1), true);    // index 1: ld -> row 1 (offset 64), zd -> row 1 too
  CHECK_EQ(ld(8) == zd(8), false);   // index 8: ld -> tile-row 1 (512), zd -> col 1 (1)
  CHECK_EQ(cosize(fd), 2048);

  print("---- local_tile == zipped_divide + slice of the rest mode\n");
  // tile (1,2) starts at row 8, col 32 -> 8*64 + 32 = 544
  CHECK_EQ(zd(make_coord(0, 0), make_coord(1, 2)), 544);
  CHECK_EQ(fd(0, 0, 1, 2), 544);
  CHECK_EQ(fd(3, 5, 1, 2), 544 + 3 * 64 + 5);

  static float buf[32 * 64];
  for (int i = 0; i < 32 * 64; ++i) buf[i] = float(i);
  auto T = make_tensor(&buf[0], A);
  auto t12 = local_tile(T, tiler, make_coord(1, 2));
  SHOW(t12.layout());
  CHECK_LAYOUT(t12.layout(), make_layout(make_shape(_8{}, _16{}), make_stride(_64{}, _1{})));
  CHECK_EQ(t12(0, 0), 544);
  CHECK_EQ(t12(7, 15), 544 + 7 * 64 + 15);

  // keep every tile: coord (_,_) -> rank-4 (8,16,4,4), same as flat_divide
  auto tall = local_tile(T, tiler, make_coord(_, _));
  SHOW(tall.layout());
  CHECK_LAYOUT(tall.layout(), fd);

  // local_tile with a (rows, cols, k) tiler and a Step that drops a mode (used in GEMMs)
  auto gA = local_tile(T, make_tile(_8{}, _16{}, Int<99>{}), make_coord(1, 2, _), Step<_1, _1, X>{});
  SHOW(gA.layout());
  CHECK_LAYOUT(gA.layout(), t12.layout());

  return lab::report("lab13_divide");
}
