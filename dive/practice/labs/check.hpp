// Small self-check helpers for the CuTe labs.
// Each lab writes the predicted value first, then CHECK_* compares it with CuTe.
#pragma once
#include <cstdio>

#include "cute/tensor.hpp"

namespace lab {

inline int g_pass = 0;
inline int g_fail = 0;

inline void check_eq(long long got, long long want, const char *expr, const char *file, int line) {
  if (got == want) {
    ++g_pass;
  } else {
    ++g_fail;
    printf("  FAIL %s:%d  %s = %lld, predicted %lld\n", file, line, expr, got, want);
  }
}

// Same shape and same stride (structural equality).
template <class A, class B>
void check_layout(A const &got, B const &want, const char *expr, const char *file, int line) {
  if (cute::rank(got) == cute::rank(want) && bool(got == want)) {
    ++g_pass;
  } else {
    ++g_fail;
    printf("  FAIL %s:%d  %s\n    got       ", file, line, expr);
    cute::print(got);
    printf("\n    predicted ");
    cute::print(want);
    printf("\n");
  }
}

// Same function: same size and A(i) == B(i) for every 1-D index i.
template <class A, class B>
void check_same_fn(A const &a, B const &b, const char *expr, const char *file, int line) {
  bool ok = int(cute::size(a)) == int(cute::size(b));
  for (int i = 0; ok && i < int(cute::size(a)); ++i) ok = int(a(i)) == int(b(i));
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    printf("  FAIL %s:%d  %s are different functions\n", file, line, expr);
  }
}

inline int report(const char *name) {
  printf("[%s] %d passed, %d failed\n", name, g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}

}  // namespace lab

#define CHECK_EQ(got, want) lab::check_eq((long long)(got), (long long)(want), #got, __FILE__, __LINE__)
#define CHECK_LAYOUT(got, want) lab::check_layout((got), (want), #got, __FILE__, __LINE__)
#define CHECK_SAME_FN(a, b) lab::check_same_fn((a), (b), #a " vs " #b, __FILE__, __LINE__)

#define SHOW(x)              \
  do {                       \
    cute::print(#x " = ");   \
    cute::print(x);          \
    cute::print("\n");       \
  } while (0)
