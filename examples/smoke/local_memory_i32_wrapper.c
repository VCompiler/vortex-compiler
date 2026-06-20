#include <vx_intrinsics.h>
#include <vx_print.h>

extern void local_memory_i32(int *out);

static int g_out[4] __attribute__((aligned(64)));
static const int g_expected[4] = {7, 11, 18, 7};

int main() {
  if (vx_core_id() != 0 || vx_warp_id() != 0 || vx_thread_id() != 0)
    return 0;

  for (int i = 0; i < 4; ++i)
    g_out[i] = -1;

  local_memory_i32(g_out);

  for (int i = 0; i < 4; ++i) {
    if (g_out[i] != g_expected[i]) {
      vx_printf("local_memory_i32 mismatch idx=%d got=%d expect=%d\n",
                i, g_out[i], g_expected[i]);
      return i + 1;
    }
  }

  vx_printf("local_memory_i32 smoke passed\n");
  return 0;
}
