#include <vortex/Runtime/BoardXDMAABI.h>
#include <vx_print.h>

extern void matmul4x4(float *a, float *b, float *c);

static float g_a[16] __attribute__((aligned(64))) = {
    1.0f,  2.0f,  3.0f,  4.0f,
    5.0f,  6.0f,  7.0f,  8.0f,
    9.0f,  10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f,
};

static float g_b[16] __attribute__((aligned(64))) = {
    16.0f, 15.0f, 14.0f, 13.0f,
    12.0f, 11.0f, 10.0f, 9.0f,
    8.0f,  7.0f,  6.0f,  5.0f,
    4.0f,  3.0f,  2.0f,  1.0f,
};

static float g_expected[16] __attribute__((aligned(64))) = {
    80.0f,  70.0f,  60.0f,  50.0f,
    240.0f, 214.0f, 188.0f, 162.0f,
    400.0f, 358.0f, 316.0f, 274.0f,
    560.0f, 502.0f, 444.0f, 386.0f,
};

static float g_c[16] __attribute__((aligned(64)));

int main() {
  if (!vortex_board_xdma_is_control_lane())
    vortex_board_xdma_exit(0);

  for (int i = 0; i < 16; ++i) {
    g_c[i] = -1.0f;
  }

  matmul4x4(g_a, g_b, g_c);

  for (int i = 0; i < 16; ++i) {
    if (g_c[i] != g_expected[i]) {
      vx_printf("matmul4x4 mismatch idx=%d got=%d expect=%d\n",
                i, (int)g_c[i], (int)g_expected[i]);
      vortex_board_xdma_exit(i + 1);
    }
  }

  vx_printf("matmul4x4 board/XDMA smoke passed\n");
  vortex_board_xdma_exit(0);
}
