#include <vx_intrinsics.h>
#include <vx_print.h>

// Kernel prototype: bare-ptr calling convention
extern void reduce_sum(float *in, float *out);

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  float in[16];
  float out[1] = {0.0f};
  float expected = 0.0f;

  for (int i = 0; i < 16; i++) {
    in[i] = (float)(i + 1);  // 1, 2, 3, ..., 16
    expected += in[i];        // 136.0
  }

  reduce_sum(in, out);

  float diff = out[0] - expected;
  if (diff < 0) diff = -diff;

  if (diff < 1e-4f) {
    vx_printf("reduce_sum passed (got %d expected %d)\n",
              (int)out[0], (int)expected);
    return 0;
  } else {
    vx_printf("reduce_sum FAILED (got %d expected %d)\n",
              (int)out[0], (int)expected);
    return 1;
  }
}
