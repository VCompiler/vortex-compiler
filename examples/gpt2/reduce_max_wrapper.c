#include <vx_intrinsics.h>
#include <vx_print.h>

extern void reduce_max(float *in, float *out);

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  float in[16];
  float out[1] = {0.0f};

  for (int i = 0; i < 16; i++) {
    in[i] = (float)(i + 1) * ((i % 2 == 0) ? 1.0f : -1.0f);
    // -2, 1, -4, 3, ... pattern => values: 1,-2,3,-4,...,15,-16
  }
  // Correction: just use simple ascending so max = 16.0
  for (int i = 0; i < 16; i++) {
    in[i] = (float)(i + 1);
  }

  reduce_max(in, out);

  float expected = 16.0f;
  float diff = out[0] - expected;
  if (diff < 0) diff = -diff;

  if (diff < 1e-4f) {
    vx_printf("reduce_max passed (got %d expected %d)\n",
              (int)out[0], (int)expected);
    return 0;
  } else {
    vx_printf("reduce_max FAILED (got %d expected %d)\n",
              (int)out[0], (int)expected);
    return 1;
  }
}
