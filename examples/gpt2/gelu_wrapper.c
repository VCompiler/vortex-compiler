#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void gelu(float *in, float *out);

static float gelu_ref(float x) {
  return x * 0.5f * (1.0f + erff(x * 0.70710678118f));
}

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  float in[16], out[16];
  float test_vals[] = {-3.0f, -2.0f, -1.0f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f,
                       1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 0.3f, -0.3f, 0.7f};

  for (int i = 0; i < 16; i++) {
    in[i] = test_vals[i];
    out[i] = 0.0f;
  }

  gelu(in, out);

  int pass = 1;
  for (int i = 0; i < 16; i++) {
    float expected = gelu_ref(in[i]);
    float diff = out[i] - expected;
    if (diff < 0) diff = -diff;
    if (diff >= 1e-4f) {
      pass = 0;
    }
  }

  if (pass) {
    vx_printf("gelu passed\n");
    return 0;
  } else {
    vx_printf("gelu FAILED\n");
    return 1;
  }
}
