#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void softmax(float *in, float *out, float *tmp_max, float *tmp_sum);

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  // 4 rows x 8 cols
  float in[32], out[32];
  float tmp_max[4], tmp_sum[4];

  // Fill with test data
  for (int i = 0; i < 32; i++) {
    in[i] = (float)(i % 8) * 0.5f - 1.5f;  // range [-1.5, 2.0]
    out[i] = 0.0f;
  }

  softmax(in, out, tmp_max, tmp_sum);

  // Verify: each row should sum to ~1.0 and all values in [0, 1]
  int pass = 1;
  for (int r = 0; r < 4; r++) {
    float row_sum = 0.0f;
    for (int c = 0; c < 8; c++) {
      float v = out[r * 8 + c];
      if (v < -1e-6f || v > 1.0f + 1e-6f) {
        pass = 0;
      }
      row_sum += v;
    }
    float diff = row_sum - 1.0f;
    if (diff < 0) diff = -diff;
    if (diff > 1e-4f) {
      pass = 0;
    }
  }

  // Also verify against reference softmax
  for (int r = 0; r < 4; r++) {
    float mx = in[r * 8];
    for (int c = 1; c < 8; c++) {
      if (in[r * 8 + c] > mx) mx = in[r * 8 + c];
    }
    float s = 0.0f;
    float ref[8];
    for (int c = 0; c < 8; c++) {
      ref[c] = expf(in[r * 8 + c] - mx);
      s += ref[c];
    }
    for (int c = 0; c < 8; c++) {
      ref[c] /= s;
      float diff = out[r * 8 + c] - ref[c];
      if (diff < 0) diff = -diff;
      if (diff > 1e-4f) {
        pass = 0;
      }
    }
  }

  if (pass) {
    vx_printf("softmax passed\n");
    return 0;
  } else {
    vx_printf("softmax FAILED\n");
    return 1;
  }
}
