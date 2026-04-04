#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void layernorm(float *in, float *gamma, float *beta,
                      float *out, float *tmp_mean, float *tmp_var);

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  const int ROWS = 4, COLS = 8;
  float in[32], gamma[8], beta[8], out[32];
  float tmp_mean[4], tmp_var[4];

  for (int i = 0; i < 32; i++)
    in[i] = (float)(i) * 0.1f - 1.5f;
  for (int j = 0; j < COLS; j++) {
    gamma[j] = 1.0f;
    beta[j] = 0.0f;
  }
  for (int i = 0; i < 32; i++) out[i] = 0.0f;
  for (int i = 0; i < 4; i++) { tmp_mean[i] = 0.0f; tmp_var[i] = 0.0f; }

  layernorm(in, gamma, beta, out, tmp_mean, tmp_var);

  // Verify: with gamma=1, beta=0, each row should have mean~0 and var~1
  int pass = 1;
  for (int r = 0; r < ROWS; r++) {
    float sum = 0.0f, sq_sum = 0.0f;
    for (int c = 0; c < COLS; c++) {
      sum += out[r * COLS + c];
      sq_sum += out[r * COLS + c] * out[r * COLS + c];
    }
    float mean = sum / COLS;
    float var = sq_sum / COLS - mean * mean;

    float mean_diff = mean < 0 ? -mean : mean;
    float var_diff = var - 1.0f;
    if (var_diff < 0) var_diff = -var_diff;

    if (mean_diff > 1e-3f || var_diff > 1e-2f) {
      pass = 0;
    }
  }

  // Also verify element-by-element against reference
  for (int r = 0; r < ROWS; r++) {
    float mean = 0.0f;
    for (int c = 0; c < COLS; c++) mean += in[r * COLS + c];
    mean /= COLS;
    float var = 0.0f;
    for (int c = 0; c < COLS; c++) {
      float d = in[r * COLS + c] - mean;
      var += d * d;
    }
    var /= COLS;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int c = 0; c < COLS; c++) {
      float ref = (in[r * COLS + c] - mean) * inv_std * gamma[c] + beta[c];
      float diff = out[r * COLS + c] - ref;
      if (diff < 0) diff = -diff;
      if (diff > 1e-4f) {
        pass = 0;
      }
    }
  }

  if (pass) {
    vx_printf("layernorm passed\n");
    return 0;
  } else {
    vx_printf("layernorm FAILED\n");
    return 1;
  }
}
