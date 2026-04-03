#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void mlp_block(float *in, float *w1, float *w2, float *hidden, float *out);

static float gelu_ref(float x) {
  return x * 0.5f * (1.0f + erff(x * 0.70710678118f));
}

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  const int S = 4, D = 8, H = 32;
  float in[S * D], w1[D * H], w2[H * D], hidden[S * H], out[S * D];

  // Simple deterministic init
  for (int i = 0; i < S * D; i++) in[i] = (float)(i % 7) * 0.1f - 0.3f;
  for (int i = 0; i < D * H; i++) w1[i] = (float)(i % 11) * 0.02f - 0.1f;
  for (int i = 0; i < H * D; i++) w2[i] = (float)(i % 13) * 0.02f - 0.12f;
  for (int i = 0; i < S * H; i++) hidden[i] = 0.0f;
  for (int i = 0; i < S * D; i++) out[i] = 0.0f;

  mlp_block(in, w1, w2, hidden, out);

  // Reference: matmul -> gelu -> matmul
  float ref_hidden[S * H], ref_out[S * D];
  for (int i = 0; i < S; i++) {
    for (int j = 0; j < H; j++) {
      float acc = 0.0f;
      for (int k = 0; k < D; k++)
        acc += in[i * D + k] * w1[k * H + j];
      ref_hidden[i * H + j] = gelu_ref(acc);
    }
  }
  for (int i = 0; i < S; i++) {
    for (int j = 0; j < D; j++) {
      float acc = 0.0f;
      for (int k = 0; k < H; k++)
        acc += ref_hidden[i * H + k] * w2[k * D + j];
      ref_out[i * D + j] = acc;
    }
  }

  int pass = 1;
  for (int i = 0; i < S * D; i++) {
    float diff = out[i] - ref_out[i];
    if (diff < 0) diff = -diff;
    if (diff > 1e-3f) {
      pass = 0;
    }
  }

  if (pass) {
    vx_printf("mlp_block passed\n");
    return 0;
  } else {
    vx_printf("mlp_block FAILED\n");
    return 1;
  }
}
