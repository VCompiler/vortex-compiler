#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void lm_head(float *input, float *gamma, float *beta,
                    float *w_proj, float *logits,
                    float *ln_out, float *ln_mean, float *ln_var);

#define SEQ 32
#define DIM 64
#define VOCAB 256

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  float input[SEQ * DIM], gamma[DIM], beta[DIM];
  float w_proj[DIM * VOCAB], logits[SEQ * VOCAB];
  float ln_out[SEQ * DIM], ln_mean[SEQ], ln_var[SEQ];

  for (int i = 0; i < SEQ * DIM; i++) input[i] = (float)(i % 19) * 0.02f - 0.18f;
  for (int j = 0; j < DIM; j++) { gamma[j] = 1.0f; beta[j] = 0.0f; }
  for (int i = 0; i < DIM * VOCAB; i++) w_proj[i] = (float)(i % 23) * 0.01f - 0.11f;
  for (int i = 0; i < SEQ * VOCAB; i++) logits[i] = 0.0f;
  for (int i = 0; i < SEQ * DIM; i++) ln_out[i] = 0.0f;
  for (int i = 0; i < SEQ; i++) { ln_mean[i] = 0.0f; ln_var[i] = 0.0f; }

  lm_head(input, gamma, beta, w_proj, logits, ln_out, ln_mean, ln_var);

  // Reference: LayerNorm + matmul
  float r_ln[SEQ * DIM], r_logits[SEQ * VOCAB];
  for (int r = 0; r < SEQ; r++) {
    float mean = 0.0f;
    for (int c = 0; c < DIM; c++) mean += input[r * DIM + c];
    mean /= DIM;
    float var = 0.0f;
    for (int c = 0; c < DIM; c++) {
      float d = input[r * DIM + c] - mean;
      var += d * d;
    }
    var /= DIM;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int c = 0; c < DIM; c++)
      r_ln[r * DIM + c] = (input[r * DIM + c] - mean) * inv_std * gamma[c] + beta[c];
  }
  for (int i = 0; i < SEQ; i++)
    for (int j = 0; j < VOCAB; j++) {
      float acc = 0.0f;
      for (int k = 0; k < DIM; k++)
        acc += r_ln[i * DIM + k] * w_proj[k * VOCAB + j];
      r_logits[i * VOCAB + j] = acc;
    }

  int pass = 1;
  float max_diff = 0.0f;
  for (int i = 0; i < SEQ * VOCAB; i++) {
    float diff = logits[i] - r_logits[i];
    if (diff < 0) diff = -diff;
    if (diff > max_diff) max_diff = diff;
    if (diff > 5e-2f) pass = 0;
  }

  if (pass) {
    vx_printf("lm_head passed (max_diff=%d.%04d)\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
    return 0;
  } else {
    vx_printf("lm_head FAILED (max_diff=%d.%04d)\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
    return 1;
  }
}
