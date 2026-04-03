#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void transformer_block(
    float *x_in, float *x_out,
    float *ln1_gamma, float *ln1_beta,
    float *wq, float *wk, float *wv, float *wo,
    float *ln2_gamma, float *ln2_beta,
    float *w1, float *w2,
    float *x_ln, float *q, float *k, float *v,
    float *score, float *prob, float *attn, float *attn_out,
    float *x_ln2, float *hidden,
    float *ln_mean, float *ln_var, float *sm_max, float *sm_sum);

#define S     4
#define D     8
#define D_FF  32
#define EPS   1e-5f

/* ---- reference helpers ---- */

static void layernorm_ref(const float *x, const float *gamma,
                          const float *beta, float *out,
                          int rows, int cols, float eps) {
  for (int r = 0; r < rows; r++) {
    float mean = 0.0f;
    for (int c = 0; c < cols; c++)
      mean += x[r * cols + c];
    mean /= cols;

    float var = 0.0f;
    for (int c = 0; c < cols; c++) {
      float d = x[r * cols + c] - mean;
      var += d * d;
    }
    var /= cols;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (int c = 0; c < cols; c++)
      out[r * cols + c] =
          (x[r * cols + c] - mean) * inv_std * gamma[c] + beta[c];
  }
}

static void matmul_ref(const float *a, const float *b, float *c,
                       int m, int n, int k_) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      float acc = 0.0f;
      for (int kk = 0; kk < k_; kk++)
        acc += a[i * k_ + kk] * b[kk * n + j];
      c[i * n + j] = acc;
    }
}

static float gelu_ref(float x) {
  return x * 0.5f * (1.0f + erff(x * 0.70710678118f));
}

static void softmax_ref(const float *in, float *out, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float mx = in[r * cols];
    for (int c = 1; c < cols; c++)
      if (in[r * cols + c] > mx)
        mx = in[r * cols + c];
    float s = 0.0f;
    for (int c = 0; c < cols; c++) {
      out[r * cols + c] = expf(in[r * cols + c] - mx);
      s += out[r * cols + c];
    }
    for (int c = 0; c < cols; c++)
      out[r * cols + c] /= s;
  }
}

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  /* ---- allocate all buffers on the stack ---- */
  float x_in[S * D], x_out[S * D];
  float ln1_gamma[D], ln1_beta[D];
  float wq[D * D], wk[D * D], wv[D * D], wo[D * D];
  float ln2_gamma[D], ln2_beta[D];
  float w1[D * D_FF], w2[D_FF * D];

  /* scratch */
  float x_ln[S * D];
  float q[S * D], k[S * D], v[S * D];
  float score[S * S], prob[S * S];
  float attn[S * D], attn_out[S * D];
  float x_ln2[S * D], hidden[S * D_FF];
  float ln_mean[S], ln_var[S];
  float sm_max[S], sm_sum[S];

  /* ---- deterministic initialisation ---- */
  for (int i = 0; i < S * D; i++)
    x_in[i] = (float)(i % 5) * 0.1f - 0.2f;

  for (int i = 0; i < D; i++) {
    ln1_gamma[i] = 1.0f;
    ln1_beta[i]  = 0.0f;
    ln2_gamma[i] = 1.0f;
    ln2_beta[i]  = 0.0f;
  }

  for (int i = 0; i < D * D; i++) {
    wq[i] = (float)(i % 7)  * 0.02f - 0.06f;
    wk[i] = (float)(i % 9)  * 0.02f - 0.08f;
    wv[i] = (float)(i % 11) * 0.02f - 0.10f;
    wo[i] = (float)(i % 13) * 0.02f - 0.12f;
  }

  for (int i = 0; i < D * D_FF; i++)
    w1[i] = (float)(i % 17) * 0.02f - 0.16f;

  for (int i = 0; i < D_FF * D; i++)
    w2[i] = (float)(i % 19) * 0.02f - 0.18f;

  /* zero all scratch buffers */
  for (int i = 0; i < S * D; i++) {
    x_ln[i] = 0.0f; q[i] = 0.0f; k[i] = 0.0f; v[i] = 0.0f;
    attn[i] = 0.0f; attn_out[i] = 0.0f; x_ln2[i] = 0.0f;
    x_out[i] = 0.0f;
  }
  for (int i = 0; i < S * S; i++) { score[i] = 0.0f; prob[i] = 0.0f; }
  for (int i = 0; i < S * D_FF; i++) hidden[i] = 0.0f;
  for (int i = 0; i < S; i++) {
    ln_mean[i] = 0.0f; ln_var[i] = 0.0f;
    sm_max[i]  = 0.0f; sm_sum[i] = 0.0f;
  }

  /* ---- call kernel ---- */
  transformer_block(
      x_in, x_out,
      ln1_gamma, ln1_beta,
      wq, wk, wv, wo,
      ln2_gamma, ln2_beta,
      w1, w2,
      x_ln, q, k, v,
      score, prob, attn, attn_out,
      x_ln2, hidden,
      ln_mean, ln_var, sm_max, sm_sum);

  /* ---- CPU reference ---- */

  /* 1. LayerNorm1 */
  float r_xln[S * D];
  layernorm_ref(x_in, ln1_gamma, ln1_beta, r_xln, S, D, EPS);

  /* 2. Q = x_ln @ Wq,  K = x_ln @ Wk,  V = x_ln @ Wv */
  float r_q[S * D], r_k[S * D], r_v[S * D];
  matmul_ref(r_xln, wq, r_q, S, D, D);
  matmul_ref(r_xln, wk, r_k, S, D, D);
  matmul_ref(r_xln, wv, r_v, S, D, D);

  /* 3. score = Q @ K^T / sqrt(D) */
  float r_score[S * S];
  for (int i = 0; i < S; i++)
    for (int j = 0; j < S; j++) {
      float acc = 0.0f;
      for (int kk = 0; kk < D; kk++)
        acc += r_q[i * D + kk] * r_k[j * D + kk]; /* K^T */
      r_score[i * S + j] = acc * (1.0f / sqrtf((float)D));
    }

  /* 4. prob = softmax(score) */
  float r_prob[S * S];
  softmax_ref(r_score, r_prob, S, S);

  /* 5. attn = prob @ V */
  float r_attn[S * D];
  matmul_ref(r_prob, r_v, r_attn, S, D, S);

  /* 6. attn_out = attn @ Wo */
  float r_attn_out[S * D];
  matmul_ref(r_attn, wo, r_attn_out, S, D, D);

  /* 7. residual1 = x_in + attn_out */
  float r_res1[S * D];
  for (int i = 0; i < S * D; i++)
    r_res1[i] = x_in[i] + r_attn_out[i];

  /* 8. LayerNorm2 */
  float r_xln2[S * D];
  layernorm_ref(r_res1, ln2_gamma, ln2_beta, r_xln2, S, D, EPS);

  /* 9. hidden = GELU(x_ln2 @ W1) */
  float r_hidden[S * D_FF];
  matmul_ref(r_xln2, w1, r_hidden, S, D_FF, D);
  for (int i = 0; i < S * D_FF; i++)
    r_hidden[i] = gelu_ref(r_hidden[i]);

  /* 10. ff_out = hidden @ W2 */
  float r_ff_out[S * D];
  matmul_ref(r_hidden, w2, r_ff_out, S, D, D_FF);

  /* 11. x_out = residual1 + ff_out */
  float r_xout[S * D];
  for (int i = 0; i < S * D; i++)
    r_xout[i] = r_res1[i] + r_ff_out[i];

  /* ---- compare ---- */
  int pass = 1;
  for (int i = 0; i < S * D; i++) {
    float diff = x_out[i] - r_xout[i];
    if (diff < 0) diff = -diff;
    if (diff > 1e-2f) {
      pass = 0;
    }
  }

  if (pass) {
    vx_printf("transformer_block passed\n");
    return 0;
  } else {
    vx_printf("transformer_block FAILED\n");
    return 1;
  }
}
