#include <vx_intrinsics.h>
#include <vx_print.h>
#include <math.h>

extern void attention(
    float *x,
    float *wq, float *wk, float *wv, float *wo,
    float *q, float *k, float *v,
    float *score, float *prob,
    float *attn, float *out,
    float *sm_max, float *sm_sum);

#define S 4
#define D 8

static void matmul_ref(float *a, float *b, float *c, int m, int n, int k_) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      c[i * n + j] = 0.0f;
      for (int kk = 0; kk < k_; kk++)
        c[i * n + j] += a[i * k_ + kk] * b[kk * n + j];
    }
}

static void softmax_ref(float *in, float *out, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float mx = in[r * cols];
    for (int c = 1; c < cols; c++)
      if (in[r * cols + c] > mx) mx = in[r * cols + c];
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

  float x[S * D];
  float wq[D * D], wk[D * D], wv[D * D], wo[D * D];
  float q[S * D], k[S * D], v[S * D];
  float score[S * S], prob[S * S];
  float attn[S * D], out[S * D];
  float sm_max[S], sm_sum[S];

  // Init
  for (int i = 0; i < S * D; i++) x[i] = (float)(i % 5) * 0.1f - 0.2f;
  for (int i = 0; i < D * D; i++) {
    wq[i] = (float)(i % 7) * 0.02f - 0.06f;
    wk[i] = (float)(i % 9) * 0.02f - 0.08f;
    wv[i] = (float)(i % 11) * 0.02f - 0.1f;
    wo[i] = (float)(i % 13) * 0.02f - 0.12f;
  }
  for (int i = 0; i < S * D; i++) { q[i]=0; k[i]=0; v[i]=0; attn[i]=0; out[i]=0; }
  for (int i = 0; i < S * S; i++) { score[i]=0; prob[i]=0; }
  for (int i = 0; i < S; i++) { sm_max[i]=0; sm_sum[i]=0; }

  attention(x, wq, wk, wv, wo, q, k, v, score, prob, attn, out, sm_max, sm_sum);

  // Reference
  float rq[S*D], rk[S*D], rv[S*D], rscore[S*S], rprob[S*S], rattn[S*D], rout[S*D];
  matmul_ref(x, wq, rq, S, D, D);
  matmul_ref(x, wk, rk, S, D, D);
  matmul_ref(x, wv, rv, S, D, D);

  // score = Q @ K^T / sqrt(d)
  for (int i = 0; i < S; i++)
    for (int j = 0; j < S; j++) {
      float acc = 0.0f;
      for (int kk = 0; kk < D; kk++)
        acc += rq[i * D + kk] * rk[j * D + kk]; // K^T
      rscore[i * S + j] = acc * (1.0f / sqrtf((float)D));
    }

  softmax_ref(rscore, rprob, S, S);
  matmul_ref(rprob, rv, rattn, S, D, S);
  matmul_ref(rattn, wo, rout, S, D, D);

  int pass = 1;
  for (int i = 0; i < S * D; i++) {
    float diff = out[i] - rout[i];
    if (diff < 0) diff = -diff;
    if (diff > 1e-3f) {
      pass = 0;
    }
  }

  if (pass) {
    vx_printf("attention passed\n");
    return 0;
  } else {
    vx_printf("attention FAILED\n");
    return 1;
  }
}
