#include <vx_intrinsics.h>
#include <vx_print.h>

extern void parallel_matmul(float *a, float *b, float *c);

#define M 4
#define K 8
#define N 8

// Global arrays — initialized at load time, visible to all cores
static float a[M * K] = {
  // (i % 7) * 0.1 + 0.1 for i in 0..31
  0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.1f,
  0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.1f, 0.2f,
  0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.1f, 0.2f, 0.3f,
  0.4f, 0.5f, 0.6f, 0.7f, 0.1f, 0.2f, 0.3f, 0.4f
};
static float b[K * N] = {
  // (i % 5) * 0.1 - 0.2 for i in 0..63
  -0.2f, -0.1f, 0.0f, 0.1f, 0.2f, -0.2f, -0.1f, 0.0f,
   0.1f,  0.2f,-0.2f,-0.1f, 0.0f,  0.1f,  0.2f,-0.2f,
  -0.1f,  0.0f, 0.1f, 0.2f,-0.2f, -0.1f,  0.0f, 0.1f,
   0.2f, -0.2f,-0.1f, 0.0f, 0.1f,  0.2f, -0.2f,-0.1f,
   0.0f,  0.1f, 0.2f,-0.2f,-0.1f,  0.0f,  0.1f, 0.2f,
  -0.2f, -0.1f, 0.0f, 0.1f, 0.2f, -0.2f, -0.1f, 0.0f,
   0.1f,  0.2f,-0.2f,-0.1f, 0.0f,  0.1f,  0.2f,-0.2f,
  -0.1f,  0.0f, 0.1f, 0.2f,-0.2f, -0.1f,  0.0f, 0.1f
};
static float c[M * N];

int main() {
  // All cores call the kernel — launch maps core_id to row index
  // Data is in .data segment, visible to all cores from the start
  parallel_matmul(a, b, c);

  // Only core 0, warp 0, thread 0 checks results
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  float ref[M * N];
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      ref[i * N + j] = 0.0f;
      for (int k = 0; k < K; k++)
        ref[i * N + j] += a[i * K + k] * b[k * N + j];
    }

  int pass = 1;
  float max_diff = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float diff = c[i] - ref[i];
    if (diff < 0) diff = -diff;
    if (diff > max_diff) max_diff = diff;
    if (diff > 1e-4f) pass = 0;
  }

  if (pass) {
    vx_printf("parallel_matmul passed (max_diff=%d.%04d)\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
  } else {
    vx_printf("parallel_matmul FAILED (max_diff=%d.%04d)\n",
              (int)max_diff, (int)((max_diff - (int)max_diff) * 10000));
  }

  return 0;
}
