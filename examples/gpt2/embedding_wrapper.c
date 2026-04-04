#include <vx_intrinsics.h>
#include <vx_print.h>

extern void embedding(int *token_ids, float *tok_table, float *pos_table, float *output);

#define SEQ 32
#define DIM 64
#define VOCAB 256

int main() {
  if (vx_thread_id() != 0 || vx_warp_id() != 0 || vx_core_id() != 0)
    return 0;

  int token_ids[SEQ];
  float tok_table[VOCAB * DIM];
  float pos_table[SEQ * DIM];
  float output[SEQ * DIM];

  // Deterministic init
  for (int i = 0; i < SEQ; i++)
    token_ids[i] = (i * 7 + 3) % VOCAB;  // pseudo-random token ids

  for (int i = 0; i < VOCAB * DIM; i++)
    tok_table[i] = (float)(i % 17) * 0.01f - 0.08f;

  for (int i = 0; i < SEQ * DIM; i++)
    pos_table[i] = (float)(i % 13) * 0.01f - 0.06f;

  for (int i = 0; i < SEQ * DIM; i++)
    output[i] = 0.0f;

  embedding(token_ids, tok_table, pos_table, output);

  // Reference
  int pass = 1;
  for (int i = 0; i < SEQ; i++) {
    int tid = token_ids[i];
    for (int j = 0; j < DIM; j++) {
      float expected = tok_table[tid * DIM + j] + pos_table[i * DIM + j];
      float diff = output[i * DIM + j] - expected;
      if (diff < 0) diff = -diff;
      if (diff > 1e-5f) {
        pass = 0;
      }
    }
  }

  if (pass) {
    vx_printf("embedding passed\n");
    return 0;
  } else {
    vx_printf("embedding FAILED\n");
    return 1;
  }
}
