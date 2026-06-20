#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

extern void local_memory_coop_i32(int *out);

enum {
  kExpectedCores = 1,
  kExpectedWarps = 2,
  kExpectedThreads = 4,
  kTotalSlots = kExpectedWarps * kExpectedThreads
};

static int g_out[kTotalSlots] __attribute__((aligned(64)));

static int expected_peer_value(int wid, int tid) {
  int peer_wid = 1 - wid;
  return peer_wid * 100 + tid + 7;
}

static void local_memory_coop_i32_task(void *arg) {
  local_memory_coop_i32((int *)arg);
}

int main() {
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();
  int num_cores = vx_num_cores();
  int num_warps = vx_num_warps();
  int num_threads = vx_num_threads();

  if (cid != 0 || wid != 0 || tid != 0)
    return 0;

  if (num_cores != kExpectedCores || num_warps != kExpectedWarps ||
      num_threads != kExpectedThreads) {
    vx_printf("local_memory_coop_i32 config mismatch c=%d w=%d t=%d\n",
              num_cores, num_warps, num_threads);
    return 2;
  }

  for (int i = 0; i < kTotalSlots; ++i)
    g_out[i] = -1;

  uint32_t total_tasks = kTotalSlots;
  int rc = vx_spawn_threads(1, &total_tasks, 0,
                            (vx_kernel_func_cb)local_memory_coop_i32_task,
                            g_out);
  if (rc != 0)
    return 3;

  for (int w = 0; w < kExpectedWarps; ++w) {
    for (int t = 0; t < kExpectedThreads; ++t) {
      int slot = w * kExpectedThreads + t;
      int expected = expected_peer_value(w, t);
      if (g_out[slot] != expected) {
        vx_printf("local_memory_coop_i32 mismatch warp=%d tid=%d got=%d "
                  "expect=%d\n",
                  w, t, g_out[slot], expected);
        return 10 + slot;
      }
    }
  }

  vx_printf("local_memory_coop_i32 smoke passed\n");
  return 0;
}
