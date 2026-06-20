#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

extern void local_memory_multi_i32(int *out);

enum {
  kExpectedCores = 2,
  kExpectedWarps = 2,
  kExpectedThreads = 4,
  kSlotsPerCore = kExpectedWarps * kExpectedThreads,
  kTotalSlots = kExpectedCores * kSlotsPerCore
};

static int g_out[kTotalSlots] __attribute__((aligned(64)));

static int expected_value(int cid, int wid, int tid) {
  return cid * 1000 + wid * 100 + tid + 7;
}

static void local_memory_multi_i32_task(void *arg) {
  local_memory_multi_i32((int *)arg);
}

int main() {
  int cid = vx_core_id();
  int wid = vx_warp_id();
  int tid = vx_thread_id();
  int num_cores = vx_num_cores();
  int num_warps = vx_num_warps();
  int num_threads = vx_num_threads();

  if (wid != 0 || tid != 0)
    return 0;

  if (num_cores != kExpectedCores || num_warps != kExpectedWarps ||
      num_threads != kExpectedThreads) {
    if (cid == 0) {
      vx_printf("local_memory_multi_i32 config mismatch c=%d w=%d t=%d\n",
                num_cores, num_warps, num_threads);
    }
    return 2;
  }

  int core_base = cid * kSlotsPerCore;
  for (int i = 0; i < kSlotsPerCore; ++i)
    g_out[core_base + i] = -1;

  uint32_t total_tasks = kTotalSlots;
  int rc = vx_spawn_threads(1, &total_tasks, 0,
                            (vx_kernel_func_cb)local_memory_multi_i32_task,
                            g_out);
  if (rc != 0)
    return 3;

  for (int w = 0; w < kExpectedWarps; ++w) {
    for (int t = 0; t < kExpectedThreads; ++t) {
      int slot = core_base + w * kExpectedThreads + t;
      int expected = expected_value(cid, w, t);
      if (g_out[slot] != expected) {
        vx_printf("local_memory_multi_i32 mismatch core=%d warp=%d tid=%d "
                  "got=%d expect=%d\n",
                  cid, w, t, g_out[slot], expected);
        return 10 + slot;
      }
    }
  }

  if (cid == 0)
    vx_printf("local_memory_multi_i32 smoke passed\n");
  return 0;
}
