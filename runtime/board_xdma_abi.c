#include "vortex/Runtime/BoardXDMAABI.h"

#include <VX_config.h>
#include <vx_intrinsics.h>

#ifndef VORTEX_BOARD_XDMA_STARTUP_ARG_CSR
#define VORTEX_BOARD_XDMA_STARTUP_ARG_CSR 0x340
#endif

#ifndef VORTEX_BOARD_XDMA_EXIT_ADDR
#ifdef IO_MPM_ADDR
#define VORTEX_BOARD_XDMA_EXIT_ADDR ((uintptr_t)(IO_MPM_ADDR + 8u))
#else
#define VORTEX_BOARD_XDMA_EXIT_ADDR ((uintptr_t)0x88u)
#endif
#endif

uintptr_t vortex_board_xdma_startup_arg(void) {
  return (uintptr_t)csr_read(VORTEX_BOARD_XDMA_STARTUP_ARG_CSR);
}

int vortex_board_xdma_is_control_lane(void) {
  return vx_core_id() == 0 && vx_warp_id() == 0 && vx_thread_id() == 0;
}

void vortex_board_xdma_host_visible_fence(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

void vortex_board_xdma_progress(volatile int *progress, int value) {
  if (!progress || !vortex_board_xdma_is_control_lane())
    return;
  *progress = value;
  vortex_board_xdma_host_visible_fence();
}

void vortex_board_xdma_exit(int status) {
  if (vortex_board_xdma_is_control_lane()) {
    vortex_board_xdma_host_visible_fence();
    *(volatile uint32_t *)VORTEX_BOARD_XDMA_EXIT_ADDR = (uint32_t)status;
    vortex_board_xdma_host_visible_fence();
  }

  vx_tmc_zero();
  for (;;) {
    __asm__ volatile("" ::: "memory");
  }
}
