#ifndef VORTEX_RUNTIME_BOARD_XDMA_ABI_H
#define VORTEX_RUNTIME_BOARD_XDMA_ABI_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define VORTEX_BOARD_XDMA_NORETURN __attribute__((noreturn))
#else
#define VORTEX_BOARD_XDMA_NORETURN
#endif

#ifdef __cplusplus
extern "C" {
#endif

uintptr_t vortex_board_xdma_startup_arg(void);
int vortex_board_xdma_is_control_lane(void);
void vortex_board_xdma_host_visible_fence(void);
void vortex_board_xdma_progress(volatile int *progress, int value);
void vortex_board_xdma_exit(int status) VORTEX_BOARD_XDMA_NORETURN;

#ifdef __cplusplus
}
#endif

#endif // VORTEX_RUNTIME_BOARD_XDMA_ABI_H
