#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_LOWERLINALGINSIDEVORTEXKERNEL
#include "vortex/Transforms/Passes.h.inc"

namespace {

static LogicalResult lowerLinalgOp(linalg::LinalgOp linalgOp,
                                   IRRewriter &rewriter) {
  if (!linalgOp.hasPureBufferSemantics()) {
    return linalgOp.emitOpError()
           << "requires buffer semantics before vortex-lower-linalg-inside-kernel";
  }

  rewriter.setInsertionPoint(linalgOp);
  if (failed(linalg::linalgOpToLoops(rewriter, linalgOp))) {
    return linalgOp.emitOpError()
           << "failed to lower into scf.for loop nests";
  }

  rewriter.eraseOp(linalgOp);
  return success();
}

struct LowerLinalgInsideVortexKernel
    : public impl::LowerLinalgInsideVortexKernelBase<
          LowerLinalgInsideVortexKernel> {
  using impl::LowerLinalgInsideVortexKernelBase<
      LowerLinalgInsideVortexKernel>::LowerLinalgInsideVortexKernelBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<linalg::LinalgOp> worklist;
    func.walk([&](linalg::LinalgOp linalgOp) { worklist.push_back(linalgOp); });

    // 先收集再逆序改写，避免新生成的循环结构干扰原始遍历顺序。
    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp linalgOp : llvm::reverse(worklist)) {
      if (!linalgOp || !linalgOp->getParentRegion())
        continue;
      if (failed(lowerLinalgOp(linalgOp, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
