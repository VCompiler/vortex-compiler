#include "vortex/InitAllDialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

namespace mlir::vortex {

void registerVortexDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, arith::ArithDialect,
                  bufferization::BufferizationDialect,
                  cf::ControlFlowDialect, func::FuncDialect,
                  linalg::LinalgDialect, math::MathDialect,
                  memref::MemRefDialect, scf::SCFDialect,
                  tensor::TensorDialect, vector::VectorDialect,
                  VortexDialect>();
}

} // namespace mlir::vortex
