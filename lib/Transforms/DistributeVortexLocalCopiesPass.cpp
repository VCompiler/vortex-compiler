#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_DISTRIBUTEVORTEXLOCALCOPIES
#include "vortex/Transforms/Passes.h.inc"

namespace {

static bool hasAddressSpace(Value value, AddressSpace expected) {
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return false;

  auto addressSpace =
      dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace());
  return addressSpace && addressSpace.getValue() == expected;
}

static bool isSupportedLocalGlobalCopy(memref::CopyOp copy) {
  bool sourceGlobal = hasAddressSpace(copy.getSource(), AddressSpace::Global);
  bool sourceLocal = hasAddressSpace(copy.getSource(), AddressSpace::Local);
  bool targetGlobal = hasAddressSpace(copy.getTarget(), AddressSpace::Global);
  bool targetLocal = hasAddressSpace(copy.getTarget(), AddressSpace::Local);
  return (sourceGlobal && targetLocal) || (sourceLocal && targetGlobal);
}

static Value getIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static SmallVector<Value> delinearizeIndex(OpBuilder &builder, Location loc,
                                           Value linearIndex,
                                           ArrayRef<int64_t> shape) {
  SmallVector<Value> indices;
  indices.reserve(shape.size());
  if (shape.empty())
    return indices;

  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t runningStride = 1;
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    strides[dim] = runningStride;
    runningStride *= shape[dim];
  }

  for (auto [dim, stride] : llvm::enumerate(strides)) {
    Value coord = linearIndex;
    if (stride != 1) {
      coord = builder.create<arith::DivUIOp>(
          loc, coord, getIndexConstant(builder, loc, stride));
    }

    // Rank-1 indices already stay in range because the loop upper bound is the
    // number of tile elements. Higher-rank coordinates need modulo by the dim.
    if (shape.size() != 1) {
      coord = builder.create<arith::RemUIOp>(
          loc, coord, getIndexConstant(builder, loc, shape[dim]));
    }
    indices.push_back(coord);
  }

  return indices;
}

static LogicalResult verifyCopyShape(memref::CopyOp copy,
                                     MemRefType sourceType,
                                     MemRefType targetType) {
  if (!sourceType.hasStaticShape() || !targetType.hasStaticShape()) {
    return copy.emitOpError()
           << "requires static-shaped source and target for distributed local copy";
  }

  if (!sourceType.getShape().equals(targetType.getShape())) {
    return copy.emitOpError()
           << "requires source and target to have identical shapes for distributed local copy";
  }

  if (sourceType.getElementType() != targetType.getElementType()) {
    return copy.emitOpError()
           << "requires source and target to have identical element types for distributed local copy";
  }

  return success();
}

static LogicalResult rewriteLocalGlobalCopy(memref::CopyOp copy,
                                            LaunchOp launch) {
  auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!sourceType || !targetType)
    return copy.emitOpError()
           << "requires ranked memref source and target for distributed local copy";

  if (failed(verifyCopyShape(copy, sourceType, targetType)))
    return failure();

  int64_t tileElements = sourceType.getNumElements();
  if (tileElements == 0) {
    copy.erase();
    return success();
  }

  OpBuilder builder(copy);
  Location loc = copy.getLoc();
  Type indexType = builder.getIndexType();

  Value numSubgroups = launch.getOperand(1);
  Value numThreads = launch.getOperand(2);
  Value subgroupId = builder.create<SubgroupIdOp>(loc, indexType);
  Value threadId = builder.create<ThreadIdOp>(loc, indexType);
  Value subgroupBase =
      builder.create<arith::MulIOp>(loc, subgroupId, numThreads);
  Value linearTid = builder.create<arith::AddIOp>(loc, subgroupBase, threadId);
  Value laneCount = builder.create<arith::MulIOp>(loc, numSubgroups, numThreads);
  Value upperBound = getIndexConstant(builder, loc, tileElements);

  auto loop = builder.create<scf::ForOp>(loc, linearTid, upperBound, laneCount);
  builder.setInsertionPointToStart(loop.getBody());
  SmallVector<Value> indices = delinearizeIndex(
      builder, loc, loop.getInductionVar(), sourceType.getShape());
  Value loaded = builder.create<memref::LoadOp>(loc, copy.getSource(), indices);
  builder.create<memref::StoreOp>(loc, loaded, copy.getTarget(), indices);

  copy.erase();
  return success();
}

struct DistributeVortexLocalCopies
    : public impl::DistributeVortexLocalCopiesBase<
          DistributeVortexLocalCopies> {
  using impl::DistributeVortexLocalCopiesBase<
      DistributeVortexLocalCopies>::DistributeVortexLocalCopiesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VortexDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<std::pair<memref::CopyOp, LaunchOp>> worklist;
    func.walk([&](memref::CopyOp copy) {
      LaunchOp launch = copy->getParentOfType<LaunchOp>();
      if (!launch || !isSupportedLocalGlobalCopy(copy))
        return;
      worklist.push_back({copy, launch});
    });

    for (auto [copy, launch] : worklist) {
      if (!copy || !copy->getParentRegion())
        continue;
      if (failed(rewriteLocalGlobalCopy(copy, launch))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::vortex
