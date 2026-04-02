#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_LOWERVORTEXLOCALMEMORY
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr StringLiteral kVxLocalMemBase = "vx_local_mem_base";
static constexpr StringLiteral kLocalFrameBytesAttrName =
    "vortex.local_frame_bytes";
static constexpr StringLiteral kLocalByteOffsetAttrName =
    "vortex.local.byte_offset";
static constexpr StringLiteral kLocalByteSizeAttrName =
    "vortex.local.byte_size";
static constexpr StringLiteral kLocalAlignmentAttrName =
    "vortex.local.alignment";

enum class LocalTraceKind {
  NotLocal,
  RootedLocal,
  UnsupportedAlias,
  EscapedLocal,
};

struct LocalTrace {
  LocalTraceKind kind = LocalTraceKind::NotLocal;
  LocalAllocOp root;
  Operation *aliasOp = nullptr;
};

static bool isExplicitLocalMemRefType(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType)
    return false;

  auto addressSpace =
      dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace());
  return addressSpace && addressSpace.getValue() == AddressSpace::Local;
}

static bool isUnsupportedLocalAliasOp(Operation *op) {
  return isa<memref::ReinterpretCastOp, memref::ExpandShapeOp,
             memref::CollapseShapeOp,
             memref::TransposeOp, memref::ViewOp>(op);
}

static LocalTrace traceLocalValue(Value value) {
  if (!isExplicitLocalMemRefType(value.getType()))
    return {};

  if (isa<BlockArgument>(value))
    return {LocalTraceKind::EscapedLocal, {}, nullptr};

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return {LocalTraceKind::EscapedLocal, {}, nullptr};

  if (auto alloc = dyn_cast<LocalAllocOp>(defOp))
    return {LocalTraceKind::RootedLocal, alloc, nullptr};

  if (auto cast = dyn_cast<memref::CastOp>(defOp))
    return traceLocalValue(cast.getSource());

  if (auto subview = dyn_cast<memref::SubViewOp>(defOp))
    return traceLocalValue(subview.getSource());

  if (isUnsupportedLocalAliasOp(defOp)) {
    LocalTrace inner = traceLocalValue(defOp->getOperand(0));
    if (inner.kind == LocalTraceKind::RootedLocal) {
      inner.kind = LocalTraceKind::UnsupportedAlias;
      inner.aliasOp = defOp;
    }
    return inner;
  }

  return {LocalTraceKind::EscapedLocal, {}, nullptr};
}

static FailureOr<func::FuncOp>
getOrCreateWrapperDecl(ModuleOp module, StringRef name, FunctionType type,
                       OpBuilder &builder, Location loc) {
  if (auto func = module.lookupSymbol<func::FuncOp>(name)) {
    if (func.getFunctionType() != type)
      return failure();
    return func;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto func = builder.create<func::FuncOp>(loc, name, type);
  func.setPrivate();
  return func;
}

static LogicalResult ensureKernelContext(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || !func->hasAttr(VortexDialect::getKernelAttrName())) {
    return op->emitOpError()
           << "requires enclosing func.func marked with vortex.kernel";
  }
  return success();
}

static LogicalResult validateLocalAllocReady(LocalAllocOp alloc) {
  if (failed(ensureKernelContext(alloc)))
    return failure();

  auto func = alloc->getParentOfType<func::FuncOp>();
  if (!func->hasAttr(kLocalFrameBytesAttrName)) {
    return alloc.emitOpError()
           << "requires running vortex-plan-local-memory-layout first";
  }

  if (!alloc->hasAttr(kLocalByteOffsetAttrName) ||
      !alloc->hasAttr(kLocalByteSizeAttrName) ||
      !alloc->hasAttr(kLocalAlignmentAttrName)) {
    return alloc.emitOpError()
           << "requires running vortex-plan-local-memory-layout first";
  }

  return success();
}

static LogicalResult validateLocalUser(Operation *op) {
  if (failed(ensureKernelContext(op)))
    return failure();

  for (Value operand : op->getOperands()) {
    LocalTrace trace = traceLocalValue(operand);
    switch (trace.kind) {
    case LocalTraceKind::NotLocal:
      continue;
    case LocalTraceKind::RootedLocal:
      break;
    case LocalTraceKind::UnsupportedAlias:
      return op->emitOpError()
             << "lowering local memref through '" << trace.aliasOp->getName()
             << "' is not implemented yet in the current MVP";
    case LocalTraceKind::EscapedLocal:
      return op->emitOpError()
             << "encountered local memref that no longer traces back to "
                "vortex.local_alloc";
    }

    if (isa<memref::CastOp, memref::SubViewOp, memref::LoadOp, memref::StoreOp,
            memref::CopyOp>(op))
      continue;

    if (isUnsupportedLocalAliasOp(op)) {
      return op->emitOpError()
             << "lowering local memref through '" << op->getName()
             << "' is not implemented yet in the current MVP";
    }

    if (isa<CallOpInterface>(op)) {
      return op->emitOpError()
             << "cannot pass local memref values through call";
    }

    if (isa<BranchOpInterface, RegionBranchOpInterface,
            RegionBranchTerminatorOpInterface>(op)) {
      return op->emitOpError()
             << "cannot pass local memref values through yield/branch/iter_args";
    }

    return op->emitOpError()
           << "local memref user '" << op->getName()
           << "' is not implemented yet in vortex-lower-local-memory";
  }

  return success();
}

static LogicalResult validateModule(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (auto alloc = dyn_cast<LocalAllocOp>(op)) {
      if (failed(validateLocalAllocReady(alloc)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    for (Value operand : op->getOperands()) {
      if (!isExplicitLocalMemRefType(operand.getType()))
        continue;
      if (failed(validateLocalUser(op)))
        return WalkResult::interrupt();
      break;
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

static Value getI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, 64);
}

static FailureOr<uint64_t> getRequiredAttrI64(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return failure();
  return attr.getInt();
}

static Value castIndexLikeToI64(IRRewriter &rewriter, Location loc,
                                Value value) {
  return getValueOrCreateCastToIndexLike(rewriter, loc, rewriter.getI64Type(),
                                         value);
}

static Value materializeIndexLikeAsI64(IRRewriter &rewriter, Location loc,
                                       OpFoldResult ofr) {
  Value value = getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
  return castIndexLikeToI64(rewriter, loc, value);
}

static FailureOr<SmallVector<Value>>
composeIndicesToLocalRoot(Value memrefValue, ValueRange currentIndices,
                          Location loc, IRRewriter &rewriter,
                          LocalAllocOp &rootAlloc) {
  if (!isExplicitLocalMemRefType(memrefValue.getType()))
    return failure();

  if (isa<BlockArgument>(memrefValue))
    return failure();

  Operation *defOp = memrefValue.getDefiningOp();
  if (!defOp)
    return failure();

  if (auto alloc = dyn_cast<LocalAllocOp>(defOp)) {
    SmallVector<Value> rootIndices;
    rootIndices.reserve(currentIndices.size());
    for (Value index : currentIndices)
      rootIndices.push_back(castIndexLikeToI64(rewriter, loc, index));
    rootAlloc = alloc;
    return rootIndices;
  }

  if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
    return composeIndicesToLocalRoot(cast.getSource(), currentIndices, loc,
                                     rewriter, rootAlloc);
  }

  if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
    auto resultType = subview.getType();
    if (static_cast<size_t>(resultType.getRank()) != currentIndices.size()) {
      subview.emitOpError()
          << "expects indices to match the subview result rank during local "
             "address lowering";
      return failure();
    }

    auto sourceType = subview.getSourceType();
    unsigned sourceRank = sourceType.getRank();
    SmallVector<OpFoldResult> mixedOffsets = subview.getMixedOffsets();
    SmallVector<OpFoldResult> mixedStrides = subview.getMixedStrides();
    if (mixedOffsets.size() != sourceRank || mixedStrides.size() != sourceRank) {
      subview.emitOpError()
          << "expects offsets/strides to match the source rank during local "
             "address lowering";
      return failure();
    }

    llvm::SmallBitVector droppedDims = subview.getDroppedDims();
    SmallVector<Value> sourceIndices;
    sourceIndices.reserve(sourceRank);

    size_t resultDim = 0;
    for (unsigned sourceDim = 0; sourceDim < sourceRank; ++sourceDim) {
      Value offsetI64 =
          materializeIndexLikeAsI64(rewriter, loc, mixedOffsets[sourceDim]);
      if (droppedDims.test(sourceDim)) {
        sourceIndices.push_back(offsetI64);
        continue;
      }

      if (resultDim >= currentIndices.size()) {
        subview.emitOpError()
            << "cannot map subview indices back to the local allocation";
        return failure();
      }

      Value indexI64 =
          castIndexLikeToI64(rewriter, loc, currentIndices[resultDim++]);
      Value strideI64 =
          materializeIndexLikeAsI64(rewriter, loc, mixedStrides[sourceDim]);
      Value scaledIndex =
          rewriter.create<arith::MulIOp>(loc, indexI64, strideI64);
      sourceIndices.push_back(
          rewriter.create<arith::AddIOp>(loc, offsetI64, scaledIndex));
    }

    if (resultDim != currentIndices.size()) {
      subview.emitOpError()
          << "cannot map all subview result indices back to the local "
             "allocation";
      return failure();
    }

    return composeIndicesToLocalRoot(subview.getSource(), sourceIndices, loc,
                                     rewriter, rootAlloc);
  }

  if (isUnsupportedLocalAliasOp(defOp)) {
    defOp->emitOpError()
        << "lowering local memref through '" << defOp->getName()
        << "' is not implemented yet in the current MVP";
    return failure();
  }

  defOp->emitOpError()
      << "encountered unexpected local memref producer during local address "
         "lowering";
  return failure();
}

static FailureOr<Value> linearizeIndices(IRRewriter &rewriter, Location loc,
                                         ValueRange indices,
                                         MemRefType rootType) {
  if (static_cast<size_t>(rootType.getRank()) != indices.size())
    return failure();

  if (indices.empty())
    return getI64Constant(rewriter, loc, 0);

  Value linear = castIndexLikeToI64(rewriter, loc, indices.front());
  for (size_t dimIndex = 1, e = indices.size(); dimIndex < e; ++dimIndex) {
    int64_t dimSize = rootType.getShape()[dimIndex - 1];
    if (ShapedType::isDynamic(dimSize))
      return failure();
    linear = rewriter.create<arith::MulIOp>(
        loc, linear, getI64Constant(rewriter, loc, dimSize));
    Value indexI64 =
        castIndexLikeToI64(rewriter, loc, indices[dimIndex]);
    linear = rewriter.create<arith::AddIOp>(loc, linear, indexI64);
  }
  return linear;
}

static FailureOr<unsigned> getElementAlignment(LocalAllocOp alloc,
                                               const DataLayout &dataLayout) {
  auto type = dyn_cast<MemRefType>(alloc.getBuffer().getType());
  if (!type)
    return failure();
  return static_cast<unsigned>(
      dataLayout.getTypeABIAlignment(type.getElementType()));
}

static FailureOr<Value>
getOrCreateLocalBaseValue(func::FuncOp func, func::FuncOp wrapper,
                          IRRewriter &rewriter,
                          llvm::DenseMap<Operation *, Value> &cache) {
  if (auto it = cache.find(func.getOperation()); it != cache.end())
    return it->second;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&func.getBody().front());
  Value localBase =
      rewriter.create<func::CallOp>(func.getLoc(), wrapper, ValueRange{})
          .getResult(0);
  cache[func.getOperation()] = localBase;
  return localBase;
}

static FailureOr<Value>
computeLocalAddress(Value memrefValue, ValueRange indices, Location loc,
                    IRRewriter &rewriter, func::FuncOp localBaseWrapper,
                    llvm::DenseMap<Operation *, Value> &localBaseCache,
                    const DataLayout &dataLayout) {
  LocalAllocOp rootAlloc;
  FailureOr<SmallVector<Value>> rootIndices =
      composeIndicesToLocalRoot(memrefValue, indices, loc, rewriter, rootAlloc);
  if (failed(rootIndices) || !rootAlloc)
    return failure();

  auto rootType = dyn_cast<MemRefType>(rootAlloc.getBuffer().getType());
  if (!rootType ||
      static_cast<size_t>(rootType.getRank()) != rootIndices->size()) {
    rootAlloc.emitOpError()
        << "local address lowering expects indices to match the root "
           "vortex.local_alloc rank";
    return failure();
  }

  FailureOr<uint64_t> byteOffset =
      getRequiredAttrI64(rootAlloc, kLocalByteOffsetAttrName);
  if (failed(byteOffset)) {
    rootAlloc.emitOpError()
        << "requires vortex.local.byte_offset from "
           "vortex-plan-local-memory-layout";
    return failure();
  }

  llvm::TypeSize elementSize =
      dataLayout.getTypeSize(rootType.getElementType());
  if (elementSize.isScalable()) {
    rootAlloc.emitOpError() << "requires fixed-size local element types";
    return failure();
  }

  FailureOr<Value> localBase = getOrCreateLocalBaseValue(
      rootAlloc->getParentOfType<func::FuncOp>(), localBaseWrapper, rewriter,
      localBaseCache);
  if (failed(localBase))
    return failure();

  Value address = *localBase;
  if (*byteOffset != 0) {
    address = rewriter.create<arith::AddIOp>(
        loc, address, getI64Constant(rewriter, loc, *byteOffset));
  }

  FailureOr<Value> linearIndex =
      linearizeIndices(rewriter, loc, *rootIndices, rootType);
  if (failed(linearIndex)) {
    rootAlloc.emitOpError()
        << "failed to linearize indices for the root vortex.local_alloc";
    return failure();
  }
  uint64_t elementBytes = elementSize.getFixedValue();
  if (elementBytes != 1) {
    *linearIndex = rewriter.create<arith::MulIOp>(
        loc, *linearIndex,
        getI64Constant(rewriter, loc, static_cast<int64_t>(elementBytes)));
  }

  address = rewriter.create<arith::AddIOp>(loc, address, *linearIndex);
  return address;
}

static LogicalResult expandLocalCopy(memref::CopyOp copy, IRRewriter &rewriter) {
  auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!sourceType || !targetType || !sourceType.hasStaticShape() ||
      !targetType.hasStaticShape() || sourceType.getShape() != targetType.getShape()) {
    return copy.emitOpError()
           << "currently requires static-shaped memref.copy when local memory is involved";
  }

  SmallVector<Value> lbs;
  SmallVector<Value> ubs;
  SmallVector<Value> steps;
  lbs.reserve(sourceType.getRank());
  ubs.reserve(sourceType.getRank());
  steps.reserve(sourceType.getRank());

  Location loc = copy.getLoc();
  rewriter.setInsertionPoint(copy);
  for (int64_t dim : sourceType.getShape()) {
    lbs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
    steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
  }

  scf::buildLoopNest(rewriter, loc, lbs, ubs, steps,
                     [&](OpBuilder &builder, Location bodyLoc,
                         ValueRange ivs) {
                       Value loaded = builder.create<memref::LoadOp>(
                           bodyLoc, copy.getSource(), ivs);
                       builder.create<memref::StoreOp>(bodyLoc, loaded,
                                                       copy.getTarget(), ivs);
                     });
  rewriter.eraseOp(copy);
  return success();
}

static LogicalResult lowerLocalLoad(ModuleOp module, memref::LoadOp load,
                                    IRRewriter &rewriter,
                                    func::FuncOp localBaseWrapper,
                                    llvm::DenseMap<Operation *, Value> &cache,
                                    const DataLayout &dataLayout) {
  LocalTrace trace = traceLocalValue(load.getMemRef());
  if (trace.kind != LocalTraceKind::RootedLocal)
    return success();

  rewriter.setInsertionPoint(load);
  FailureOr<Value> address =
      computeLocalAddress(load.getMemRef(), load.getIndices(), load.getLoc(),
                          rewriter, localBaseWrapper, cache, dataLayout);
  if (failed(address))
    return failure();

  FailureOr<unsigned> alignment = getElementAlignment(trace.root, dataLayout);
  if (failed(alignment)) {
    return load.emitOpError()
           << "failed to compute local load alignment from root allocation";
  }

  rewriter.setInsertionPoint(load);
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value ptr =
      rewriter.create<LLVM::IntToPtrOp>(load.getLoc(), ptrType, *address);
  Value lowered =
      rewriter.create<LLVM::LoadOp>(load.getLoc(), load.getType(), ptr,
                                    *alignment);
  rewriter.replaceOp(load, lowered);
  return success();
}

static LogicalResult lowerLocalStore(ModuleOp module, memref::StoreOp store,
                                     IRRewriter &rewriter,
                                     func::FuncOp localBaseWrapper,
                                     llvm::DenseMap<Operation *, Value> &cache,
                                     const DataLayout &dataLayout) {
  LocalTrace trace = traceLocalValue(store.getMemRef());
  if (trace.kind != LocalTraceKind::RootedLocal)
    return success();

  rewriter.setInsertionPoint(store);
  FailureOr<Value> address =
      computeLocalAddress(store.getMemRef(), store.getIndices(), store.getLoc(),
                          rewriter, localBaseWrapper, cache, dataLayout);
  if (failed(address))
    return failure();

  FailureOr<unsigned> alignment = getElementAlignment(trace.root, dataLayout);
  if (failed(alignment)) {
    return store.emitOpError()
           << "failed to compute local store alignment from root allocation";
  }

  rewriter.setInsertionPoint(store);
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value ptr =
      rewriter.create<LLVM::IntToPtrOp>(store.getLoc(), ptrType, *address);
  rewriter.create<LLVM::StoreOp>(store.getLoc(), store.getValueToStore(), ptr,
                                 *alignment);
  rewriter.eraseOp(store);
  return success();
}

struct LowerVortexLocalMemory
    : public impl::LowerVortexLocalMemoryBase<LowerVortexLocalMemory> {
  using impl::LowerVortexLocalMemoryBase<
      LowerVortexLocalMemory>::LowerVortexLocalMemoryBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect, scf::SCFDialect, VortexDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (failed(validateModule(module))) {
      signalPassFailure();
      return;
    }

    SmallVector<LocalAllocOp> localAllocs;
    SmallVector<memref::CastOp> localCasts;
    SmallVector<memref::SubViewOp> localSubViews;
    SmallVector<memref::CopyOp> localCopies;
    SmallVector<memref::LoadOp> localLoads;
    SmallVector<memref::StoreOp> localStores;

    module.walk([&](LocalAllocOp op) { localAllocs.push_back(op); });
    module.walk([&](memref::CastOp op) {
      if (traceLocalValue(op.getResult()).kind != LocalTraceKind::NotLocal)
        localCasts.push_back(op);
    });
    module.walk([&](memref::SubViewOp op) {
      if (traceLocalValue(op.getResult()).kind != LocalTraceKind::NotLocal)
        localSubViews.push_back(op);
    });
    module.walk([&](memref::CopyOp op) {
      LocalTrace sourceTrace = traceLocalValue(op.getSource());
      LocalTrace targetTrace = traceLocalValue(op.getTarget());
      if (sourceTrace.kind == LocalTraceKind::RootedLocal ||
          targetTrace.kind == LocalTraceKind::RootedLocal)
        localCopies.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (memref::CopyOp copy : localCopies) {
      if (!copy || !copy->getParentRegion())
        continue;
      if (failed(expandLocalCopy(copy, rewriter))) {
        signalPassFailure();
        return;
      }
    }

    module.walk([&](memref::LoadOp op) {
      if (traceLocalValue(op.getMemRef()).kind == LocalTraceKind::RootedLocal)
        localLoads.push_back(op);
    });
    module.walk([&](memref::StoreOp op) {
      if (traceLocalValue(op.getMemRef()).kind == LocalTraceKind::RootedLocal)
        localStores.push_back(op);
    });

    bool needsLocalBaseWrapper = !localLoads.empty() || !localStores.empty();

    func::FuncOp localBaseWrapper;
    if (needsLocalBaseWrapper) {
      OpBuilder builder(module.getContext());
      auto wrapperType =
          builder.getFunctionType({}, TypeRange{builder.getI64Type()});
      FailureOr<func::FuncOp> wrapper = getOrCreateWrapperDecl(
          module, kVxLocalMemBase, wrapperType, builder, module.getLoc());
      if (failed(wrapper)) {
        module.emitError()
            << "wrapper declaration type mismatch for " << kVxLocalMemBase;
        signalPassFailure();
        return;
      }
      localBaseWrapper = *wrapper;
    }

    llvm::DenseMap<Operation *, Value> localBaseCache;
    for (memref::LoadOp load : localLoads) {
      if (!load || !load->getParentRegion())
        continue;
      DataLayout dataLayout = DataLayout::closest(load->getParentOfType<func::FuncOp>());
      if (failed(lowerLocalLoad(module, load, rewriter, localBaseWrapper,
                                localBaseCache, dataLayout))) {
        signalPassFailure();
        return;
      }
    }

    for (memref::StoreOp store : localStores) {
      if (!store || !store->getParentRegion())
        continue;
      DataLayout dataLayout =
          DataLayout::closest(store->getParentOfType<func::FuncOp>());
      if (failed(lowerLocalStore(module, store, rewriter, localBaseWrapper,
                                 localBaseCache, dataLayout))) {
        signalPassFailure();
        return;
      }
    }

    for (memref::CastOp cast : llvm::reverse(localCasts)) {
      if (!cast || !cast->getParentRegion())
        continue;
      if (!cast.use_empty()) {
        cast.emitOpError()
            << "expected all local memref.cast users to be lowered away";
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(cast);
    }

    for (memref::SubViewOp subview : llvm::reverse(localSubViews)) {
      if (!subview || !subview->getParentRegion())
        continue;
      if (!subview.use_empty()) {
        subview.emitOpError()
            << "expected all local memref.subview users to be lowered away";
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(subview);
    }

    for (LocalAllocOp alloc : llvm::reverse(localAllocs)) {
      if (!alloc || !alloc->getParentRegion())
        continue;
      if (!alloc.getBuffer().use_empty()) {
        alloc.emitOpError()
            << "expected all vortex.local_alloc users to be lowered away";
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(alloc);
    }
  }
};

} // namespace

} // namespace mlir::vortex
