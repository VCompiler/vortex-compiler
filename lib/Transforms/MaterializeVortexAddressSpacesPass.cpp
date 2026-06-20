#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_MATERIALIZEVORTEXADDRESSSPACES
#include "vortex/Transforms/Passes.h.inc"

namespace {

static Type materializeGlobalAddressSpace(Type type, MLIRContext *context) {
  // 这里只改写 kernel 接口上的 memref 类型，而且仅在它还使用默认 memory
  // space 时才补成显式的 global。
  Attribute globalAddressSpace =
      AddressSpaceAttr::get(context, AddressSpace::Global);

  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    if (memrefType.getMemorySpace())
      return type;
    return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                           memrefType.getLayout(), globalAddressSpace);
  }

  if (auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(type)) {
    if (unrankedMemRefType.getMemorySpace())
      return type;
    return UnrankedMemRefType::get(unrankedMemRefType.getElementType(),
                                   globalAddressSpace);
  }

  return type;
}

static MemRefType retagMemRefTypeMemorySpace(MemRefType type,
                                             Attribute memorySpace) {
  return MemRefType::get(type.getShape(), type.getElementType(),
                         type.getLayout(), memorySpace);
}

static BaseMemRefType retagBaseMemRefTypeMemorySpace(BaseMemRefType type,
                                                     Attribute memorySpace) {
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return retagMemRefTypeMemorySpace(memrefType, memorySpace);
  if (auto unrankedType = dyn_cast<UnrankedMemRefType>(type))
    return UnrankedMemRefType::get(unrankedType.getElementType(), memorySpace);
  return type;
}

static LogicalResult rewriteSubviewResultTypes(func::FuncOp func) {
  SmallVector<memref::SubViewOp> worklist;
  func.walk([&](memref::SubViewOp subview) { worklist.push_back(subview); });

  IRRewriter rewriter(func.getContext());
  for (memref::SubViewOp subview : worklist) {
    auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
    auto resultType = dyn_cast<MemRefType>(subview.getResult().getType());
    if (!sourceType || !resultType)
      continue;

    Attribute sourceMemorySpace = sourceType.getMemorySpace();
    if (!sourceMemorySpace || resultType.getMemorySpace() == sourceMemorySpace)
      continue;

    MemRefType newResultType =
        retagMemRefTypeMemorySpace(resultType, sourceMemorySpace);
    rewriter.setInsertionPoint(subview);
    auto newSubview = rewriter.create<memref::SubViewOp>(
        subview.getLoc(), newResultType, subview.getSource(),
        subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides());
    newSubview->setAttrs(subview->getAttrs());
    rewriter.replaceOp(subview, newSubview.getResult());
  }

  return success();
}

static LogicalResult rewriteCastResultTypes(func::FuncOp func) {
  SmallVector<memref::CastOp> worklist;
  func.walk([&](memref::CastOp castOp) { worklist.push_back(castOp); });

  IRRewriter rewriter(func.getContext());
  for (memref::CastOp castOp : worklist) {
    auto sourceType = dyn_cast<BaseMemRefType>(castOp.getSource().getType());
    auto resultType = dyn_cast<BaseMemRefType>(castOp.getResult().getType());
    if (!sourceType || !resultType)
      continue;

    Attribute sourceMemorySpace = sourceType.getMemorySpace();
    if (!sourceMemorySpace || resultType.getMemorySpace() == sourceMemorySpace)
      continue;

    BaseMemRefType newResultType =
        retagBaseMemRefTypeMemorySpace(resultType, sourceMemorySpace);
    rewriter.setInsertionPoint(castOp);
    auto newCast = rewriter.create<memref::CastOp>(castOp.getLoc(),
                                                   newResultType,
                                                   castOp.getSource());
    newCast->setAttrs(castOp->getAttrs());
    rewriter.replaceOp(castOp, newCast.getResult());
  }

  return success();
}

struct MaterializeVortexAddressSpaces
    : public impl::MaterializeVortexAddressSpacesBase<
          MaterializeVortexAddressSpaces> {
  using impl::MaterializeVortexAddressSpacesBase<
      MaterializeVortexAddressSpaces>::MaterializeVortexAddressSpacesBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    SmallVector<Type> newInputTypes;
    FunctionType oldFunctionType = func.getFunctionType();
    newInputTypes.reserve(oldFunctionType.getNumInputs());

    bool changed = false;
    for (Type inputType : oldFunctionType.getInputs()) {
      Type newInputType =
          materializeGlobalAddressSpace(inputType, &getContext());
      changed |= newInputType != inputType;
      newInputTypes.push_back(newInputType);
    }

    if (!changed)
      return;

    // 如果只改函数类型、不改它的 symbol users，IR 会不一致，所以这一版直接拒绝。
    Operation *parentOp = func->getParentOp();
    if (parentOp && !SymbolTable::symbolKnownUseEmpty(func, parentOp)) {
      func.emitOpError()
          << "cannot materialize Vortex address spaces on a kernel with symbol "
             "users yet";
      signalPassFailure();
      return;
    }

    func.setType(FunctionType::get(&getContext(), newInputTypes,
                                   oldFunctionType.getResults()));

    // 外部声明只需要更新函数类型，不需要改 block 参数。
    if (func.isExternal())
      return;

    for (auto [index, newInputType] : llvm::enumerate(newInputTypes))
      func.getArgument(index).setType(newInputType);

    // 前端 bridge 产物里会立刻出现以 kernel 参数为基底的 subview/cast。
    // 只改函数参数类型会让这些 alias op 的结果类型还停留在默认地址空间，
    // 从而在 verifier 阶段直接失配。这里先做一个窄修复，只传播到
    // 当前已经明确会出现的 view/cast 结果类型上。
    if (failed(rewriteSubviewResultTypes(func)) ||
        failed(rewriteCastResultTypes(func))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::vortex
