#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
  }
};

} // namespace

} // namespace mlir::vortex
