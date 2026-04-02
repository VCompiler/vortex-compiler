#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_NORMALIZEONNXFRONTEND
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kONNXEntryPointOpName = "onnx.EntryPoint";
static constexpr llvm::StringLiteral kFrontendKernelMarker = "vortex.entry";
static constexpr llvm::StringLiteral kONNXAttrPrefix = "onnx.";
static constexpr llvm::StringLiteral kONNXMLIRAttrPrefix = "onnx-mlir.";

static bool isFrontendAttrName(StringRef name) {
  return name.starts_with(kONNXAttrPrefix) ||
         name.starts_with(kONNXMLIRAttrPrefix);
}

static void stripFrontendAttrs(Operation *op) {
  SmallVector<StringAttr> attrsToRemove;
  for (NamedAttribute namedAttr : op->getAttrs()) {
    if (isFrontendAttrName(namedAttr.getName().strref()))
      attrsToRemove.push_back(namedAttr.getName());
  }

  for (StringAttr attrName : attrsToRemove)
    op->removeAttr(attrName);
}

static void stripFuncBoundaryFrontendAttrs(func::FuncOp func) {
  for (unsigned argIndex = 0, e = func.getNumArguments(); argIndex < e;
       ++argIndex) {
    SmallVector<StringAttr> attrsToRemove;
    for (NamedAttribute namedAttr : func.getArgAttrDict(argIndex)) {
      if (isFrontendAttrName(namedAttr.getName().strref()))
        attrsToRemove.push_back(namedAttr.getName());
    }
    for (StringAttr attrName : attrsToRemove)
      func.removeArgAttr(argIndex, attrName);
  }

  for (unsigned resultIndex = 0, e = func.getNumResults(); resultIndex < e;
       ++resultIndex) {
    SmallVector<StringAttr> attrsToRemove;
    for (NamedAttribute namedAttr : func.getResultAttrDict(resultIndex)) {
      if (isFrontendAttrName(namedAttr.getName().strref()))
        attrsToRemove.push_back(namedAttr.getName());
    }
    for (StringAttr attrName : attrsToRemove)
      func.removeResultAttr(resultIndex, attrName);
  }
}

static LogicalResult materializeFrontendEntryPoint(Operation *entryOp,
                                                   ModuleOp module) {
  auto funcAttr = entryOp->getAttrOfType<FlatSymbolRefAttr>("func");
  if (!funcAttr) {
    return entryOp->emitOpError()
           << "requires FlatSymbolRefAttr 'func' to identify the entry "
              "function";
  }

  auto func = module.lookupSymbol<func::FuncOp>(funcAttr.getValue());
  if (!func) {
    return entryOp->emitOpError()
           << "references unknown func.func @" << funcAttr.getValue();
  }

  // 这里先只把 ONNX 入口信息折叠成现有后半段已经认识的临时标记，
  // 避免前端桥接阶段就引入新的 kernel 识别语义。
  func->setAttr(kFrontendKernelMarker, UnitAttr::get(module.getContext()));
  return success();
}

struct NormalizeONNXFrontend
    : public impl::NormalizeONNXFrontendBase<NormalizeONNXFrontend> {
  using impl::NormalizeONNXFrontendBase<
      NormalizeONNXFrontend>::NormalizeONNXFrontendBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    SmallVector<Operation *> entryOps;
    for (Operation &op : module.getBody()->without_terminator()) {
      if (op.getName().getStringRef() == kONNXEntryPointOpName)
        entryOps.push_back(&op);
    }

    for (Operation *entryOp : entryOps) {
      if (failed(materializeFrontendEntryPoint(entryOp, module))) {
        signalPassFailure();
        return;
      }
    }

    for (Operation *entryOp : entryOps)
      entryOp->erase();

    module.walk([&](Operation *op) {
      stripFrontendAttrs(op);
      if (auto func = dyn_cast<func::FuncOp>(op))
        stripFuncBoundaryFrontendAttrs(func);
    });
  }
};

} // namespace

} // namespace mlir::vortex
