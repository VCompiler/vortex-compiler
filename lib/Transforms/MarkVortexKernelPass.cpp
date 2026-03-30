#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "vortex/Dialect/Vortex/IR/VortexDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::vortex {

#define GEN_PASS_DEF_MARKVORTEXKERNEL
#include "vortex/Transforms/Passes.h.inc"

namespace {

// 在 kernel 识别还没有完全结构化之前，先用这个临时标记承接前置流程。
static constexpr llvm::StringLiteral kTemporaryKernelAttrName = "vortex.entry";

static bool isNamedKernel(ArrayRef<std::string> kernelNames,
                          StringRef funcName) {
  // 第一版只做精确名字匹配，避免引入隐式推断。
  return llvm::any_of(kernelNames,
                      [&](const std::string &name) { return name == funcName; });
}

struct MarkVortexKernel
    : public impl::MarkVortexKernelBase<MarkVortexKernel> {
  using impl::MarkVortexKernelBase<MarkVortexKernel>::MarkVortexKernelBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    // 这个 pass 故意保持很窄：只负责给入口函数打标记，不在这里改函数签名，
    // 也不在这里生成 launch 等执行结构。
    if (func->hasAttr(VortexDialect::getKernelAttrName()))
      return;

    bool shouldMark = isNamedKernel(kernelNames, func.getSymName());
    if (!shouldMark && recognizeEntryAttr)
      shouldMark = func->hasAttr(kTemporaryKernelAttrName);

    if (!shouldMark)
      return;

    func->setAttr(VortexDialect::getKernelAttrName(),
                  UnitAttr::get(&getContext()));
    if (removeEntryAttr)
      func->removeAttr(kTemporaryKernelAttrName);
  }
};

} // namespace

} // namespace mlir::vortex
