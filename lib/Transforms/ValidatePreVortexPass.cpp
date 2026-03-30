#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::vortex {

#define GEN_PASS_DEF_VALIDATEPREVORTEX
#include "vortex/Transforms/Passes.h.inc"

namespace {

// 这份列表定义了当前的 "pre-vortex" 合约。后续如果有新的 lowering 阶段引入
// 其他 dialect，需要显式更新这条边界。
static constexpr llvm::StringLiteral kAllowedPreVortexDialects[] = {
    "affine", "arith", "bufferization", "builtin", "cf",    "func",
    "linalg", "math",  "memref",        "scf",     "tensor", "vector"};

static bool isAllowedPreVortexDialect(StringRef dialectNamespace) {
  return llvm::is_contained(ArrayRef(kAllowedPreVortexDialects),
                            dialectNamespace);
}

static std::string buildAllowedDialectList() {
  // 在诊断里直接展开允许列表，让报错本身就是自解释的。
  SmallVector<StringRef> values;
  values.reserve(ArrayRef(kAllowedPreVortexDialects).size());
  for (StringRef dialect : kAllowedPreVortexDialects)
    values.push_back(dialect);
  return llvm::join(values, ", ");
}

struct ValidatePreVortex
    : public impl::ValidatePreVortexBase<ValidatePreVortex> {
  using impl::ValidatePreVortexBase<
      ValidatePreVortex>::ValidatePreVortexBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    WalkResult result = func.walk([&](Operation *op) {
      if (op == func.getOperation())
        return WalkResult::advance();

      StringRef dialectNamespace = op->getName().getDialectNamespace();
      if (isAllowedPreVortexDialect(dialectNamespace))
        return WalkResult::advance();

      // 第一次遇到越界 op 就直接失败，让边界问题尽量早暴露、尽量好定位。
      op->emitOpError() << "pre-vortex IR does not allow dialect '"
                        << dialectNamespace << "'; allowed dialects: "
                        << buildAllowedDialectList();
      return WalkResult::interrupt();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::vortex
