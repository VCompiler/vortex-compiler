#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

namespace mlir::vortex {

#define GEN_PASS_DEF_SUMMARIZEPREVORTEX
#include "vortex/Transforms/Passes.h.inc"

namespace {

static ArrayAttr makeStringArrayAttr(MLIRContext *context,
                                     const std::set<std::string> &values) {
  // 用 std::set 保证 summary 在多次运行之间保持稳定顺序。
  Builder builder(context);
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (const std::string &value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

static std::string stringifyAttribute(Attribute attr) {
  if (!attr)
    return "<default>";

  std::string storage;
  llvm::raw_string_ostream os(storage);
  attr.print(os);
  return os.str();
}

static void collectMemorySpace(Type type, std::set<std::string> &memorySpaces) {
  if (auto memrefType = dyn_cast<MemRefType>(type))
    memorySpaces.insert(stringifyAttribute(memrefType.getMemorySpace()));
}

static bool shouldIgnoreInSummary(Operation *op) {
  // 常量和 terminator 会增加噪声，但对规划 pre-vortex 边界帮助不大。
  return isa<arith::ConstantOp>(op) || op->hasTrait<OpTrait::IsTerminator>();
}

struct SummarizePreVortex
    : public impl::SummarizePreVortexBase<SummarizePreVortex> {
  using impl::SummarizePreVortexBase<
      SummarizePreVortex>::SummarizePreVortexBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    std::set<std::string> dialects;
    std::set<std::string> opNames;
    std::set<std::string> memorySpaces;

    // 把函数接口类型也纳入统计，这样即便函数体很小，kernel 边界上的 memory
    // space 信息也不会丢。
    for (Type type : func.getFunctionType().getInputs())
      collectMemorySpace(type, memorySpaces);
    for (Type type : func.getFunctionType().getResults())
      collectMemorySpace(type, memorySpaces);

    func.walk([&](Operation *op) {
      if (op == func.getOperation())
        return;

      for (Value operand : op->getOperands())
        collectMemorySpace(operand.getType(), memorySpaces);
      for (Value result : op->getResults())
        collectMemorySpace(result.getType(), memorySpaces);

      if (shouldIgnoreInSummary(op))
        return;

      opNames.insert(op->getName().getStringRef().str());
      dialects.insert(op->getName().getDialectNamespace().str());
    });

    func->setAttr("vortex.pre_vortex_ops",
                  makeStringArrayAttr(&getContext(), opNames));
    func->setAttr("vortex.pre_vortex_dialects",
                  makeStringArrayAttr(&getContext(), dialects));

    if (memorySpaces.empty())
      func->removeAttr("vortex.pre_vortex_memory_spaces");
    else
      func->setAttr("vortex.pre_vortex_memory_spaces",
                    makeStringArrayAttr(&getContext(), memorySpaces));
  }
};

} // namespace

} // namespace mlir::vortex
