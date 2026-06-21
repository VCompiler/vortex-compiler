#include "vortex/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"

#include "vortex/Dialect/Vortex/IR/VortexAttributes.h"
#include "vortex/Dialect/Vortex/IR/VortexDialect.h"
#include "vortex/Dialect/Vortex/IR/VortexOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vortex {

#define GEN_PASS_DEF_LEGALIZEVORTEXFORLLVM
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr StringLiteral kLowerLocalMemoryPassName =
    "vortex-lower-local-memory";
static constexpr StringLiteral kLaunchHelperAttr = "vortex.launch_helper";

static IntegerAttr wrapNumericMemorySpace(MLIRContext *context,
                                          unsigned addressSpace) {
  return IntegerAttr::get(IntegerType::get(context, 64), addressSpace);
}

static bool isExplicitLocalMemRefType(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType)
    return false;

  auto addressSpace =
      dyn_cast_or_null<AddressSpaceAttr>(memrefType.getMemorySpace());
  return addressSpace && addressSpace.getValue() == AddressSpace::Local;
}

static Type convertNonLocalVortexMemRefType(Type type) {
  auto convertMemorySpace = [](Attribute memorySpace) -> Attribute {
    auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
    if (!addressSpace)
      return {};

    switch (addressSpace.getValue()) {
    case AddressSpace::Global:
    case AddressSpace::Private:
      return wrapNumericMemorySpace(addressSpace.getContext(),
                                    static_cast<unsigned>(
                                        addressSpace.getValue()));
    case AddressSpace::Local:
      return {};
    }

    return {};
  };

  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    Attribute convertedMemorySpace =
        convertMemorySpace(memrefType.getMemorySpace());
    if (!convertedMemorySpace)
      return {};

    return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                           memrefType.getLayout(), convertedMemorySpace);
  }

  if (auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(type)) {
    Attribute convertedMemorySpace =
        convertMemorySpace(unrankedMemRefType.getMemorySpace());
    if (!convertedMemorySpace)
      return {};

    return UnrankedMemRefType::get(unrankedMemRefType.getElementType(),
                                   convertedMemorySpace);
  }

  return {};
}

static void rewriteModuleMemorySpaces(ModuleOp module) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([](MemRefType type) -> std::optional<Type> {
    Type converted = convertNonLocalVortexMemRefType(type);
    if (!converted)
      return std::nullopt;
    return converted;
  });
  replacer.addReplacement([](UnrankedMemRefType type) -> std::optional<Type> {
    Type converted = convertNonLocalVortexMemRefType(type);
    if (!converted)
      return std::nullopt;
    return converted;
  });

  replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

static bool isOperationInPayload(Operation *op,
                                 const llvm::DenseSet<Operation *> &payloadOps) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (payloadOps.contains(current))
      return true;
  }
  return false;
}

static bool isDefinedInPayload(Value value,
                               const llvm::DenseSet<Operation *> &payloadOps) {
  Operation *definingOp = value.getDefiningOp();
  return definingOp && isOperationInPayload(definingOp, payloadOps);
}

static bool shouldCloneExternalValueIntoLaunchHelper(Value value) {
  Operation *definingOp = value.getDefiningOp();
  return definingOp && isa<arith::ConstantOp>(definingOp);
}

static std::string getUniqueLaunchHelperName(ModuleOp module,
                                             func::FuncOp parent) {
  std::string base = (parent.getName() + "_vortex_launch_body").str();
  std::string name = base;
  for (unsigned ordinal = 0; module.lookupSymbol(name); ++ordinal)
    name = (Twine(base) + "_" + Twine(ordinal)).str();
  return name;
}

static LogicalResult outlineLaunchPayloadForSpawn(LaunchOp launch,
                                                  IRRewriter &rewriter) {
  if (launch.getBody().empty())
    return success();

  Block &body = launch.getBody().front();
  SmallVector<Operation *> bodyOps;
  for (Operation &op : body.without_terminator())
    bodyOps.push_back(&op);

  std::optional<unsigned> firstTmcIndex;
  std::optional<unsigned> lastTmcIndex;
  for (auto [index, op] : llvm::enumerate(bodyOps)) {
    if (!isa<TmcOp>(op))
      continue;
    if (!firstTmcIndex)
      firstTmcIndex = index;
    lastTmcIndex = index;
  }

  if (!firstTmcIndex || !lastTmcIndex || *firstTmcIndex >= *lastTmcIndex)
    return success();

  SmallVector<Operation *> payloadOps;
  for (unsigned i = *firstTmcIndex + 1; i < *lastTmcIndex; ++i)
    payloadOps.push_back(bodyOps[i]);
  if (payloadOps.empty())
    return success();

  llvm::DenseSet<Operation *> payloadSet;
  for (Operation *op : payloadOps)
    payloadSet.insert(op);

  SmallVector<Value> captures;
  for (Operation *payloadOp : payloadOps) {
    payloadOp->walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (isDefinedInPayload(operand, payloadSet))
          continue;
        if (shouldCloneExternalValueIntoLaunchHelper(operand))
          continue;
        if (!llvm::is_contained(captures, operand))
          captures.push_back(operand);
      }
    });
  }

  bool hasEscapingUse = false;
  for (Operation *payloadOp : payloadOps) {
    payloadOp->walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (isOperationInPayload(use.getOwner(), payloadSet))
            continue;
          op->emitError()
              << "cannot outline vortex.launch payload for spawn because a "
                 "payload result is used outside the payload";
          hasEscapingUse = true;
        }
      }
    });
  }
  if (hasEscapingUse)
    return failure();

  ModuleOp module = launch->getParentOfType<ModuleOp>();
  func::FuncOp parent = launch->getParentOfType<func::FuncOp>();
  if (!module || !parent)
    return launch.emitOpError()
           << "requires enclosing module and function for launch outlining";

  SmallVector<Type> captureTypes;
  captureTypes.reserve(captures.size());
  for (Value capture : captures)
    captureTypes.push_back(capture.getType());

  Location loc = launch.getLoc();
  std::string helperName = getUniqueLaunchHelperName(module, parent);
  auto helperType = rewriter.getFunctionType(captureTypes, {});

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(parent);
    auto helper = rewriter.create<func::FuncOp>(loc, helperName, helperType);
    helper.setPrivate();
    helper->setAttr(kLaunchHelperAttr, rewriter.getUnitAttr());

    Block *entry = helper.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    IRMapping mapping;
    for (auto [capture, argument] : llvm::zip(captures, entry->getArguments()))
      mapping.map(capture, argument);

    for (Operation *payloadOp : payloadOps) {
      payloadOp->walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (mapping.contains(operand) ||
              isDefinedInPayload(operand, payloadSet))
            continue;
          if (shouldCloneExternalValueIntoLaunchHelper(operand))
            rewriter.clone(*operand.getDefiningOp(), mapping);
        }
      });
    }

    for (Operation *op : payloadOps)
      rewriter.clone(*op, mapping);

    rewriter.create<func::ReturnOp>(loc);
  }

  rewriter.setInsertionPointAfter(bodyOps[*firstTmcIndex]);
  rewriter.create<func::CallOp>(loc, helperName, TypeRange{}, captures);

  for (Operation *op : llvm::reverse(payloadOps))
    rewriter.eraseOp(op);

  return success();
}

static LogicalResult inlineLaunch(LaunchOp launch, IRRewriter &rewriter) {
  if (failed(outlineLaunchPayloadForSpawn(launch, rewriter)))
    return failure();

  Block &body = launch.getBody().front();
  Operation *yield = body.getTerminator();

  rewriter.setInsertionPoint(launch);
  rewriter.inlineBlockBefore(&body, launch);
  rewriter.eraseOp(yield);
  rewriter.eraseOp(launch);
  return success();
}

static LogicalResult inlineLaunchesInKernel(func::FuncOp func,
                                            IRRewriter &rewriter) {
  if (!func->hasAttr(VortexDialect::getKernelAttrName()))
    return success();

  SmallVector<LaunchOp> launches;
  func.walk([&](LaunchOp launch) { launches.push_back(launch); });

  // 先处理更内层的 launch，避免外层 body 被搬运后让内层工作列表失效。
  for (LaunchOp launch : llvm::reverse(launches)) {
    if (failed(inlineLaunch(launch, rewriter)))
      return failure();
  }
  return success();
}

static LogicalResult validateFunctionTypeForLLVM(func::FuncOp func) {
  FunctionType type = func.getFunctionType();
  for (auto [index, argType] : llvm::enumerate(type.getInputs())) {
    if (!isExplicitLocalMemRefType(argType))
      continue;
    return func.emitOpError()
           << "requires running " << kLowerLocalMemoryPassName
           << " before vortex-legalize-for-llvm; function argument #" << index
           << " still uses #vortex.address_space<local>";
  }

  for (auto [index, resultType] : llvm::enumerate(type.getResults())) {
    if (!isExplicitLocalMemRefType(resultType))
      continue;
    return func.emitOpError()
           << "requires running " << kLowerLocalMemoryPassName
           << " before vortex-legalize-for-llvm; function result #" << index
           << " still uses #vortex.address_space<local>";
  }

  return success();
}

static LogicalResult validateModuleReadyForLLVM(ModuleOp module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (failed(validateFunctionTypeForLLVM(func)))
      return failure();
  }

  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (isa<LaunchOp, YieldOp, PredicatedOp, DivergentIfOp>(op)) {
      auto func = op->getParentOfType<func::FuncOp>();
      if (!func || !func->hasAttr(VortexDialect::getKernelAttrName())) {
        op->emitOpError()
            << "requires enclosing func.func marked with vortex.kernel";
      } else {
        op->emitOpError()
            << "expected all structured vortex control-flow operations to be "
               "eliminated by vortex-legalize-for-llvm";
      }
      return WalkResult::interrupt();
    }

    if (isa<LocalAllocOp>(op)) {
      op->emitOpError()
          << "requires running " << kLowerLocalMemoryPassName
          << " before vortex-legalize-for-llvm";
      return WalkResult::interrupt();
    }

    for (Value result : op->getResults()) {
      if (!isExplicitLocalMemRefType(result.getType()))
        continue;
      op->emitOpError()
          << "requires running " << kLowerLocalMemoryPassName
          << " before vortex-legalize-for-llvm; result still uses "
             "#vortex.address_space<local>";
      return WalkResult::interrupt();
    }

    for (Value operand : op->getOperands()) {
      if (!isExplicitLocalMemRefType(operand.getType()))
        continue;
      op->emitOpError()
          << "requires running " << kLowerLocalMemoryPassName
          << " before vortex-legalize-for-llvm; operand still uses "
             "#vortex.address_space<local>";
      return WalkResult::interrupt();
    }

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isExplicitLocalMemRefType(arg.getType()))
            continue;
          op->emitOpError()
              << "requires running " << kLowerLocalMemoryPassName
              << " before vortex-legalize-for-llvm; block argument still uses "
                 "#vortex.address_space<local>";
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

struct LegalizeVortexForLLVM
    : public impl::LegalizeVortexForLLVMBase<LegalizeVortexForLLVM> {
  using impl::LegalizeVortexForLLVMBase<
      LegalizeVortexForLLVM>::LegalizeVortexForLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    func::FuncDialect, scf::SCFDialect, VortexDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    {
      OpPassManager pipeline(ModuleOp::getOperationName());
      pipeline.addNestedPass<func::FuncOp>(createLowerAffinePass());
      if (failed(runPipeline(pipeline, module))) {
        signalPassFailure();
        return;
      }
    }

    IRRewriter rewriter(&getContext());
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp func) { functions.push_back(func); });
    for (func::FuncOp func : functions) {
      if (failed(inlineLaunchesInKernel(func, rewriter))) {
        signalPassFailure();
        return;
      }
    }

    rewriteModuleMemorySpaces(module);

    if (failed(validateModuleReadyForLLVM(module))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::vortex
