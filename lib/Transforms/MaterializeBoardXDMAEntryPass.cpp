#include "vortex/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <limits>

namespace mlir::vortex {

#define GEN_PASS_DEF_MATERIALIZEBOARDXDMAENTRY
#include "vortex/Transforms/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kLoweredKernelMarker =
    "vortex.kernel_entry";
static constexpr llvm::StringLiteral kStartupArg =
    "vortex_board_xdma_startup_arg";
static constexpr llvm::StringLiteral kExit = "vortex_board_xdma_exit";
static constexpr int64_t kDescriptorHeaderBytes = 8;

static Type getPointerSizedIntegerType(MLIRContext *context, int64_t xlen) {
  return IntegerType::get(context, static_cast<unsigned>(xlen));
}

static FailureOr<LLVM::LLVMFuncOp>
getOrCreateLLVMFuncDecl(ModuleOp module, StringRef name,
                        LLVM::LLVMFunctionType type, OpBuilder &builder,
                        Location loc) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    if (func.getFunctionType() != type)
      return failure();
    return func;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto func = builder.create<LLVM::LLVMFuncOp>(loc, name, type,
                                               LLVM::Linkage::External);
  func.setSymVisibility("private");
  return func;
}

static FailureOr<LLVM::LLVMFuncOp> findKernelToWrap(ModuleOp module,
                                                    StringRef kernelName) {
  if (!kernelName.empty()) {
    auto kernel = module.lookupSymbol<LLVM::LLVMFuncOp>(kernelName);
    if (!kernel) {
      module.emitError() << "requested board/XDMA kernel '" << kernelName
                         << "' was not found";
      return failure();
    }
    if (!kernel->hasAttr(kLoweredKernelMarker)) {
      kernel.emitError() << "requested board/XDMA kernel must carry "
                         << kLoweredKernelMarker;
      return failure();
    }
    return kernel;
  }

  SmallVector<LLVM::LLVMFuncOp> kernels;
  for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
    if (func->hasAttr(kLoweredKernelMarker))
      kernels.push_back(func);
  }

  if (kernels.empty()) {
    module.emitError() << "expected one lowered Vortex kernel marked with "
                       << kLoweredKernelMarker;
    return failure();
  }
  if (kernels.size() != 1) {
    module.emitError()
        << "expected exactly one lowered Vortex kernel marked with "
        << kLoweredKernelMarker
        << "; pass --kernel-name to select one explicitly";
    return failure();
  }
  return kernels.front();
}

static LogicalResult validateKernelSignature(LLVM::LLVMFuncOp kernel) {
  LLVM::LLVMFunctionType type = kernel.getFunctionType();
  MLIRContext *context = kernel.getContext();
  Type voidType = LLVM::LLVMVoidType::get(context);

  if (type.isVarArg())
    return kernel.emitError()
           << "board/XDMA entry wrapper does not support variadic kernels";
  if (type.getReturnType() != voidType)
    return kernel.emitError()
           << "board/XDMA entry wrapper expects a void-returning kernel";
  if (kernel.empty())
    return kernel.emitError()
           << "board/XDMA entry wrapper expects a defined kernel body";

  for (unsigned i = 0, e = type.getNumParams(); i < e; ++i) {
    if (!llvm::isa<LLVM::LLVMPointerType>(type.getParamType(i))) {
      return kernel.emitError()
             << "board/XDMA entry wrapper currently supports only "
                "bare-pointer lowered kernel arguments; argument "
             << i << " has type " << type.getParamType(i);
    }
  }
  return success();
}

static LogicalResult validateOptions(Operation *op, int64_t xlen,
                                     int64_t argBaseOffset,
                                     int64_t argStride) {
  if (xlen != 32 && xlen != 64)
    return op->emitError() << "--xlen must be 32 or 64";
  if (argBaseOffset < kDescriptorHeaderBytes)
    return op->emitError()
           << "--arg-base-offset must leave room for the 8-byte descriptor "
              "header";
  if (argStride <= 0)
    return op->emitError() << "--arg-stride must be positive";
  return success();
}

static LogicalResult checkedSlotOffset(Operation *op, int64_t argBaseOffset,
                                       int64_t argStride, unsigned index,
                                       int32_t &offset) {
  int64_t rawOffset = argBaseOffset + argStride * static_cast<int64_t>(index);
  if (rawOffset < 0 || rawOffset > std::numeric_limits<int32_t>::max()) {
    return op->emitError() << "computed descriptor slot offset " << rawOffset
                           << " does not fit in an LLVM GEP constant index";
  }
  offset = static_cast<int32_t>(rawOffset);
  return success();
}

static Value createI32Constant(OpBuilder &builder, Location loc,
                               int64_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), value);
}

static LogicalResult materializeEntry(ModuleOp module, LLVM::LLVMFuncOp kernel,
                                      LLVM::LLVMFuncOp startup,
                                      LLVM::LLVMFuncOp exit,
                                      StringRef entryName, int64_t argBaseOffset,
                                      int64_t argStride,
                                      int64_t exitStatus,
                                      OpBuilder &builder) {
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(entryName)) {
    module.emitError() << "cannot generate board/XDMA entry '" << entryName
                       << "' because that symbol already exists";
    return failure();
  }

  Location loc = kernel.getLoc();
  MLIRContext *context = module.getContext();
  Type i32Type = builder.getI32Type();
  Type i64Type = builder.getI64Type();
  Type i8Type = builder.getI8Type();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  auto mainType = LLVM::LLVMFunctionType::get(i32Type, {});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto main = builder.create<LLVM::LLVMFuncOp>(loc, entryName, mainType,
                                               LLVM::Linkage::External);
  Block *entry = main.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value descriptorRaw = builder.create<LLVM::CallOp>(loc, startup, ValueRange{})
                            .getResult();
  Value descriptor =
      builder.create<LLVM::IntToPtrOp>(loc, ptrType, descriptorRaw);

  LLVM::LLVMFunctionType kernelType = kernel.getFunctionType();
  SmallVector<Value> args;
  args.reserve(kernelType.getNumParams());
  for (unsigned i = 0, e = kernelType.getNumParams(); i < e; ++i) {
    int32_t offset = 0;
    if (failed(checkedSlotOffset(module.getOperation(), argBaseOffset,
                                 argStride, i, offset)))
      return failure();

    Value slotPtr = builder.create<LLVM::GEPOp>(
        loc, ptrType, i8Type, descriptor, ArrayRef<LLVM::GEPArg>{offset});
    Value rawAddress = builder.create<LLVM::LoadOp>(loc, i64Type, slotPtr, 8);
    args.push_back(builder.create<LLVM::IntToPtrOp>(
        loc, kernelType.getParamType(i), rawAddress));
  }

  builder.create<LLVM::CallOp>(loc, kernel, args);
  Value status = createI32Constant(builder, loc, exitStatus);
  builder.create<LLVM::CallOp>(loc, exit, ValueRange{status});
  Value returnStatus = createI32Constant(builder, loc, exitStatus);
  builder.create<LLVM::ReturnOp>(loc, returnStatus);
  return success();
}

struct MaterializeBoardXDMAEntry
    : public impl::MaterializeBoardXDMAEntryBase<
          MaterializeBoardXDMAEntry> {
  using impl::MaterializeBoardXDMAEntryBase<
      MaterializeBoardXDMAEntry>::MaterializeBoardXDMAEntryBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (failed(validateOptions(module.getOperation(), xlen, argBaseOffset,
                               argStride))) {
      signalPassFailure();
      return;
    }

    FailureOr<LLVM::LLVMFuncOp> kernel =
        findKernelToWrap(module, StringRef(kernelName));
    if (failed(kernel) || failed(validateKernelSignature(*kernel))) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(&getContext());
    Location loc = (*kernel).getLoc();
    Type voidType = LLVM::LLVMVoidType::get(&getContext());
    Type startupResultType = getPointerSizedIntegerType(&getContext(), xlen);
    Type i32Type = builder.getI32Type();

    auto startupType = LLVM::LLVMFunctionType::get(startupResultType, {});
    auto exitType = LLVM::LLVMFunctionType::get(voidType, {i32Type});

    FailureOr<LLVM::LLVMFuncOp> startup = getOrCreateLLVMFuncDecl(
        module, kStartupArg, startupType, builder, loc);
    if (failed(startup)) {
      module.emitError() << "runtime declaration type mismatch for "
                         << kStartupArg;
      signalPassFailure();
      return;
    }

    FailureOr<LLVM::LLVMFuncOp> exit =
        getOrCreateLLVMFuncDecl(module, kExit, exitType, builder, loc);
    if (failed(exit)) {
      module.emitError() << "runtime declaration type mismatch for " << kExit;
      signalPassFailure();
      return;
    }

    if (failed(materializeEntry(module, *kernel, *startup, *exit,
                                StringRef(entryName), argBaseOffset, argStride,
                                exitStatus, builder))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace mlir::vortex
