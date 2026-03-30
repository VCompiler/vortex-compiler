#ifndef VORTEX_DIALECT
#define VORTEX_DIALECT

include "mlir/IR/OpBase.td"

def Vortex_Dialect : Dialect {
  let name = "vortex";
  let cppNamespace = "::mlir::vortex";
  let summary = "Vortex target-specific GPU dialect";
  let description = [{
    The Vortex dialect carries the target-specific execution, memory, and
    synchronization semantics that sit between the high-level pre-vortex MLIR
    layer and the LLVM lowering path for the existing Vortex backend.
  }];

  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral getKernelAttrName() {
      return "vortex.kernel";
    }

    void registerAttributes();
  }];
}

class Vortex_Op<string mnemonic, list<Trait> traits = []>
    : Op<Vortex_Dialect, mnemonic, traits>;

#endif // VORTEX_DIALECT
