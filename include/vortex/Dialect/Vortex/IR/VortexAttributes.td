#ifndef VORTEX_ATTRIBUTES
#define VORTEX_ATTRIBUTES

include "vortex/Dialect/Vortex/IR/VortexDialect.td"
include "vortex/Dialect/Vortex/IR/VortexEnums.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class Vortex_Attr<string name, list<Trait> traits = []>
    : AttrDef<Vortex_Dialect, name, traits>;

def AddressSpaceAttr : EnumAttr<Vortex_Dialect, VortexAddressSpace,
                                "address_space"> {
  let summary = "Vortex address-space attribute";
  let description = [{
    Marks a Vortex-visible memory space. This attribute is intended to appear in
    memref memory-space positions before lowering to LLVM address spaces.
  }];
  let assemblyFormat = "`<` $value `>`";
}

def ScopeAttr : EnumAttr<Vortex_Dialect, VortexScope, "scope"> {
  let summary = "Vortex synchronization scope attribute";
  let description = [{
    Identifies the convergence scope used by vortex.barrier and vortex.fence.
  }];
  let assemblyFormat = "`<` $value `>`";
}

#endif // VORTEX_ATTRIBUTES
