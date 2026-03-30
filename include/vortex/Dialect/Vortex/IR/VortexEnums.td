#ifndef VORTEX_ENUMS
#define VORTEX_ENUMS

include "mlir/IR/EnumAttr.td"

def VortexAddressSpace : I32EnumAttr<
    "AddressSpace",
    "Vortex memory spaces used when lowering into target-specific storage.",
    [
      I32EnumAttrCase<"Global", 0, "global">,
      I32EnumAttrCase<"Local", 1, "local">,
      I32EnumAttrCase<"Private", 2, "private">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::vortex";
}

def VortexScope : I32EnumAttr<
    "Scope",
    "Execution scopes supported by Vortex synchronization operations.",
    [
      I32EnumAttrCase<"Subgroup", 0, "subgroup">,
      I32EnumAttrCase<"Core", 1, "core">
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::vortex";
}

#endif // VORTEX_ENUMS
