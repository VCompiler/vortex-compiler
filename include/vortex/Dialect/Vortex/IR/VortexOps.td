#ifndef VORTEX_OPS
#define VORTEX_OPS

include "vortex/Dialect/Vortex/IR/VortexAttributes.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Vortex_LaunchOp : Vortex_Op<
    "launch",
    [AutomaticAllocationScope, SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Launch a Vortex-mapped execution region";
  let description = [{
    `vortex.launch` marks a region that has already been mapped from the
    pre-vortex scheduling layer onto the Vortex execution hierarchy.

    The three operands describe the number of participating `core`, `subgroup`,
    and `thread` instances. Inside the body, `vortex.core_id`,
    `vortex.subgroup_id`, and `vortex.thread_id` are used to query the current
    execution position.
  }];

  let arguments = (ins Index:$cores, Index:$subgroups, Index:$threads);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$cores `,` $subgroups `,` $threads $body attr-dict";
  let hasRegionVerifier = 1;
}

def Vortex_CoreIdOp : Vortex_Op<"core_id", [Pure]> {
  let summary = "Query the current core id";
  let description = [{
    Returns the current core id inside a `vortex.launch` region.
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Vortex_SubgroupIdOp : Vortex_Op<"subgroup_id", [Pure]> {
  let summary = "Query the current subgroup id";
  let description = [{
    Returns the current subgroup id inside a `vortex.launch` region.
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Vortex_ThreadIdOp : Vortex_Op<"thread_id", [Pure]> {
  let summary = "Query the current thread id within a subgroup";
  let description = [{
    Returns the current thread id inside a `vortex.launch` region.
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Vortex_LocalAllocOp : Vortex_Op<"local_alloc"> {
  let summary = "Allocate Vortex per-core local memory";
  let description = [{
    Allocates a memref in Vortex local memory. The result memref must use
    `#vortex.address_space<local>` as its memory space.
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs Res<AnyMemRef, "",
                          [MemAlloc<AutomaticAllocationScopeResource,
                                    0, FullEffect>]>:$buffer);
  let assemblyFormat = "`(` $dynamicSizes `)` attr-dict `:` type($buffer)";
  let hasVerifier = 1;
}

def Vortex_BarrierOp : Vortex_Op<"barrier"> {
  let summary = "Synchronize Vortex execution at a given scope";
  let description = [{
    Synchronizes execution and local visibility at the specified Vortex scope.
  }];

  let arguments = (ins ScopeAttr:$scope);
  let assemblyFormat = "$scope attr-dict";
}

def Vortex_FenceOp : Vortex_Op<"fence"> {
  let summary = "Enforce ordering and visibility at a Vortex scope";
  let description = [{
    Models an explicit memory-ordering boundary before lowering to the target
    barrier/fence intrinsic layer.
  }];

  let arguments = (ins ScopeAttr:$scope);
  let assemblyFormat = "$scope attr-dict";
}

def Vortex_YieldOp : Vortex_Op<"yield", [Pure, Terminator,
                                          ParentOneOf<["LaunchOp"]>]> {
  let summary = "Terminate a vortex.launch region";
  let assemblyFormat = "attr-dict";
}

#endif // VORTEX_OPS
