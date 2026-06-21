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


def Vortex_PredicatedOp : Vortex_Op<"predicated", [SingleBlock]> {
  let summary = "Execute a region under an explicit lane predicate";
  let description = [{
    `vortex.predicated` materializes a short lane-varying single-sided branch
    as an explicit predicated region. The region must be nested under a
    `vortex.launch`, must not take block arguments, and is terminated by
    `vortex.yield`.
  }];

  let arguments = (ins I1:$pred);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$pred $body attr-dict";
  let hasRegionVerifier = 1;
}

def Vortex_DivergentIfOp : Vortex_Op<"divergent_if", [SingleBlock]> {
  let summary = "Execute an explicit divergent SIMT branch";
  let description = [{
    `vortex.divergent_if` preserves both sides of a lane-varying branch in a
    structured Vortex operation. Both regions must be nested under a
    `vortex.launch`, must not take block arguments, and are terminated by
    `vortex.yield`.
  }];

  let arguments = (ins I1:$pred);
  let regions = (region SizedRegion<1>:$thenRegion,
                         SizedRegion<1>:$elseRegion);
  let assemblyFormat = "$pred $thenRegion `else` $elseRegion attr-dict";
  let hasRegionVerifier = 1;
}

def Vortex_TMaskOp : Vortex_Op<"tmask"> {
  let summary = "Read the current active thread mask";
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Vortex_TmcOp : Vortex_Op<"tmc"> {
  let summary = "Set the current active thread mask";
  let arguments = (ins Index:$mask);
  let assemblyFormat = "$mask attr-dict";
}

def Vortex_PredOp : Vortex_Op<"pred"> {
  let summary = "Apply a positive predicate against a restore mask";
  let arguments = (ins I1:$pred, Index:$mask);
  let assemblyFormat = "$pred `,` $mask attr-dict";
}

def Vortex_PredNOp : Vortex_Op<"pred_n"> {
  let summary = "Apply a negative predicate against a restore mask";
  let arguments = (ins I1:$pred, Index:$mask);
  let assemblyFormat = "$pred `,` $mask attr-dict";
}

def Vortex_SplitOp : Vortex_Op<"split"> {
  let summary = "Split divergent control flow on a positive predicate";
  let arguments = (ins I1:$pred);
  let results = (outs Index:$stackPtr);
  let assemblyFormat = "$pred attr-dict `:` type($stackPtr)";
}

def Vortex_SplitNOp : Vortex_Op<"split_n"> {
  let summary = "Split divergent control flow on a negative predicate";
  let arguments = (ins I1:$pred);
  let results = (outs Index:$stackPtr);
  let assemblyFormat = "$pred attr-dict `:` type($stackPtr)";
}

def Vortex_JoinOp : Vortex_Op<"join"> {
  let summary = "Join divergent control flow using a split stack pointer";
  let arguments = (ins Index:$stackPtr);
  let assemblyFormat = "$stackPtr attr-dict";
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
                                          ParentOneOf<[
                                            "LaunchOp",
                                            "PredicatedOp",
                                            "DivergentIfOp"
                                          ]>]> {
  let summary = "Terminate a structured Vortex region";
  let assemblyFormat = "attr-dict";
}

#endif // VORTEX_OPS
