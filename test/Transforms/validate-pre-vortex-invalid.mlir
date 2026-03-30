// RUN: not vx-opt %s --allow-unregistered-dialect --vortex-validate-pre-vortex 2>&1 | FileCheck %s

func.func @bad() {
  "test.unsupported"() : () -> ()
  return
}

// CHECK: error:
// CHECK-SAME: pre-vortex IR does not allow dialect 'test'
