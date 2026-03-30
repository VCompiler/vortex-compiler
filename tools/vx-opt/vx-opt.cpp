#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "vortex/InitAllDialects.h"
#include "vortex/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::vortex::registerVortexPassesAndPipelines();

  mlir::DialectRegistry registry;
  mlir::vortex::registerVortexDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Vortex pre-dialect optimizer driver\n",
                        registry));
}
