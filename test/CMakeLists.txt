configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(VORTEX_TEST_DEPENDS
  FileCheck
  not
  vx-opt
)

add_lit_testsuite(check-vortex "Running the Vortex regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${VORTEX_TEST_DEPENDS}
)
set_target_properties(check-vortex PROPERTIES FOLDER "Tests")

add_lit_testsuites(VORTEX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${VORTEX_TEST_DEPENDS}
)
