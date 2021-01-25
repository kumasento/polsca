configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  MAIN_CONFIG
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  )

set(PHISM_TEST_DEPENDS
  FileCheck count not
  phism-opt
  )

add_lit_testsuite(check-phism "Running the Phism regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PHISM_TEST_DEPENDS}
  )
set_target_properties(check-phism PROPERTIES FOLDER "Tests")

add_lit_testsuites(PHISM ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${PHISM_TEST_DEPS}
)
