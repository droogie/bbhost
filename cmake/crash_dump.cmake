# Also included by the dependency-free native Windows test project.
set(crash_dump_root "${CMAKE_CURRENT_LIST_DIR}/..")
add_executable(crash_dump_helper "${crash_dump_root}/tools/crash_dump_helper.cpp")
target_include_directories(crash_dump_helper PRIVATE "${crash_dump_root}/src")
target_compile_features(crash_dump_helper PRIVATE cxx_std_17)
target_link_libraries(crash_dump_helper PRIVATE dbghelp)
target_link_options(crash_dump_helper PRIVATE -municode)

add_executable(crash_dump_fixture "${crash_dump_root}/tests/crash_dump_fixture.cpp"
  "${crash_dump_root}/src/host/win_crash.cpp" "${crash_dump_root}/src/host/crash_dump.cpp")
target_include_directories(crash_dump_fixture PRIVATE "${crash_dump_root}/src")
target_compile_features(crash_dump_fixture PRIVATE cxx_std_17)
target_compile_definitions(crash_dump_fixture PRIVATE BBHOST_DUMP_WAIT_MS=5000)
add_dependencies(crash_dump_fixture crash_dump_helper)

add_executable(crash_dump_slow_helper "${crash_dump_root}/tools/crash_dump_helper.cpp")
target_include_directories(crash_dump_slow_helper PRIVATE "${crash_dump_root}/src")
target_compile_features(crash_dump_slow_helper PRIVATE cxx_std_17)
target_compile_definitions(crash_dump_slow_helper PRIVATE BBHOST_DUMP_TEST_DELAY_MS=30000)
target_link_libraries(crash_dump_slow_helper PRIVATE dbghelp)
target_link_options(crash_dump_slow_helper PRIVATE -municode)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND NOT CMAKE_CROSSCOMPILING)
  add_test(NAME crash_dump_test COMMAND "${Python3_EXECUTABLE}"
    "${crash_dump_root}/tests/crash_dump_test.py"
    --fixture "$<TARGET_FILE:crash_dump_fixture>"
    --helper "$<TARGET_FILE:crash_dump_helper>"
    --slow-helper "$<TARGET_FILE:crash_dump_slow_helper>")
  set_tests_properties(crash_dump_test PROPERTIES TIMEOUT 180)
endif()
