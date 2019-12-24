# CMake generated Testfile for 
# Source directory: /home/xilinx/nest_fpga_compe/testsuite/selftests
# Build directory: /home/xilinx/nest_fpga_compe/testsuite/selftests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(selftests/test_pass.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_pass.sli")
add_test(selftests/test_goodhandler.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_goodhandler.sli")
add_test(selftests/test_lazyhandler.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_lazyhandler.sli")
add_test(selftests/test_fail.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_fail.sli")
set_tests_properties(selftests/test_fail.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_stop.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_stop.sli")
set_tests_properties(selftests/test_stop.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_badhandler.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_badhandler.sli")
set_tests_properties(selftests/test_badhandler.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_pass_or_die.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_pass_or_die.sli")
set_tests_properties(selftests/test_pass_or_die.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_assert_or_die_b.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_assert_or_die_b.sli")
set_tests_properties(selftests/test_assert_or_die_b.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_assert_or_die_p.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_assert_or_die_p.sli")
set_tests_properties(selftests/test_assert_or_die_p.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_fail_or_die.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_fail_or_die.sli")
set_tests_properties(selftests/test_fail_or_die.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_crash_or_die.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_crash_or_die.sli")
set_tests_properties(selftests/test_crash_or_die.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_failbutnocrash_or_die_crash.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_failbutnocrash_or_die_crash.sli")
set_tests_properties(selftests/test_failbutnocrash_or_die_crash.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_failbutnocrash_or_die_pass.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_failbutnocrash_or_die_pass.sli")
set_tests_properties(selftests/test_failbutnocrash_or_die_pass.sli PROPERTIES  WILL_FAIL "TRUE")
add_test(selftests/test_passorfailbutnocrash_or_die.sli "/usr/nest/bin/nest" "/usr/nest/share/doc/nest/selftests/test_passorfailbutnocrash_or_die.sli")
set_tests_properties(selftests/test_passorfailbutnocrash_or_die.sli PROPERTIES  WILL_FAIL "TRUE")
