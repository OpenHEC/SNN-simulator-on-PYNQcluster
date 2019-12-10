# Install script for directory: /home/xilinx/nest_fpga_compe

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/nest")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/nest" TYPE FILE FILES
    "/home/xilinx/nest_fpga_compe/LICENSE"
    "/home/xilinx/nest_fpga_compe/README.md"
    "/home/xilinx/nest_fpga_compe/NEWS"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/xilinx/nest_fpga_compe/doc/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/examples/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/extras/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/lib/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/libnestutil/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/librandom/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/models/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/sli/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/nest/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/nestkernel/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/precise/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/testsuite/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/topology/cmake_install.cmake")
  include("/home/xilinx/nest_fpga_compe/pynest/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/xilinx/nest_fpga_compe/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
