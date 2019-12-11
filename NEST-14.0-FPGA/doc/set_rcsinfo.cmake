# set_rcsinfo.cmake.in
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

execute_process(
  COMMAND /home/xilinx/nest_fpga_compe/extras/create_rcsinfo.sh /home/xilinx/nest_fpga_compe /tmp
  OUTPUT_VARIABLE RCSINFO
  OUTPUT_STRIP_TRAILING_WHITESPACE
)


execute_process(
  COMMAND /bin/sed -i "" -e "/^PROJECT_NUMBER/ s/=.*/= 2.14.0, ${RCSINFO} /" /home/xilinx/nest_fpga_compe/doc/normaldoc.conf
  COMMAND /bin/sed -i "" -e "/^PROJECT_NUMBER/ s/=.*/= 2.14.0, ${RCSINFO} /" /home/xilinx/nest_fpga_compe/doc/fulldoc.conf
)
