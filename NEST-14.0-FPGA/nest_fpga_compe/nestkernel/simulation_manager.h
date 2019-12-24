/*
 *  simulation_manager.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SIMULATION_MANAGER_H
#define SIMULATION_MANAGER_H

// C++ includes:
#include <vector>

// Includes from libnestutil:
#include "manager_interface.h"

// Includes from nestkernel:
#include "nest_time.h"
#include "nest_types.h"

// Includes from sli:
#include "dictdatum.h"
#include<iostream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/fb.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_BASEADDR0         0x43C00000

#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_AP_CTRL              0x00
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_GIE                  0x04
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_IER                  0x08
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_ISR                  0x0c
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_NUM_DATA             0x10
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_NUM_DATA             32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_STATE_DATA           0x18
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_STATE_DATA           32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM0_V_DATA 0x20
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_INPUT_STREAM0_V_DATA 32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM1_V_DATA 0x28
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_INPUT_STREAM1_V_DATA 32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM2_V_DATA 0x30
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_INPUT_STREAM2_V_DATA 32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM3_V_DATA 0x38
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_INPUT_STREAM3_V_DATA 32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_OUTPUT_FIRE_V_DATA   0x40
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_OUTPUT_FIRE_V_DATA   32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_OUTPUT_V_DATA        0x48
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_OUTPUT_V_DATA        32
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_OUTPUT_W_V_DATA      0x50
#define XIAF_PSC_EXP_PS_TOP_CTRL_BUS_BITS_OUTPUT_W_V_DATA      32

#define WriteReg(BaseAddress, RegOffset, Data) *(volatile unsigned int*)((BaseAddress) + (RegOffset)) = (Data)
#define ReadReg(BaseAddress, RegOffset) *(volatile unsigned int*)((BaseAddress) + (RegOffset))

#define INPUT_STREAM0 0x10000000
#define INPUT_STREAM1 0x11000000
#define INPUT_STREAM2 0x12000000
#define INPUT_STREAM3 0x13000000

#define QUERTER_INPUT 0x1000000
#define QUERTER_INPUT_2 0x1000000/2
#define MAX_NUM 100000
#define NUM_OTHER_M 1000
#define IAF_PSC_EXP_ID 72

#define N_THR 2
#define NUM_NEURON 100000


using namespace std;
namespace nest
{
class Node;

class SimulationManager : public ManagerInterface
{
public:
  SimulationManager();

  virtual void initialize();
  virtual void finalize();

  virtual void set_status( const DictionaryDatum& );
  virtual void get_status( DictionaryDatum& );

  /**
      check for errors in time before run
      @throws KernelException if illegal time passed
  */
  void assert_valid_simtime( Time const& );

  /*
     Simulate can be broken up into .. prepare... run.. run.. cleanup..
     instead of calling simulate multiple times, and thus reduplicating
     effort in prepare, cleanup many times.
  */

  /**
     Initialize simulation for a set of run calls.
     Must be called before a sequence of runs, and again after cleanup.
  */
  void prepare();
  /**
     Run a simulation for another `Time`. Can be repeated ad infinitum with
     calls to get_status(), but any changes to the network are undefined,
     leading serious risk of incorrect results.
  */
  void run( Time const& );
  /**
     Closes a set of runs, doing finalizations such as file closures.
     After cleanup() is called, no more run()s can be called before another
     prepare() call.
  */
  void cleanup();

  /**
   * Simulate for the given time .
   * calls prepare(); run(Time&); cleanup();
   */
  void simulate( Time const& );

  /**
   * Returns true if waveform relaxation is used.
   */
  bool use_wfr() const;

  /**
   * Get the desired communication interval for the waveform relaxation
   */
  double get_wfr_comm_interval() const;

  /**
   * Get the convergence tolerance of the waveform relaxation method
   */
  double get_wfr_tol() const;

  /**
   * Get the interpolation order of the waveform relaxation method
   */
  size_t get_wfr_interpolation_order() const;

  /**
   * Get the time at the beginning of the current time slice.
   */
  Time const& get_slice_origin() const;

  /**
   * Get the time at the beginning of the previous time slice.
   */
  Time const get_previous_slice_origin() const;

  /**
   * Precise time of simulation.
   * @note The precise time of the simulation is defined only
   *       while the simulation is not in progress.
   */
  Time const get_time() const;

  /**
   * Return true, if the SimulationManager has already been simulated for some
   * time. This does NOT indicate that simulate has been called (i.e. if
   * Simulate is called with 0 as argument, the flag is still set to false.)
   */
  bool has_been_simulated() const;

  /**
   * Reset the SimulationManager to the state at T = 0.
   */
  void reset_network();

  /**
   * Get slice number. Increased by one for each slice. Can be used
   * to choose alternating buffers.
   */
  size_t get_slice() const;

  //! Return current simulation time.
  // TODO: Precisely how defined? Rename!
  Time const& get_clock() const;

  //! Return start of current time slice, in steps.
  // TODO: rename / precisely how defined?
  delay get_from_step() const;

  //! Return end of current time slice, in steps.
  // TODO: rename / precisely how defined?
  delay get_to_step() const;

  /**
   * Independent parameters of the model.
   */
  /**
   * Independent parameters of the model.
   */



/*
 *  hw_maamp_init
 *  input: void
 *  output: fd of memeory
 */
int hw_mmap_init()
{
	int fd=0;
	fd = open("/dev/mem", O_RDWR);

	if(fd == -1) {
		printf("Open memory failed\r\n");
		return -1;
	}

	else return fd;
}
/*
 *  hw_mmap
 *  input:  physics address, map length
 * 	return: virtual address
 */

void * hw_mmap(uint32_t phy_addr, uint32_t map_len)
{

	void *virtual_addr_in;
	virtual_addr_in = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, hw_mmap_init(), (off_t)phy_addr);
	if(virtual_addr_in == MAP_FAILED)
	{
		perror("Virtual_addr_in mappong for absolute memory access failed!\n");
	}
	return virtual_addr_in;
}

/*
 *  hw_unmap
 *  input: start address, map_length
 */
void hw_unmap(void * start, uint32_t length)
{
	if(length < 0x1000) length = 0x1000;
	munmap(start, length);
}



//INIT_1
float *input00;
float *input01;
float *input02;
float *input03;



int   V_1_RefractoryCounts_[N_THR][NUM_NEURON];
float P_1_V_reset_[N_THR][NUM_NEURON];



unsigned int gid[N_THR][NUM_NEURON];
unsigned int num_g[N_THR][NUM_NEURON];

int num_neuron[N_THR];
bool init_flag[N_THR];
unsigned int num_otherg[N_THR][NUM_OTHER_M];
int num_others[N_THR];
double iaf_init[N_THR];
double iaf_update_[N_THR];
double iaf_back[N_THR];
double time_others[N_THR];
double time_neuron[N_THR];
double time_deliver[N_THR];
double time_gather;


int  get_num_neuron(const int num_tid);
bool get_init_flag(const int num_tid);
//void update_neuron(const Time& origin, const long from, const long to);
void init_sharemem_para();
void unmap_sharemem_para();

void handle_weight_ex(const int num_tid,long unsigned int gid_l,float value);
void handle_weight_in(const int num_tid,long unsigned int gid_l,float value);
void set_V_1(const int num_tid,int num,float P20_,float P11ex_,float P11in_,float P21ex_,float P21in_, float P22_,float weighted_spikes_ex_,float weighted_spikes_in_,int RefractoryCounts_);
void set_S_1(const int num_tid,int num,float i_0_,float i_1_,float i_syn_ex_,float i_syn_in_,float V_m_,int r_ref_);
void set_P_1(const int num_tid,int num, float I_e_,float Theta_,float V_reset_);

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void iaf_update(unsigned long int addr_base,int state,unsigned int num_iaf_neuron,
		unsigned int addr_input0,unsigned int addr_input1,unsigned int addr_input2,unsigned int addr_input3)
{
	      unsigned int ap_idle;
		  unsigned int ap_done;

		  unsigned long int PhysicalAddress = addr_base;
		  int map_len = 0xf8;
		  int fd = open("/dev/mem", O_RDWR);

		  unsigned char *xbase_address;
		  xbase_address = (unsigned char *)mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)PhysicalAddress);
		  if(xbase_address == MAP_FAILED)
		  {
			  perror("1:Init Mapping memory for absolute memory access failed.\n");
			  return ;
		  }

		  while(1)
		  {
			  ap_idle = ((ReadReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_AP_CTRL) >> 2) && 0x1);
			  if(ap_idle)
				  break;
		  }
		//================================
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_NUM_DATA ,num_iaf_neuron);
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_STATE_DATA ,state);
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM0_V_DATA ,addr_input0);
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM1_V_DATA ,addr_input1);
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM2_V_DATA ,addr_input2);
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_INPUT_STREAM3_V_DATA ,addr_input3);
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_OUTPUT_V_DATA ,addr_input3);
		WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_OUTPUT_W_V_DATA ,addr_input2);
                WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_OUTPUT_FIRE_V_DATA ,addr_input1+QUERTER_INPUT_2);

		//================================
		  WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_GIE, 0x0);
		  WriteReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_AP_CTRL, 0x1);//Start

		  while(1)
		  {
			  ap_done = ((ReadReg(xbase_address, XIAF_PSC_EXP_PS_TOP_CTRL_BUS_ADDR_AP_CTRL) >> 1) && 0x1);
			  if(ap_done)
				  break;
		  }

			munmap((void *)xbase_address, map_len);
			close(fd);
}


private:
  void call_update_(); //!< actually run simulation, aka wrap update_
  void update_();      //! actually perform simulation
  bool wfr_update_( Node* );
  void advance_time_();   //!< Update time to next time step
  void print_progress_(); //!< TODO: Remove, replace by logging!

  Time clock_;            //!< SimulationManager clock, updated once per slice
  delay slice_;           //!< current update slice
  delay to_do_;           //!< number of pending cycles.
  delay to_do_total_;     //!< number of requested cycles in current simulation.
  delay from_step_;       //!< update clock_+from_step<=T<clock_+to_step_
  delay to_step_;         //!< update clock_+from_step<=T<clock_+to_step_
  timeval t_slice_begin_; //!< Wall-clock time at the begin of a time slice
  timeval t_slice_end_;   //!< Wall-clock time at the end of time slice
  long t_real_;     //!< Accumunated wall-clock time spent simulating (in us)
  bool simulating_; //!< true if simulation in progress
  bool simulated_; //!< indicates whether the SimulationManager has already been
                   //!< simulated for sometime
  bool exit_on_user_signal_; //!< true if update loop was left due to signal
  // received
  bool inconsistent_state_; //!< true after exception during update_
                            //!< simulation must not be resumed
  bool print_time_;         //!< Indicates whether time should be printed during
                            //!< simulations (or not)
  bool use_wfr_;            //!< Indicates wheter waveform relaxation is used
  double wfr_comm_interval_; //!< Desired waveform relaxation communication
                             //!< interval (in ms)
  double wfr_tol_; //!< Convergence tolerance of waveform relaxation method
  long wfr_max_iterations_; //!< maximal number of iterations used for waveform
                            //!< relaxation
  size_t wfr_interpolation_order_; //!< interpolation order for waveform
                                   //!< relaxation method
};

inline Time const&
SimulationManager::get_slice_origin() const
{
  return clock_;
}

inline Time const
SimulationManager::get_time() const
{
  assert( not simulating_ );
  return clock_ + Time::step( from_step_ );
}

inline bool
SimulationManager::has_been_simulated() const
{
  return simulated_;
}

inline size_t
SimulationManager::get_slice() const
{
  return slice_;
}

inline Time const&
SimulationManager::get_clock() const
{
  return clock_;
}

inline delay
SimulationManager::get_from_step() const
{
  return from_step_;
}

inline delay
SimulationManager::get_to_step() const
{
  return to_step_;
}

inline bool
SimulationManager::use_wfr() const
{
  return use_wfr_;
}

inline double
SimulationManager::get_wfr_comm_interval() const
{
  return wfr_comm_interval_;
}

inline double
SimulationManager::get_wfr_tol() const
{
  return wfr_tol_;
}

inline size_t
SimulationManager::get_wfr_interpolation_order() const
{
  return wfr_interpolation_order_;
}


inline
int SimulationManager::get_num_neuron(const int num_tid)
{
    return num_neuron[num_tid];
}

inline
bool SimulationManager::get_init_flag(const int num_tid)
{
    return init_flag[num_tid];
}

inline void 
SimulationManager::set_V_1(const int num_tid,int num, float P20_,float P11ex_,float P11in_,float P21ex_,float P21in_, float P22_,float weighted_spikes_ex_,float weighted_spikes_in_,int RefractoryCounts_)
{
   if(num_tid==0){
   input00[4*num+1]=P20_;
   input00[4*num+2]=P11ex_;
   input00[4*num+3]=P11in_;
   input01[4*num]=P21ex_;
   input01[4*num+1]=P21in_;
   input01[4*num+2]=P22_;
   input02[4*num+2]=weighted_spikes_ex_;
   input02[4*num+3]=weighted_spikes_in_;
   V_1_RefractoryCounts_[num_tid][num]=RefractoryCounts_;}
}
inline void 
SimulationManager::set_S_1(const int num_tid, int num,float i_0_,float i_1_,float i_syn_ex_,float i_syn_in_,float V_m_,int r_ref_)
{
   if(num_tid==0){
   input02[4*num]=i_0_;
   input02[4*num+1]=i_1_;
   input03[4*num]=i_syn_ex_;
   input03[4*num+1]=i_syn_in_;
   input03[4*num+3]=V_m_;
   input03[4*num+2]=r_ref_;}
}

inline void 
SimulationManager::set_P_1(const int num_tid,int num,float I_e_,float Theta_,float V_reset_)
{

   if(num_tid==0){
   input00[4*num]=I_e_;
   P_1_V_reset_[num_tid][num]=V_reset_;
   input01[4*num+3]=Theta_;}
}

inline void SimulationManager::handle_weight_ex(const int num_tid,long unsigned int gid_l,float value)
{
  unsigned int num= gid[num_tid][gid_l];
  if(num_tid==0){
  input02[4*num+2]+=value; }
}

inline void SimulationManager::handle_weight_in(const int num_tid,long unsigned int gid_l,float value)
{
   unsigned int num= gid[num_tid][gid_l];
  if(num_tid==0){
  input02[4*num+3]+=value; }
}



inline void 
SimulationManager::init_sharemem_para(){

input00=(float *)hw_mmap(INPUT_STREAM0, QUERTER_INPUT);
input01=(float *)hw_mmap(INPUT_STREAM1, QUERTER_INPUT);
input02=(float *)hw_mmap(INPUT_STREAM2, QUERTER_INPUT);
input03=(float *)hw_mmap(INPUT_STREAM3, QUERTER_INPUT);

}

inline void 
SimulationManager::unmap_sharemem_para(){

hw_unmap(input00, QUERTER_INPUT);
hw_unmap(input01, QUERTER_INPUT);
hw_unmap(input02, QUERTER_INPUT);
hw_unmap(input03, QUERTER_INPUT);

}

}


#endif /* SIMULATION_MANAGER_H */
