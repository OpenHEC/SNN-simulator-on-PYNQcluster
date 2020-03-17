# SNN-simulator-on-PYNQcluster
Openhec lab，Jiangnan university
## Introduction
A Spiking neural network simulator NEST base on FPGA‘s cluster（LIF NEURON）  
* Spiking neuron network simulator NEST  
* SNN image classification  
* Neuron computing and STDP accelerator base on FPGA  
* MPI communication between PYNQ  

This system consists of PYNN like brain framework, NEST simulator, PYNQ framework, FPGA neurons and STDP hardware modules. As shown in picture, the top-level application design language is Python. With the assistance of the PYNN architecture, the NEST simulator is called. Various commands are interpreted by the python interpreter and the SLI interpreter, and then enter the NEST kernel. The underlying network creation according to various commands includes neuron creation, synapse connection creation, simulation time setting, etc.

On this basis, we designed FPGA neuron acceleration module and FPGA STDP synapse acceleration module, and provided acceleration modules for different computation-intensive points according to the network topology and computing requirements.  
<div align=center><img src="https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/overview.png"/></div>

The general platform of this project integrates 16 PYNQ boards, and the board-level connection follows the TCP/IP protocol. The PYNQ-Z2 development board is based on the ZYNQ XC7Z020 FPGA and is equipped with Ethernet, HDMI input/output, MIC input, audio output, Arduino interface, Raspberry Pi interface, 2 Pmods, user LEDs, buttons and switches.  
<div align=center><img src="https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/1.jpg"/></div>
## Quick Start
 * If you would like to use the Jupyter notebook on PYNQ to perform the simulation on NEST, please click [here](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/readme.md).
 * Petalinux  
 https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/tree/master/petalinux
 * Install NEST-14.0-FPGA  
   * In this project, we use PYNQ-Z2 v2.5 PYNQ image. (If you use the old version of the image file, it should also work.)
   * Installing PyNN requires: 
   ```
    sudo apt-get install -y \  
    cython \  
    libgsl-dev \  
    libltdl-dev \  
    libncurses-dev \  
    libreadline-dev \  
    python-all-dev \  
    python-numpy \  
    python-scipy \  
    python-matplotlib \  
    python-nose \  
    openmpi-bin \  
    libopenmpi-dev
    ```
   * Configure NEST:  
    ```
    cd NEST-14.0-FPGA  
    cmake -Dwith-python=3 -DCMAKE_INSTALL_PREFIX:PATH=./ </path/to/NEST/src>  
    ```
   * Compile and install NEST:  
    ```
    make  
    make install  
    make installcheck  
    source </path/to/nest_install_dir>/bin/nest_vars.sh  
   ```
   * References: https://nest-simulator.readthedocs.io/en/stable/installation/linux_install.html
 * Install PYNN  
   * Installing PyNN requires:    
    ```
    Python (version 2.7, 3.3-3.7)  
    a recent version of the NumPy package  
    the lazyarray package  
    the Neo package (>= 0.5.0)  
    at least one of the supported simulators: e.g. NEURON, NEST, or Brian.  
   ```
   * Optional dependencies are:    
    ```
    mpi4py (if you wish to run distributed simulations using MPI)  
    either Jinja2 or Cheetah (templating engines)  
    the CSA library  
    ```
    * Install PyNN:  
    ```
    pip install pyNN    
    ```
    * References:http://neuralensemble.org/docs/PyNN/installation.html  
 * Run snn_object:
    * 1 PYNQ node:
     ```
    cd snn_object
    python3 dump-c1-spikes.py --training-dir airplanes_10_6 --dataset-label train
    ```
    * 8 PYNQ node:
    ```
    mpirun -n 8 -machinefile ./machinefile python3 dump-c1-spikes.py --training-dir airplanes_10_6 --dataset-label train
    ````
    * 1 PYNQ node run with jupyter notebook:
    dump_c1_spikes.ipynb  
   others commond: https://github.com/roberthangu/snn_object_recognition 
## Repo organization
The repo is organized as follows:
 * snn_object：Describes a new biologically plausible mechanism for generating intermediate-level visual representations using an          unsupervised learning scheme.
   * iaf_psc_exp.bin：LIF NEURON bitstream
 * NEST-14.0-FPGA：implemention Spiking neural network simulator NEST on FPGA cluster  
 * hls: NEST LIF Neuron accelerator implemented in vivado HLS 2018.2.  
 * vivado: creating vivado project to get block_design.tcl and bitstream.
 * NEST_PYNQ_Jupyter: An example of image classification based on Jupyter Notebook on PYNQ.
## References
### NEST  
NEST is a simulator for spiking neural network models, ideal for networks of any size.  
NEST Simulation:http://www.nest-simulator.org    
NEST-14.0 github：https://github.com/nest/nest-simulator/releases/tag/v2.14.0  
### pyNN  
A Python package for simulator-independent specification of neuronal network models.  
PyNN :http://neuralensemble.org/PyNN/  
### NEST base on GPU  
Implemention Spiking neural network simulator NEST on multi-GPU and distributed GPU  
https://github.com/pnquanganh/opencl-nest  
Nguyen Q A P, Andelfinger P, Cai W, et al. Transitioning Spiking Neural Network Simulators to Heterogeneous Hardware[C]//Proceedings of the 2019 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation. ACM, 2019: 115-126.
### SNN simulator base on FPGA
A.Podobas,S.Matsuoka,Luk Wayne.Designing and accelerating spiking neural networks using OpenCL for FPGAs[C].//International   Conference on Field Programmable Technology (ICFPT).Melbourne,VIC, Australia: IEEE,2017.  
### snn_object_recognition on PYNN-NEST  
Describes a new biologically plausible mechanism for generating intermediate-level visual representations using an unsupervised learning   scheme.  
https://github.com/roberthangu/snn_object_recognition  
Masquelier, Timothée, Thorpe S J. Unsupervised Learning of Visual Features through Spike Timing Dependent Plasticity[J].PLoS     Computational Biology, 2007, 3(2):e31.  
### others
Schuman C D , Potok T E , Patton R M , et al. A Survey of Neuromorphic Computing and Neural Networks in Hardware
