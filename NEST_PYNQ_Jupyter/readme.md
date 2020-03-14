# Run the NEST on Jupyter notebook in PYNQ
If you want to run NEST on Jupyter notebook of PYNQ, you can refer to the following steps to execute.
## Install NEST
### Install NEST requires
    sudo apt-get install -y cython libgsl-dev libltdl-dev libncurses-dev libreadline-dev python3-all-dev python3-numpy python3-scipy python3-matplotlib python3-nose openmpi-bin libopenmpi-dev
### Download
In this project, we use nest-simulator-2.14, you can download it at https://github.com/nest/nest-simulator/releases/tag/v2.14.0
### Unpack the tarball
    tar -xzvf nest-simulator-2.14.tar.gz
### Create a build directory
    mkdir nest
### Configure NEST
    cd nest-simulator-2.14 
    cmake -Dwith-python=3 -DCMAKE_INSTALL_PREFIX:PATH=</nest/install/path/> ./
/install/path is very important, we will use it later.
### Compile and install NEST
    make
    make install
### Add environment variables
    source </path/to/nest_install_dir>/bin/nest_vars.sh
### In the terminal type
    python3
    import nest
The following picture appears to prove that the installation was successful.
![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/1.png)
* References: https://nest-simulator.readthedocs.io/en/stable/installation/linux_install.html
## File introduction
### Create a new file
Enter jupyter notebook，create a new file

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/2.png)
### Install dependent libraries
    !pip3 install PyNN

    ![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/3.png)

Install cython higher than 0.28.5

    !pip3 install cython==0.28.5
    !pip3 install scikit-learn
    
These will take a long time.
### Download file
    !git clone https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster.git
    
Enter snn_object_recognition folder.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/7.png)
