### Install NEST requires:
    sudo apt-get install -y cython libgsl-dev libltdl-dev libncurses-dev libreadline-dev python3-all-dev python3-numpy python3-scipy python3-matplotlib python3-nose openmpi-bin libopenmpi-dev
In this project, we use nest-simulator-2.14, you can download it at https://github.com/nest/nest-simulator/releases/tag/v2.14.0
### Unpack the tarball
    tar -xzvf nest-simulator-2.14.tar.gz
### Create a build directory:
    mkdir nest
### Configure NEST:
    cd nest-simulator-2.14 
    cmake -Dwith-python=3 -DCMAKE_INSTALL_PREFIX:PATH=</nest/install/path/> ./
(/install/path is very important, we will use it later)
### Compile and install NEST:
    make
    make install
### Add environment variables
    source </path/to/nest_install_dir>/bin/nest_vars.sh
### In the terminal type:
    python3
  Once in Python you can type:
    import nest
  The following picture appears to prove that the installation was successful.
