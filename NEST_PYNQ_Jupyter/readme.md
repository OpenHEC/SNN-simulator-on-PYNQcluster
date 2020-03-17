# Run the NEST on Jupyter Notebook in PYNQ
If you want to run NEST on Jupyter Notebook of PYNQ, you can refer to the following steps to execute. An example of image classification is run below.

In this project, we use PYNQ-Z2 v2.5 PYNQ image. 
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
Enter jupyter notebook，create a new file.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/2.png)
### Install dependent libraries
    !pip3 install PyNN

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/3.png)

Install cython higher than 0.28.5,these will take a long time.

    !pip3 install cython==0.28.5
    !pip3 install scikit-learn

### Download file
    !git clone https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster.git
    
### snn_object_recognition
Enter NEST_PYNQ_Jupyter -> snn_object_recognition.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/7.png)

In these folder , you can use others images , make sure the image size is constant.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/8.png)

The labels of the images should also be changed.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/9.png)

## Picture classification
Open dump-c1-spikes_train.ipynb, copy the nest installation path </nest/install/path/> to here. 

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/10.png)

You can change these parameters, make sure path is right，and then run. This will take a lot of time.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/11.png)

* ll_** error occurs and changes ll_** to hl_** in pynn's error file path. PyNN may have some errors, which can be found by going to the NEST official website and retrieving the error information.

Run the dump-c1-spikes_validation.ipynb using a similar operation. Finally, the following two files are generated

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/12.png)

Open learn-features.ipynb, set the path to the generated c1_dumpfile, run this file.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/13.png)

At the end of the run, it will be generated weights in S2_weights folder.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/14.png)

Next, open dump-c2-spikes.ipynb, change corresponding file path, run this file.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/15.png)

The last step, open classify-images.ipynb, change corresponding file path, run this file. At the end of run, you can get the result. At the same time, the corresponding file will be generated in log_final folder.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/17.png)

This example only uses 20 images as a demonstration, and you can add images to the dataset for better results.


## More Example
If you want to run more NEST examples in PYNQ, you can download them in https://nest-simulator.readthedocs.io/en/stable/examples/index.html. Add the NEST file path in the references section of these examples and it will work.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/18.png)

This is the result of the hpc_benchmark.ipynb.

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/NEST_PYNQ_Jupyter/image/19.png)
