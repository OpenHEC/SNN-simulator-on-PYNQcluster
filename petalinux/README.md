# PetaLinux
Ubuntu 16.04 with Vivado v2018.2 and PetaLinux 2018.2
## Prepare for Petalinux
First, make sure that you have installed PetaLinux and Vivado in Ubuntu 16.04(See the official Xilinx manual and User Guide or just google or bing it) . After you have installed PetaLinux and Vivado, you should source related settings.sh in order to use petalinux and cross-compiler tool-chain in shell. like these two cmds: source /opt/pkg/petalinux/settings.sh and source /opt/Xilinx/Vivado/2018.2/settings64.sh. Then, copy two files (hardware description file design_1_wrapper.hdf and bitstream file design_1_wrapper.bit that generated from Vivado project)to one directory.  
## Create PetaLinux Project and modify the Device-tree to reserve memory
Use petalinux-create cmd to create one petalinux project. If you dont know how to create one project, you can type **petalinux-create -h or --help** to get help and more informations. Here, I just type **petalinux-create -t project -n nest_laf --template zynq** to create a petalinux project named nest_laf. Then, **cd nest_laf/project-spec/meta-user/recipes-bsp/device-tree/files/** to find the **system-user.dtsi** and modify it like below:
```
/include/ "system-conf.dtsi"
/ {
	reserved-memory {
		#address-cells = <1>;
		#size-cells = <1>;
		ranges;

		reserved: buffer@0x10000000 {
			 no-map;
			 reg = <0x10000000 0x10000000>;
		};
	};

	reserved-driver@0 {
		compatible = "xlnx,reserved-memory";
		memory-region = <&reserved>;
	};
	
};
```
reference:https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841683/Linux+Reserved+Memory  
## Config project with .hdf file
Third, back to the project directory nest_iaf/, and use type this cmd to initilize your project: **petalinux-config --get-hw-description DIR_where_you_put_the_design_1_wrapper.hdf**. Then, it will come to one menu like below:  
Here, I like to set rootfs from SD card. So that, the files that you used can be saved in SD card when you power-off it. And, then save configuraiton and exit.   
## Further config or build project
If you have other configuraitons, you can use **petalinux-config** or **petalinux-config -c XXX** to further config this project. More informations are available in petalinux-config -h. After configuraiotn, you can type **petalinux-build** to build the whole project. After that, type **petalinux-package -boot --fsbl image/linux/zynq_fsbl.elf --fpga --u-boot --force** to generate BOOT.BIN. Finally, you will get three files for petalinux: BOOT.BIN, image.ub and rootfs.cpio.  
## Partition SD Card and unzip rootfs
Just use Disks tool in ubuntu to partition SD card into two file systems: one FAT and one EXT4. Copy BOOT.BIN and image.ub into FAT file system and copy rootfs.cpio into EXT4 fs.(If you meet permission denied, use sudo). In EXT4 fs, type cmd **sudo pax -rvf rootfs.cpio** to unzip the rootfs. Then, umount EXT4 and FAT. Here, the Petalinux has been implemented.  
