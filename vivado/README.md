## Vivado block design
This reop is about creating vivado project to get block_design.tcl and bitstream.Follow below step:   

1.create a new project and select pynq-z2 as your board  
2.import ip iaf_psc_exp_ps_top and add ip.  
3.connect ip follow the picture block_design.png.  

![image](https://github.com/OpenHEC/SNN-simulator-on-PYNQcluster/blob/master/vivado/block_design.png)  

4.generate bitstream.  
5.export block_design.  
6.get the block_design design_1.tcl and bitstream design_1_wrapper.bit for pynq (existed in generated_demo).  
7.change two files to the same name, like nest.tcl and nest.bit for pynq.  
