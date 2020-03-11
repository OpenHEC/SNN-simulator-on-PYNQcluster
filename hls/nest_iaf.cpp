#include "ap_int.h"
#include "stdint.h"
#include "hls_stream.h"
#include <string.h>
///AXI
#define AXI_SIZE 64
#define AXI_SIZE_32 32
#define NUM_NEURON 20
#define NUM_NEURON_NEST 48904
#define MAX_FIRE 50000

#define round_div_upper(x, y) 		((int32_t)((x + y - 1 ) / y))
#define N 20
typedef float 				float32_t;
typedef short 				int16_t;
typedef unsigned short 		uint16_t;
#define VALUE_BITS 32
union float32_uint32_c {
	uint32_t  u32data;
	float32_t f32data;
};

INLINE uint32_t float32_to_uint32(float32_t value) {
	float32_uint32_c c;
	c.f32data = value;
	return c.u32data;
}

INLINE float32_t uint32_to_float32(uint32_t value) {
	float32_uint32_c c;
	c.u32data = value;
	return c.f32data;
}

INLINE uint64_t float32_to_uint64(float32_t value1, float32_t value2) {
	union float32_uint64_c {
		uint64_t  u64data;
		struct {
			float32_t f32data1;
			float32_t f32data2;
		};
	} c;
	c.f32data1 = value1;
	c.f32data2 = value2;
	return c.u64data;
}

INLINE uint64_t uint32_to_uint64(float32_t value1, float32_t value2) {
	union uint32_to_uint64 {
		uint64_t  u64data;
		struct {
			uint32_t u32data1;
			uint32_t u32data2;
		};
	} c;
	c.u32data1 = value1;
	c.u32data2 = value2;
	return c.u64data;
}

void iaf_psc_exp_ps_update(
				 int num_neuron,
				 float fire[MAX_FIRE],
		         float S_1_i_0_[NUM_NEURON],
				 float S_1_i_1_[NUM_NEURON],
				 float S_1_i_syn_ex_[NUM_NEURON],
				 float S_1_i_syn_in_[NUM_NEURON],
				 float S_1_V_m_[NUM_NEURON],
				 float S_1_r_ref_[NUM_NEURON],
				 float V_1_weighted_spikes_ex_[NUM_NEURON],
				 float V_1_weighted_spikes_in_[NUM_NEURON],

				 float P_1_I_e_[NUM_NEURON],
				 //inital
				 float V_1_P20_[NUM_NEURON],         //!< Time resolution [ms]
				 float V_1_P11ex_[NUM_NEURON], //!< Refractory time in steps
				 float V_1_P11in_[NUM_NEURON],    //!< exp(-h/tau_m) - 1
				 float V_1_P21ex_[NUM_NEURON],   //!< exp(-h/tau_ex) - 1
				 float V_1_P21in_[NUM_NEURON],   //!< exp(-h/tau_in) - 1
				 float V_1_P22_[NUM_NEURON],
				 float P_1_Theta_[NUM_NEURON],
				 ap_uint<AXI_SIZE> *input_stream0,
				 ap_uint<AXI_SIZE> *input_stream1,
				 ap_uint<AXI_SIZE> *input_stream2,
				 ap_uint<AXI_SIZE> *input_stream3,
				 ap_uint<AXI_SIZE> *output_fire,
				 ap_uint<AXI_SIZE> *output,
				 ap_uint<AXI_SIZE> *output_w
                 )
{
	ap_uint<AXI_SIZE> temp1[2*N];
	ap_uint<AXI_SIZE> temp2[2*N];
	ap_uint<AXI_SIZE> temp3[2*N];
	ap_uint<AXI_SIZE> temp4[2*N];
	ap_uint<AXI_SIZE> tempo[2*N];
	ap_uint<AXI_SIZE> tempfire[MAX_FIRE];
	ap_uint<AXI_SIZE> tempow[2*N];
	int fire_count=0;

	for(int num=0;num<round_div_upper(num_neuron, N);num++){
#pragma HLS PIPELINE

	memcpy(temp1,(ap_uint<AXI_SIZE> *)(input_stream0+2*num*N),2*N*sizeof(ap_uint<AXI_SIZE>));
	memcpy(temp2,(ap_uint<AXI_SIZE> *)(input_stream1+2*num*N),2*N*sizeof(ap_uint<AXI_SIZE>));
	memcpy(temp3,(ap_uint<AXI_SIZE> *)(input_stream2+2*num*N),2*N*sizeof(ap_uint<AXI_SIZE>));
	memcpy(temp4,(ap_uint<AXI_SIZE> *)(input_stream3+2*num*N),2*N*sizeof(ap_uint<AXI_SIZE>));

	for(int num_com=0;num_com<N;num_com++)
	{
		ap_uint<VALUE_BITS> temp11=temp1[2*num_com].range(31,0);
		ap_uint<VALUE_BITS> temp12=temp1[2*num_com].range(63,32);
		ap_uint<VALUE_BITS> temp13=temp1[2*num_com+1].range(31,0);
		ap_uint<VALUE_BITS> temp14=temp1[2*num_com+1].range(63,32);

		ap_uint<VALUE_BITS> temp21=temp2[2*num_com].range(31,0);
		ap_uint<VALUE_BITS> temp22=temp2[2*num_com].range(63,32);
		ap_uint<VALUE_BITS> temp23=temp2[2*num_com+1].range(31,0);
		ap_uint<VALUE_BITS> temp24=temp2[2*num_com+1].range(64,32);

		ap_uint<VALUE_BITS> temp31=temp3[2*num_com].range(31,0);
		ap_uint<VALUE_BITS> temp32=temp3[2*num_com].range(63,32);
		ap_uint<VALUE_BITS> temp33=temp3[2*num_com+1].range(31,0);
		ap_uint<VALUE_BITS> temp34=temp3[2*num_com+1].range(63,32);

		ap_uint<VALUE_BITS> temp41=temp4[2*num_com].range(31,0);
		ap_uint<VALUE_BITS> temp42=temp4[2*num_com].range(63,32);
		ap_uint<VALUE_BITS> temp43=temp4[2*num_com+1].range(31,0);
		ap_uint<VALUE_BITS> temp44=temp4[2*num_com+1].range(63,32);

		P_1_I_e_[num_com]=uint32_to_float32(temp11.to_uint());
		V_1_P20_[num_com]=uint32_to_float32(temp12.to_uint());
		V_1_P11ex_[num_com]=uint32_to_float32(temp13.to_uint());
		V_1_P11in_[num_com]=uint32_to_float32(temp14.to_uint());

		V_1_P21ex_[num_com]=uint32_to_float32(temp21.to_uint());
		V_1_P21in_[num_com]=uint32_to_float32(temp22.to_uint());
		V_1_P22_[num_com]=uint32_to_float32(temp23.to_uint());
		P_1_Theta_[num_com]=uint32_to_float32(temp24.to_uint());

		S_1_i_0_[num_com]=uint32_to_float32(temp31.to_uint());
		S_1_i_1_[num_com]=uint32_to_float32(temp32.to_uint());
		V_1_weighted_spikes_ex_[num_com]=uint32_to_float32(temp33.to_uint());
		V_1_weighted_spikes_in_[num_com]=uint32_to_float32(temp34.to_uint());

		S_1_i_syn_ex_[num_com]=uint32_to_float32(temp41.to_uint());
		S_1_i_syn_in_[num_com]=uint32_to_float32(temp42.to_uint());
		S_1_r_ref_[num_com]=uint32_to_float32(temp43.to_uint());
		S_1_V_m_[num_com]=uint32_to_float32(temp44.to_uint());

	}

	for(int num_com=0;num_com<N;num_com++){
	#pragma HLS UNROLL

		float S_1_i_syn_ex_c=S_1_i_syn_ex_[num_com];
		float S_1_i_syn_in_c=S_1_i_syn_in_[num_com];
		float S_1_r_ref_c=S_1_r_ref_[num_com];
		float S_1_i_1_c=S_1_i_1_[num_com];
		S_1_i_syn_ex_c = S_1_i_syn_ex_c+V_1_weighted_spikes_ex_[num_com];
		S_1_i_syn_in_c = S_1_i_syn_in_c+V_1_weighted_spikes_in_[num_com];

		if (S_1_r_ref_c  == 0 ) // neuron not refractory, so evolve V
		{
		  float temp_vm=S_1_V_m_[num_com] * V_1_P22_[num_com];
		  float temp_iex=S_1_i_syn_ex_c* V_1_P21ex_[num_com];
		  float temp_iin=S_1_i_syn_in_c * V_1_P21in_[num_com];
		  float temp_ie=( P_1_I_e_[num_com] + S_1_i_0_[num_com] ) * V_1_P20_[num_com];
		  S_1_V_m_[num_com] = temp_vm+ temp_iex+ temp_iin + temp_ie;
		}
		else
		{
		  S_1_r_ref_[num_com]=S_1_r_ref_c-1;
		}

		S_1_i_syn_ex_[num_com] =(S_1_i_syn_ex_c-S_1_i_1_c)* V_1_P11ex_[num_com]+S_1_i_1_c ;
		S_1_i_syn_in_[num_com] =S_1_i_syn_in_c* V_1_P11in_[num_com];
		if(S_1_V_m_[num_com]>P_1_Theta_[num_com])
		{
			    tempfire[fire_count]=float32_to_uint64((float)(num*N+num_com),0);
				fire_count++;
		}
	}
    for(int num_com=0;num_com<N;num_com++)
    {
    	tempo[2*num_com] =float32_to_uint64(S_1_i_syn_ex_[num_com],S_1_i_syn_in_[num_com]);
    	tempo[2*num_com+1] =float32_to_uint64(S_1_r_ref_[num_com],S_1_V_m_[num_com]);

    	tempow[2*num_com] =float32_to_uint64(S_1_i_0_[num_com],S_1_i_1_[num_com]);
    	tempow[2*num_com+1] =float32_to_uint64(0,0);
    }


    memcpy((ap_uint<AXI_SIZE> *)(output+2*num*N),tempo,2*N*sizeof(ap_uint<AXI_SIZE>));
    memcpy((ap_uint<AXI_SIZE> *)(output_w+2*num*N),tempow,2*N*sizeof(ap_uint<AXI_SIZE>));
	}
	memcpy((ap_uint<AXI_SIZE> *)(output_fire),tempfire,fire_count*sizeof(ap_uint<AXI_SIZE>));
}



void iaf_psc_exp_ps_top(int num,int state,
		//int   *input_r_ref_,
		ap_uint<AXI_SIZE> *input_stream0,
		ap_uint<AXI_SIZE> *input_stream1,
		ap_uint<AXI_SIZE> *input_stream2,
		ap_uint<AXI_SIZE> *input_stream3,
		ap_uint<AXI_SIZE> *output_fire,
		ap_uint<AXI_SIZE> *output,
		ap_uint<AXI_SIZE> *output_w
        )
{

#pragma HLS INTERFACE s_axilite register port=return bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=state bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=num bundle=CTRL_BUS

#pragma HLS INTERFACE s_axilite register port=input_stream0 bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=input_stream1 bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=input_stream2 bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=input_stream3 bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=output_fire bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=output bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=output_w bundle=CTRL_BUS


#pragma HLS INTERFACE m_axi depth=100 port=output offset=slave bundle=DATA_INPUT3
#pragma HLS INTERFACE m_axi depth=100 port=output_w offset=slave bundle=DATA_INPUT2
#pragma HLS INTERFACE m_axi depth=100 port=output_fire offset=slave bundle=DATA_INPUT1
#pragma HLS INTERFACE m_axi depth=100 port=input_stream3 offset=slave bundle=DATA_INPUT3
#pragma HLS INTERFACE m_axi depth=100 port=input_stream2 offset=slave bundle=DATA_INPUT2
#pragma HLS INTERFACE m_axi depth=100 port=input_stream1 offset=slave bundle=DATA_INPUT1
#pragma HLS INTERFACE m_axi depth=100 port=input_stream0 offset=slave bundle=DATA_INPUT0


	 float P_1_I_e_[NUM_NEURON];
	 //inital
	 float V_1_P20_[NUM_NEURON];
	 float V_1_P11ex_[NUM_NEURON];
	 float V_1_P11in_[NUM_NEURON];
	 float V_1_P21ex_[NUM_NEURON];
	 float V_1_P21in_[NUM_NEURON];
	 float V_1_P22_[NUM_NEURON];

	 float i_0_[NUM_NEURON];
	 float i_1_[NUM_NEURON];
	 float i_syn_ex_[NUM_NEURON];
	 float i_syn_in_[NUM_NEURON];

	 float V_m_[NUM_NEURON];
	 float r_ref_[NUM_NEURON];
	 float V_1_weighted_spikes_ex_[NUM_NEURON];
	 float V_1_weighted_spikes_in_[NUM_NEURON];
	 float P_1_Theta_[NUM_NEURON];
	 float fire[NUM_NEURON];
	iaf_psc_exp_ps_update(num,fire,i_0_,i_1_,i_syn_ex_,i_syn_in_,V_m_,r_ref_,V_1_weighted_spikes_ex_,V_1_weighted_spikes_in_,
			P_1_I_e_,V_1_P20_,V_1_P11ex_,V_1_P11in_,V_1_P21ex_,V_1_P21in_,V_1_P22_,P_1_Theta_,
			input_stream0,input_stream1,input_stream2,input_stream3,output_fire,output,output_w);
}
