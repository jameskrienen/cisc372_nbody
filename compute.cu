#include "config.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>

#define NUMELEMENTS 1024
#define BLOCK_SIZE 16

__global__ void computeAccelMatrix(vector3* accels, vector3* d_pos, double* d_mass) {
	int i =  blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    
	if (i < NUMENTITIES && j < NUMENTITIES) {
        	vector3 distance;
        	double magnitude, magnitude_sq,accelmag;
	    
		if (i == j) {
			for (int k = 0;k < 3; k++) {
				accels[i * NUMELEMENTS + j][k] = 0;
			}
        	} else {
		
			for (int k=0;k<3;k++) distance[k]=d_pos[i][k]-d_pos[j][k];
                	
			magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                        magnitude=sqrt(magnitude_sq);
                        accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
			
			for (int k = 0; k < 3; k++) {

                    
				//FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
    				//filling vector with out FILL_VECTOR
				double tmp = accelmag*distance[k]/magnitude;
				accels[i * NUMELEMENTS + j][k] = tmp;
				accels[j * NUMELEMENTS + i][k] = tmp;
			}	
	}
    }
}


__global__ void sumAccelMat(vector3* accels, vector3* d_vel, vector3* d_pos) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMELEMENTS) {
        vector3 accel_sum={0,0,0};
        for (int j = 0; j < NUMELEMENTS; j++){
            for (int k = 0; k < 3 ; k++) {
                accel_sum[k] += accels[i * NUMELEMENTS + j][k];
            }
        }
        for (int k = 0; k < 3; k++){
            d_vel[i][k]+=accel_sum[k]*INTERVAL;
            d_pos[i][k]+=d_vel[i][k]*INTERVAL;
        }
    }
}


void compute(vector3 *d_hPos, vector3 *d_hVel, double *d_mass) {

    vector3* d_accels;

    cudaMalloc((void **)&d_accels, NUMELEMENTS * NUMELEMENTS * sizeof(vector3));

    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((NUMELEMENTS + dimBlock.x - 1) / dimBlock.x,
                (NUMELEMENTS + dimBlock.y - 1) / dimBlock.y);

    computeAccelMatrix<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
    sumAccelMat<<<(NUMELEMENTS + 255) / 256, 256>>>(d_accels, d_hPos, d_hVel);

    cudaFree(d_accels);
}
