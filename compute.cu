#include "config.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>

#define NUMELEMENTS 1024
#define BLOCK_SIZE 16

__global__ void computeAccelMatrix(vector3* accels, vector3* h_pos, double* mass) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < NUMENTITIES &&  < NUMENTITIES) {
        if (i == j) {
            FILL_VECTOR(accels[i][j],0,0,0);
        } else {
            vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
                    double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                    double magnitude=sqrt(magnitude_sq);
                    double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
                    FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
    }
}


__global__ void sumAccelMat(vector3 accels, vector3* d_hVel, vector3* d_hPos) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    vector3 accel_sum={0,0,0};
    for (j=0;j<NUMENTITIES;j++){
        for (k=0;k<3;k++)
            accel_sum[k]+=accels[i][j][k];
    }

    for (k=0;k<3;k++){
        hVel[i][k]+=accel_sum[k]*INTERVAL;
        hPos[i][k]+=hVel[i][k]*INTERVAL;
    
}


void compute(vector3 *d_hPos, vector3 *d_hVel, double *mass) {

    vector3* d_accels;

    cudaMalloc((void **)&d_accels, NUMELEMENTS * NUMELEMENTS * sizeof(vector3));

    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((NUMELEMENTS + dimBlock.x - 1) / dimBlock.x,
                (NUMELEMENTS + dimBlock.y - 1) / dimBlock.y);

    computeAccelnMatrix<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
    sumAccelMat<<<(NUMELEMENTS + 255) / 256, 256>>>(d_accels, d_hPos, d_hVel);

    cudaFree(accels);
}