
#include "config.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>

// Define constants
#define NUMELEMENTS 1024
#define BLOCK_SIZE 16

// Kernel to compute acceleration matrix based on gravitational forces
__global__ void computeAccelMatrix(vector3* accels, vector3* d_pos, double* d_mass) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMELEMENTS && j < NUMELEMENTS) {
        vector3 distance;
        double magnitude, magnitude_sq, accelmag;

        if (i == j) {
            // Diagonal elements (self-interaction) are set to zero
            for (int k = 0; k < 3; k++) {
                accels[i * NUMELEMENTS + j][k] = 0;
            }
        } else {
            // Compute distance vector and magnitudes
            for (int k = 0; k < 3; k++) {
                distance[k] = d_pos[i][k] - d_pos[j][k];
            }
            magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            magnitude = sqrt(magnitude_sq);

            // Compute gravitational acceleration magnitude
            accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;

            // Compute and set acceleration vectors
            for (int k = 0; k < 3; k++) {
                double tmp = accelmag * distance[k] / magnitude;
                accels[i * NUMELEMENTS + j][k] = tmp;
                accels[j * NUMELEMENTS + i][k] = tmp; // Symmetric matrix
            }
        }
    }
}

// Kernel to sum up accelerations and update velocities and positions
__global__ void sumAccelMat(vector3* accels, vector3* d_vel, vector3* d_pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMELEMENTS) {
        vector3 accel_sum = {0, 0, 0};
        // Sum up accelerations
        for (int j = 0; j < NUMELEMENTS; j++) {
            for (int k = 0; k < 3; k++) {
                accel_sum[k] += accels[i * NUMELEMENTS + j][k];
            }
        }
        // Update velocities and positions
        for (int k = 0; k < 3; k++) {
            d_vel[i][k] += accel_sum[k] * INTERVAL;
            d_pos[i][k] += d_vel[i][k] * INTERVAL;
        }
    }
}

// Function to perform the main computation on the device
void compute(vector3 *d_hPos, vector3 *d_hVel, double *d_mass) {
    vector3* d_accels;

    // Allocate device memory for acceleration matrix
    cudaMalloc((void **)&d_accels, NUMELEMENTS * NUMELEMENTS * sizeof(vector3));

    // Set up grid and block dimensions for kernel launches
    dim3 dimBlock(16, 16);
    dim3 dimGrid((NUMELEMENTS + dimBlock.x - 1) / dimBlock.x,
                (NUMELEMENTS + dimBlock.y - 1) / dimBlock.y);

    // Launch kernels to compute acceleration matrix and update velocities/positions
    computeAccelMatrix<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
    sumAccelMat<<<(NUMELEMENTS + 255) / 256, 256>>>(d_accels, d_hPos, d_hVel);

    // Free allocated memory
    cudaFree(d_accels);
}
