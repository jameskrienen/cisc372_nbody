#ifndef COMPUTE_H
#define COMPUTE_H

#include "vector.h"

extern vector3 *d_hPos, *d_hVel;
extern double *d_mass;

void compute(vector3 *d_hPos, vector3 *d_hVel, double *d_mass);
__global__ void sumAccelMat(vector3* accels, vector3* d_hVel, vector3* d_hPos);
__global__ void computeAccelMatrix(vector3* accels, vector3* h_pos, double* mass);

#endif
