
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// Global variables representing the objects in the system
vector3 *hVel, *d_hVel;    // Host and device velocity arrays
vector3 *hPos, *d_hPos;    // Host and device position arrays
double *mass, *d_mass;      // Host and device mass arrays

// Function to initialize host memory for the system
void initHostMemory(int numObjects);

// Function to free memory allocated by initHostMemory
void freeHostMemory();

// Function to fill the first NUMPLANETS+1 entries of the entity arrays with an estimation of our solar system
void planetFill();

// Function to fill the rest of the objects in the system randomly
void randomFill(int start, int count);

// Function to print the entire system to the supplied file
void printSystem(FILE* handle);

int main(int argc, char **argv)
{
    clock_t t0 = clock();   // Start time

    int t_now;

    srand(1234);  // Seed for random number generation
    initHostMemory(NUMENTITIES);  // Allocate memory for host arrays
    planetFill();  // Fill the first entries with solar system data
    randomFill(NUMPLANETS + 1, NUMASTEROIDS);  // Fill the rest with random data

    // Allocate memory for device arrays and copy initial data
    cudaMalloc(&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc(&d_mass, sizeof(double) * NUMENTITIES);

    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * (NUMPLANETS + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * (NUMPLANETS + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * (NUMPLANETS + 1), cudaMemcpyHostToDevice);

    // Print initial system if DEBUG is defined
    #ifdef DEBUG
    printSystem(stdout);
    #endif

    // Time evolution loop
    for (t_now = 0; t_now < DURATION; t_now += INTERVAL)
    {
        compute(d_hPos, d_hVel, d_mass);  // Perform computation on the device
    }

    // Copy the final results back to the host
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

    // Calculate and print total execution time
    clock_t t1 = clock() - t0;
    #ifdef DEBUG
    printSystem(stdout);
    #endif
    printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);

    // Free allocated memory
    cudaFree(d_hVel);
    cudaFree(d_hPos);
    cudaFree(d_mass);
    freeHostMemory();

    return 0;
}

// Function to initialize host memory for the system
void initHostMemory(int numObjects)
{
    hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
    hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
    mass = (double *)malloc(sizeof(double) * numObjects);
}

// Function to free memory allocated by initHostMemory
void freeHostMemory()
{
    free(hVel);
    free(hPos);
    free(mass);
}

// Function to fill the first NUMPLANETS+1 entries of the entity arrays with an estimation of our solar system
void planetFill()
{
    int i, j;
    double data[][7] = {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE};

    for (i = 0; i <= NUMPLANETS; i++)
    {
        for (j = 0; j < 3; j++)
        {
            hPos[i][j] = data[i][j];
            hVel[i][j] = data[i][j + 3];
        }
        mass[i] = data[i][6];
    }
}

// Function to fill the rest of the objects in the system randomly
void randomFill(int start, int count)
{
    int i, j, c = start;

    for (i = start; i < start + count; i++)
    {
        for (j = 0; j < 3; j++)
        {
            hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
            hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
            mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
        }
    }
}

// Function to print the entire system to the supplied file
void printSystem(FILE *handle)
{
    int i, j;

    for (i = 0; i < NUMENTITIES; i++)
    {
        fprintf(handle, "pos=(");
        for (j = 0; j < 3; j++)
        {
            fprintf(handle, "%lf,", hPos[i][j]);
        }
        fprintf(handle, "),v=(");
        for (j = 0; j < 3; j++)
        {
            fprintf(handle, "%lf,", hVel[i][j]);
        }
        fprintf(handle, "),m=%lf\n", mass[i]);
    }
}

