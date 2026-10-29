/*
@file           001_hello_workd.cu
@purpose        Vector addition
@author         Rudraksha R. Bandodkar
@additionals
1. Adding two vectors with 1024 elements each 
 
A -> 0,1,2...1023
B -> 0,1,2...1023

2. Adding two vectors with 2048 elements each
A -> 0,1,... 2048
B -> 0,1,... 2048

Assinging more blocks keeping the total threads = no of operands, helps in increasing the SMs thereby increasing GPU utility and thus faster speed of execution.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "cuda.h" 

#define SIZE 2048 // Define the size of the vectors

// CUDA kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int n){
    // int i = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    C[idx] = A[idx] + B[idx];
}


int main(){
    // STEP 1. Allocate memory space
    int *A, *B, *C;         // host vectors
    int *d_A, *d_B, *d_C;   // device vectors
    int size = SIZE * sizeof(int);
    
    // STEP 2. Allocate and initialize host vectors
    A = (int*) malloc(size);
    B = (int*) malloc(size);
    C = (int*) malloc(size);

    // STEP 3. Allocate device vectors
    /*
        cudaMalloc needs to write a value into your pointer variable, and in C a function can only change a variable if you pass it a address

        A pointer to a generic pointer [in C its void*](we don't want int**, float** struct** every time we create an type) is void**
    */
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // STEP 4. Initialize the inputs 
    for (int i = 0; i < SIZE; i++){
        A[i] = i;
        B[i] = SIZE - i;
    }  

    // COPY FROM HOST (CPU) to DEVICE (GPU)
    /* dest, src, size, direction */
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // TIMING 
    /*create events for timing*/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    // STEP 5. Launch the Vector Add CUDA kernel
    cudaEventRecord(start);
    vectorAdd << < 2, 1024 >> >(d_A, d_B, d_C, SIZE);
    // vectorAdd << < 64, 32 >> >(d_A, d_B, d_C, SIZE);
    cudaEventRecord(stop);
     
    
    // STEP 6. Copy the result back to host
    /* dest, src, size, direction */
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
 
    cudaEventSynchronize(stop); // make sure stop has completed before reading

    // wait for the result
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    for (int i = 0; i < SIZE; i++){
        printf("%d + %d  = %d\n", A[i], B[i], C[i]);
    }
    printf("Execution time: %f milliseconds \n", milliseconds);

    // STEP 7. Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}