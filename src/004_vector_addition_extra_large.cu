/*
@file           004_vector_addition_extra_large.cu
@purpose        Vector addition on a vector too large to handle in one go,
                processed in chunks
@author         Rudraksha R. Bandodkar
@additionals
TOTAL_SIZE = 2^30 elements, CHUNK_SIZE = 2^27 elements -> 8 chunks
Each chunk buffer is 512 MB (3 on the host, 3 on the device)
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE (1024 * 1024 * 1024) // total elements in the vector
#define CHUNK_SIZE (1024 * 1024 * 128)  // elements per chunk, adjust based on available host RAM
#define BLOCK_SIZE 1024                 // number of threads per block


// CUDA kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int chunkSize){
    // int i = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < chunkSize) // ensure we don't go out of bounds
        C[idx] = A[idx] + B[idx];
}

// fill a host array with random values in [0, 99]
void random_ints(int *x, int size){
    for (int i = 0; i < size; i++){
        x[i] = rand() % 100;
    }
}


int main(){
    // STEP 1. Declare memory
    int *chunk_A, *chunk_B, *chunk_C;   // host vectors for chunk
    int *d_A, *d_B, *d_C;               // device vectors
    size_t chunkSizeBytes = CHUNK_SIZE * sizeof(int); // size of a chunk in bytes

    // STEP 2. Allocate host vectors (once, reused for every chunk)
    chunk_A = (int*) malloc(chunkSizeBytes);
    chunk_B = (int*) malloc(chunkSizeBytes);
    chunk_C = (int*) malloc(chunkSizeBytes);
    if (!chunk_A || !chunk_B || !chunk_C){
        printf("Host allocation failed\n");
        return 1;
    }

    // STEP 3. Allocate device vectors (once, reused for every chunk)
    /*
        cudaMalloc needs to write a value into your pointer variable, and in C a function can only change a variable if you pass it a address

        A pointer to a generic pointer [in C its void*](we don't want int**, float** struct** every time we create an type) is void**
    */
    cudaMalloc((void**)&d_A, chunkSizeBytes);
    cudaMalloc((void**)&d_B, chunkSizeBytes);
    cudaMalloc((void**)&d_C, chunkSizeBytes);

    // STEP 4. Process the vectors chunk by chunk
    for (long long offset = 0; offset < TOTAL_SIZE; offset += CHUNK_SIZE)
    {
        // calculate the current chunk size (may be smaller at the end of the vector)
        int currentChunkSize = (int)((TOTAL_SIZE - offset) < CHUNK_SIZE ? (TOTAL_SIZE - offset) : CHUNK_SIZE);
        // calculate the number of blocks for the kernel
        int numBlocks = (currentChunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        printf("\nOffset %lld\n", offset);

        // populate the chunks with random data
        random_ints(chunk_A, currentChunkSize);
        random_ints(chunk_B, currentChunkSize);

        // COPY FROM HOST (CPU) to DEVICE (GPU)
        /* dest, src, size, direction */
        cudaMemcpy(d_A, chunk_A, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, chunk_B, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice);

        // STEP 5. Launch the Vector Add CUDA kernel
        vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, currentChunkSize);

        // STEP 6. Copy the result back to host
        /* dest, src, size, direction */
        cudaMemcpy(chunk_C, d_C, currentChunkSize * sizeof(int), cudaMemcpyDeviceToHost);

        // wait for the result
        cudaDeviceSynchronize(); // redundant, since cudaMemcpy waits for the kernel already

        // spot-check the first 100 results of this chunk
        for (int i = 0; i < 100; i++){
            printf("%d + %d  = %d\n", chunk_A[i], chunk_B[i], chunk_C[i]);
        }
    }

    // STEP 7. Cleanup (once, after all chunks are done)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(chunk_A);
    free(chunk_B);
    free(chunk_C);

    return 0;
}