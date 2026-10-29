/*
@file           002_hello_world.cu
@purpose        To understand gridDim, blockDim, warps
@author         Rudraksha R. Bandodkar
@additionals
RTX 3090 (cc 8.6)
SMs: 82
Threads per SM = 1536
Max blocks that can be resident on one SM at the same time = 16
Threads per block (limit) = 1024
Max resident blocks on the GPU = 16 * 82 = 1312
Max resident threads on the GPU = 1536 * 82 = 125,952
(more blocks than this can be launched, the extra ones wait in a queue)
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// KERNEL: print block, thread and warp IDs
__global__ void test01()
{
    // each warp consists of 32 threads
    // warps per block = threads per block / threads per warp
    // eg: 1024 / 32 = 32 warps per block
    int warpId = threadIdx.x / 32;

    /* warp id using % */
    // int tid = threadIdx.x;
    // int laneId = tid % 32;
    // int warpId = (tid - laneId) / 32; // overwork

    // global warp id (unique across the whole grid)
    // int gid = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    // int globalWarp = gid / 32;

    printf("\nThe block ID is %d --- The thread id is %d --- The warp ID %d\n", blockIdx.x, threadIdx.x, warpId);
}

// KERNEL: print the launch configuration
__global__ void printGPUDetails()
{
    printf("Grid Dim (No of blocks per grid): %d\n", gridDim.x);
    printf("Block Dim (No of threads per block): %d\n", blockDim.x);
}

int main()
{
    // kernel_name<<< num_of_blocks, number_of_threads_per_block >>>();
    printGPUDetails<<<1, 1>>>();
    cudaDeviceSynchronize(); // wait for GPU to finish + flush printf buffer

    // 1400 blocks > 1312 resident blocks, the remaining ones wait for a free slot
    test01<<<1400, 1>>>();
    // test01<<<1, 2048>>>(); // doesn't work, since max limit in total threads per-block is 1024 for Ampere architecture
    cudaDeviceSynchronize(); // wait for GPU to finish + flush printf buffer
    // fflush(stdout);       // flush stdout so redirect captures it

    return 0;
}