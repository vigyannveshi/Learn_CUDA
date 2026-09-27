#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// KERNEL
__global__ void test01()
{   
    // each warp consists of 32 threads
    // each block consists of 1024 threads (warps per block = threads/block / threads/warp ) = 1024 / 32 = 32
    int warpId = threadIdx.x / 32;

    // print the block, thread and warp IDs
    printf("\nThe block ID is %d --- The thread id is %d --- The warp ID %d\n", blockIdx.x, threadIdx.x, warpId); 
}

int main()
{
    // kernel_name<<< num_of_blocks, number of threads_per_block>>>(); 
    test01 <<< 2 ,64 >>> ();
    // test01 <<< 1 ,2048 >>> (); // doesn't work, since max limit in total threads per-block is 1024 for Ampere architecture

    cudaDeviceSynchronize(); // wait for GPU to finish + flush printf buffer
    // fflush(stdout);            // flush stdout so redirect captures it
    return 0;
}