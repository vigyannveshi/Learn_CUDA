#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// KERNEL
__global__ void test01()
{
    // print the block and thread IDs
    printf("\nThe block ID is %d --- The thread id is %d\n", blockIdx.x, threadIdx.x); 
}

int main()
{
    // kernel_name<<< num_of_blocks, number of threads_per_block>>>(); 
    test01 <<< 2 ,1024 >>> ();
    // test01 <<< 1 ,2048 >>> (); // doesn't work, since max limit in total threads per-block is 1024 for Ampere architecture

    cudaDeviceSynchronize(); // wait for GPU to finish + flush printf buffer
    return 0;
}