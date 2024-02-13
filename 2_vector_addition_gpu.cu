// header files
#include <cuda.h>
#include <cuda_runtime.h>



// CPU side host program
void vecAdd(float* h_A, float *h_B, float *h_C, int n){
    int size = n*sizeof(float);
    float *d_A=NULL, *d_B=NULL, *d_C=NULL;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Initiating vectors on cuda device
    printf("Initiate device vectors\n");
    err=cudaMalloc((void **)&d_A, size);
    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err=cudaMalloc((void **)&d_B, size);
    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err=cudaMalloc((void **)&d_C, size);
    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copying vectors from host memory to the CUDA device
    printf("Copy input data from host memory to the CUDA device\n");

    err=cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err=cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err=cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threads    

}
