/* 
Vector Addition GPU:
*/

// header files
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

/* Kernel function Prototype*/
// __global__ void vectorAdd(float *, float *,float *, int);

/* Kernel function Implementation/Definition */
__global__ void vectorAdd(const float *A,const  float *B, float *C, int n ){
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    if (i<n){
        C[i]=A[i]+B[i];
    }
}


/* CPU side host program */ 
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

    // Launching Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n+threadsPerBlock -1 )/threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads \n", threadsPerBlock,blocksPerGrid);

    /* GPU KERNEL */
    vectorAdd<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, n);
    /* GPU KERNEL */

    err=cudaGetLastError();

    // device function (CUDA kernel) called from host does not have return type
    // CUDA runtime functions (execute in host side) can have return type

    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Device to Host Memory Transfer
    printf("Copy output data from output device to the host memory\n");
    err = cudaMemcpy(h_C ,d_C ,size , cudaMemcpyDeviceToHost);
    if (err !=cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free all the allocated memory in the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Verify that the result vector is correct 
    for (int i = 0; i<n; i++){
        if (fabs(h_A[i]+h_B[i]-h_C[i])>1e-5){
            fprintf(stderr, "Result verification failed at element %d\n",i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test Passed");
}

int main(void){
    int n;
    printf("Enter the vector length: ");
    scanf("%d",&n);
    int size = n*sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < n; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Running the vector addition

    vecAdd(h_A, h_B, h_C, n);
    return 0;
}