/* 
Matrix Multiplication using GPU
--> This is not a optimized GPU program
--> It can be further accelerated using several primitives that GPU offer to it.
 */
#include <stdio.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

__global__
void MatrixMulKernel(float * d_M, float* d_N, float* d_P, int n){
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if ((i<n) && (j<n)){
        float Pvalue =0.0;
        for(int k=0; k<n;k++){
            Pvalue+=d_M[i*n+k]*d_N[k*n+j];
        }
        d_P[i*n+j]=Pvalue;
    }
}

int main() {
    int n = 500;
    int size = n * n;

    // Allocate memory for host matrices
    float *h_M = (float *)malloc(size * sizeof(float));
    float *h_N = (float *)malloc(size * sizeof(float));
    float *h_P = (float *)malloc(size * sizeof(float));

    // Initialize host matrices with some values
    for (int i = 0; i < size; i++) {
        h_M[i] = i;  // Example initialization, you can set your own values
        h_N[i] = i;
    }

    // Allocate memory on the device for matrices
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, size * sizeof(float));
    cudaMalloc(&d_N, size * sizeof(float));
    cudaMalloc(&d_P, size * sizeof(float));

    // Transfer host matrices to device
    cudaMemcpy(d_M, h_M, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid(2, 2);    // 2x2 grid
    dim3 block(8, 8);   // 8x8 threads per block

    // Perform matrix multiplication on GPU
    MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, n);

    // Transfer result matrix from device to host
    cudaMemcpy(h_P, d_P, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result (printing only a part of the matrix for brevity)
    // printf("M: ");
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%.2f\t", h_M[i * n + j]);
    //     }
    //     printf("\n");
    // }
    // printf("N: ");
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%.2f\t", h_N[i * n + j]);
    //     }
    //     printf("\n");
    // }
    // printf("Result matrix (partial):\n");
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%.2f\t", h_P[i * n + j]);
    //     }
    //     printf("\n");
    // }

    printf("successfully calculated!");

    
    // Free host and device memory
    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}