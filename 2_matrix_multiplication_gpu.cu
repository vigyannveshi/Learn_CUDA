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
    if ((i<N) && (j<N)){
        float Pvalue =0.0;
        for(int k=0; k<N;k++){
            Pvalue+=d_M[i*n+k]*d_N[k*n+j]
        }
        d_P[i*n+j]=Pvalue;
    }
}

int main(){
    int size = 16*16
    cudaMemcpy(d_M,M,size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N,N,size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(2,2,1);  // (2x2x1) --> 4 Thread blocks/grid
    dim3 block(8,8,1); // (8x8x1) --> 64 Threads/block
    // Total threads --> 4*64=256
    int N=16; // n is the number of rows and columns
    MatrixMulKernel <<<grid,block>>>(d_M,d_N,d_P,n);
    cudaMemcpy(P,d_P,size*sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}