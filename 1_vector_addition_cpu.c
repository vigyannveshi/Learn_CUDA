/*
    Vector Addition:
    --> Three vectors of variable length having floating point data are created on host device dynamically created. 
    --> Vectors are initialized with random numbers between (0-1).
    --> Addition operation performed on these vectors.
    --> Testing of output (absolute(Sum_vector - vector_1 - vector_2)<1e-5 -- expected)
    --> Vectors are freed.
    --> Time Complexity: O[n], where n is the length of vectors.
*/

// header files
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void vecAdd(float* h_A, float* h_B,float* h_C,int n){
    for( int i=0; i<n;i++){
        h_C[i]=h_A[i] + h_B[i];
    }
}

int main(){
    float *h_A,*h_B,*h_C;
    int n;
    
    // Taking the input vectors:
    printf("Enter the length of vectors: ");
    scanf("%d", &n);

    h_A=(float*)malloc(n*sizeof(float)); 
    h_B=(float*)malloc(n*sizeof(float));
    h_C=(float*)malloc(n*sizeof(float));

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

    // Running the vecAdd function:
    vecAdd(h_A,h_B,h_C,n);

    // Testing results:
    for (int i = 0; i<n; i++){
        if (fabs(h_A[i]+h_B[i]-h_C[i])>1e-5){
            fprintf(stderr, "Result verification failed at element %d\n",i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test passed!");

    // Freeing allocated memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}