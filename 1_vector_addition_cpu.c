/*
Vector Addition:
--> This is a basic vector addition program that runs entirely on CPU
--> Vectors of dynamic length can be created
*/


#include <stdio.h>
#include <stdlib.h>


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

    for (int i = 0; i < n; i++) {
        printf("A[%d]: ", i);
        scanf("%f", &h_A[i]);
        printf("B[%d]: ", i);
        scanf("%f", &h_B[i]);
    }

    // running the vecAdd function:
    vecAdd(h_A,h_B,h_C,n);

    // printing the output:
    printf("C= [ ");
    for(int i=0; i<n;i++){
        printf("%f, ",h_C[i]);
    }
    printf("]");

    // Freeing allocated memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}