#include <stdio.h>
#include <stdlib.h>

void MatrixMultiplication(float *M, float *N, float *P, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float Pvalue = 0.0;
            for (int k = 0; k < n; ++k) {
                // Use single subscripting with proper indexing
                Pvalue += M[i * n + k] * N[k * n + j];
            }
            // Assign the computed value to the correct position in P
            P[i * n + j] = Pvalue;
        }
    }
}


int main() {
    int n = 3; // Example matrix size

    // Allocate memory for matrices
    float *A = (float *)malloc(n * n * sizeof(float));
    float *B = (float *)malloc(n * n * sizeof(float));
    float *C = (float *)malloc(n * n * sizeof(float)); // Result matrix

    // Initialize matrices A and B with some values
    for (int i = 0; i < n * n; ++i) {
        A[i] = i;
        B[i] = i;
    }

    // Print A
    printf("A:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%0.2f\t", A[i * n + j]);
        }
        printf("\n");
    }

    // Print B
    printf("B:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%0.2f\t", B[i * n + j]);
        }
        printf("\n");
    }
    // Perform matrix multiplication
    MatrixMultiplication(A, B, C, n);

    // Print the result matrix C
    printf("Result Matrix:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%0.2f\t", C[i * n + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}