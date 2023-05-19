#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 32

__global__ void matrixMultiplication(int* matrixA, int* matrixB, int* matrixC, int rowsA, int columnsA, int columnsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < columnsB) {
        int sum = 0;
        for (int k = 0; k < columnsA; ++k) {
            sum += matrixA[row * columnsA + k] * matrixB[k * columnsB + col];
        }
        matrixC[row * columnsB + col] = sum;
    }
}

void fillMatrix(int* matrix, int rows, int columns) {
    for (int i = 0; i < rows * columns; ++i) {
        matrix[i] = rand() % 10;
    }
}

void printMatrix(int* matrix, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            printf("%d ", matrix[i * columns + j]);
        }
        printf("\n");
    }
}

int main() {
    int rowsA, columnsA, columnsB;
    printf("Enter the dimensions of matrix A (rowsA): ");
   
       scanf("%d", &rowsA);
    printf("Enter the dimensions of matrix B (columnsA x columnsB): ");
    scanf("%d %d", &columnsA, &columnsB);

    int* matrixA = NULL;
    int* matrixB = NULL;
    int* matrixC = NULL;
    int* dev_matrixA = NULL;
    int* dev_matrixB = NULL;
    int* dev_matrixC = NULL;

    size_t sizeA = rowsA * columnsA * sizeof(int);
    size_t sizeB = columnsA * columnsB * sizeof(int);
    size_t sizeC = rowsA * columnsB * sizeof(int);

    matrixA = (int*)malloc(sizeA);
    matrixB = (int*)malloc(sizeB);
    matrixC = (int*)malloc(sizeC);

    srand(time(NULL));

    fillMatrix(matrixA, rowsA, columnsA);
    fillMatrix(matrixB, columnsA, columnsB);

    cudaMalloc((void**)&dev_matrixA, sizeA);
    cudaMalloc((void**)&dev_matrixB, sizeB);
    cudaMalloc((void**)&dev_matrixC, sizeC);

    cudaMemcpy(dev_matrixA, matrixA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixB, matrixB, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((columnsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiplication<<<gridDim, blockDim>>>(dev_matrixA, dev_matrixB, dev_matrixC, rowsA, columnsA, columnsB);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(matrixC, dev_matrixC, sizeC, cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    printMatrix(matrixA, rowsA, columnsA);

    printf("Matrix B:\n");
    printMatrix(matrixB, columnsA, columnsB);

    printf("Resultant Matrix C:\n");
    printMatrix(matrixC, rowsA, columnsB);

    printf("%f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixC);
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
