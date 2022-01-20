#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void readMatrixAndVectorFromFile();
void readFloatRowFormFile(FILE* fp, int size_of_row, float** data);
void readIntRowFromFile(FILE* fp, int size_of_row, int** data);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

char* matrixFileName = "matrix_ell_coo_7.csv";
int dimension = 0;

// Daten der Matrix im ELL Format
int data_ell_size = 0;
int cols_ell_size = 0;
float* data_ell = NULL;
int* cols_ell = NULL;

// Daten der Matrix im COO Format
int data_coo_size = 0;
float* data_coo = NULL;
int* rows_coo = NULL;
int* cols_coo = NULL;

// Der Ergebnisvektor
float* vector = NULL;

int main()
{
    
    readMatrixAndVectorFromFile();

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1,size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

void readMatrixAndVectorFromFile()
{
    dimension = 0;
    int MAX_FLOAT_STRING_LENGTH = 8;
    FILE* fp;
    fp = fopen(matrixFileName, "r");
    fscanf(fp, "%d", &dimension);

    // Lese ELL Format
    fscanf(fp, "%d,%d", &data_ell_size, &cols_ell_size);

    data_ell = (float*)malloc(data_ell_size * sizeof(float));
    cols_ell = (int*)malloc(cols_ell_size * sizeof(int));
    vector = (float*)malloc(dimension * sizeof(float));
    readFloatRowFormFile(fp, data_ell_size, &data_ell);
    readIntRowFromFile(fp, cols_ell_size, &cols_ell);

    printf("Dimension: %d\nELL Data Size: %d\nELL Cols Size: %d\n", dimension, data_ell_size, cols_ell_size);

    printf("Data ELL: ");
    for (int i = 0; i < data_ell_size; i++) {
        printf("%f ", data_ell[i]);
    }
    printf("\n");
    printf("Cols ELL: ");
    for (int i = 0; i < cols_ell_size; i++) {
        printf("%d ", cols_ell[i]);
    }
    printf("\n");
    
    // Lese COO Format
    fscanf(fp, "%d", &data_coo_size);
    data_coo = (float*)malloc(data_coo_size * sizeof(float));
    rows_coo = (int*)malloc(data_coo_size * sizeof(int));
    cols_coo = (int*)malloc(data_coo_size * sizeof(int));
    readFloatRowFormFile(fp, data_coo_size, &data_coo);
    readIntRowFromFile(fp, data_coo_size, &rows_coo);
    readIntRowFromFile(fp, data_coo_size, &cols_coo);

    printf("COO Data Size: %d\n", data_coo_size);

    printf("Data COO: ");
    for (int i = 0; i < data_coo_size; i++) {
        printf("%f ", data_coo[i]);
    }
    printf("\n");
    printf("Rows COO: ");
    for (int i = 0; i < data_coo_size; i++) {
        printf("%d ", rows_coo[i]);
    }
    printf("\n");

    printf("Cols COO: ");
    for (int i = 0; i < data_coo_size; i++) {
        printf("%d ", cols_coo[i]);
    }
    printf("\n");

    // Lese Vektor
    readFloatRowFormFile(fp, dimension, &vector);
    printf("Vector: ");
    for (int i = 0; i < dimension; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");

    fclose(fp);
    return;
}

void readFloatRowFormFile(FILE* fp, int size_of_row, float** data) {

    int MAX_FLOAT_STRING_LENGTH = 8;

    // Lese die ELL Daten
    char* row = (char*)malloc((size_of_row * MAX_FLOAT_STRING_LENGTH + size_of_row - 1) * sizeof(char));
    fscanf(fp, "%s\n", row);
    int fromIndex = 0;
    int floatsRead = 0;
    for (int k = 0; k < strlen(row); k++)
    {
        if (row[k] == ',')
        {
            char* substr = (char*)malloc(MAX_FLOAT_STRING_LENGTH);
            strncpy(substr, row + fromIndex, k - fromIndex - 1);
            float x = atof(substr);
            (*data)[floatsRead] = x;
            floatsRead++;
            fromIndex = k + 1;
        }
    }

    char* substr = (char*)malloc(MAX_FLOAT_STRING_LENGTH);
    strncpy(substr, row + fromIndex, strlen(row) - fromIndex - 1);
    (*data)[floatsRead] = atof(substr);

    return;
}

void readIntRowFromFile(FILE* fp, int size_of_row, int** data) {
    int MAX_INT_STRING_LENGTH = 8;
    char* substr;
    char* row_cols = (char*)malloc((size_of_row * MAX_INT_STRING_LENGTH + size_of_row - 1) * sizeof(char));

    fscanf(fp, "%s\n", row_cols);
    int fromIndex = 0;
    int intsRead = 0;
    for (int k = 0; k < strlen(row_cols); k++)
    {
        if (row_cols[k] == ',')
        {
            substr = (char*)malloc(MAX_INT_STRING_LENGTH);
            strncpy(substr, row_cols + fromIndex, k - fromIndex);
            (*data)[intsRead] = atoi(substr);
            intsRead++;
            fromIndex = k + 1;
        }
    }

    substr = (char*)malloc(MAX_INT_STRING_LENGTH);
    strncpy(substr, row_cols + fromIndex, strlen(row_cols) - fromIndex);
    (*data)[intsRead] = atoi(substr);
}