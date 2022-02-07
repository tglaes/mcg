#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void readMatrixAndVectorFromFile();
void readFloatRowFormFile(FILE* fp, int size_of_row, float** data);
void readIntRowFromFile(FILE* fp, int size_of_row, int** data);

__global__ void test(int dimension, float* data_ell, int data_ell_size, int* cols_ell, int cols_ell_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    printf("This is thread number %d\n", idx );
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

    //test<<<1, 10 >>> (dimension, data_ell, data_ell_size, cols_ell, cols_ell_size);

    cudaFree(data_ell);
    cudaFree(cols_ell);
    cudaFree(data_coo);
    cudaFree(rows_coo);
    cudaFree(cols_coo);
    cudaFree(vector);

    return 0;
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

    cudaMallocManaged(&data_ell, data_ell_size * sizeof(float));
    cudaMallocManaged(&cols_ell,cols_ell_size * sizeof(int));
    cudaMallocManaged(&vector, dimension * sizeof(float));
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
    cudaMallocManaged(&data_coo, data_coo_size * sizeof(float));
    cudaMallocManaged(&rows_coo, data_coo_size * sizeof(int));
    cudaMallocManaged(&cols_coo, data_coo_size * sizeof(int));
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
            free(substr);
        }
    }
    char* substr = (char*)malloc(MAX_FLOAT_STRING_LENGTH);
    strncpy(substr, row + fromIndex, strlen(row) - fromIndex - 1);
    (*data)[floatsRead] = atof(substr);

    free(row);
    free(substr);
    return;
}

void readIntRowFromFile(FILE* fp, int size_of_row, int** data) {
    int MAX_INT_STRING_LENGTH = 8;
    char* substr;
    char* row = (char*)malloc((size_of_row * MAX_INT_STRING_LENGTH + size_of_row - 1) * sizeof(char));

    fscanf(fp, "%s\n", row);
    int fromIndex = 0;
    int intsRead = 0;
    for (int k = 0; k < strlen(row); k++)
    {
        if (row[k] == ',')
        {
            substr = (char*)malloc(MAX_INT_STRING_LENGTH);
            strncpy(substr, row + fromIndex, k - fromIndex);
            (*data)[intsRead] = atoi(substr);
            intsRead++;
            fromIndex = k + 1;
            free(substr);
        }
    }

    substr = (char*)malloc(MAX_INT_STRING_LENGTH);
    strncpy(substr, row + fromIndex, strlen(row) - fromIndex);
    (*data)[intsRead] = atoi(substr);
    free(row);
    free(substr);
    return;
}