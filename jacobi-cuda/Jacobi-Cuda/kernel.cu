#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif
#include <string>

void readMatrixAndVectorFromFile();
void readFloatRowFormFile(FILE* fp, int size_of_row, float** data);
void readIntRowFromFile(FILE* fp, int size_of_row, int** data);
int calculate_grid_dimension(int matrix_dimension);
__global__ void jacobi(int matrix_dimension, int* prefix_array, int* rows_coo, int data_coo_size, int size_of_coo_row);

const char* matrix_file_name = "matrix_ell_coo_15.csv";
int matrix_dimension = 0;

// Daten der Matrix im ELL Format
int data_ell_size = 0;
int cols_ell_size = 0;
int size_of_ell_row = 0;
float* data_ell = NULL;
int* cols_ell = NULL;

// Daten der Matrix im COO Format
int data_coo_size = 0;
int size_of_coo_row = 0;
float* data_coo = NULL;
int* rows_coo = NULL;
int* cols_coo = NULL;

// Der Ergebnisvektor
float* vector = NULL;

// Prefix array
int* prefix_array = NULL;

int main()
{
    readMatrixAndVectorFromFile();
    int grid_dimension = calculate_grid_dimension(matrix_dimension);
    cudaMallocManaged(&prefix_array, data_coo_size/ size_of_coo_row);
    jacobi<<<grid_dimension, 1024 >>> (matrix_dimension, prefix_array ,rows_coo, data_coo_size, size_of_coo_row);

    cudaFree(data_ell);
    cudaFree(cols_ell);
    cudaFree(data_coo);
    cudaFree(rows_coo);
    cudaFree(cols_coo);
    cudaFree(vector);

    return 0;
}

__global__ void jacobi(int matrix_dimension, int* prefix_array, int* rows_coo, int data_coo_size, int size_of_coo_row)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < (data_coo_size / size_of_coo_row)) {
        printf("IDX:%d = %d\n", idx, rows_coo[idx * size_of_coo_row]);
        prefix_array[idx] = rows_coo[idx * size_of_coo_row];
    }

    __syncthreads();

    if (idx < matrix_dimension) {
        printf("This is thread number %d\n", idx);

        int row_offset = 0;
        bool is_coo_row = false;

        for (int i = 0; i < (data_coo_size / size_of_coo_row); i++) {
            if (prefix_array[i] == idx) {
                // Zeile ist eine COO Zeile
                is_coo_row = true;
                break;
            }
            else {
                // Zeile ist eine ELL Zeile
                // Berechne row_offset (Wie viele COO Zeilen gab es bis idx)
                if (prefix_array[i] < idx) {
                    row_offset++;
                }
                else {
                    break;
                }
            }
        }
        if (is_coo_row) {
            printf("IDX:%d is a COO row\n", idx);
        }
        else {
            printf("IDX:%d is a ELL row with row_offset %d\n", idx, row_offset);
        }
    }
}

int calculate_grid_dimension(int dimension) {
    return (int)ceil(dimension / static_cast<double>(1024));
}

void readMatrixAndVectorFromFile()
{
    FILE* fp;
    fp = fopen(matrix_file_name, "r");
    fscanf(fp, "%d,%d", &matrix_dimension, &size_of_ell_row);

    // Lese ELL Format
    fscanf(fp, "%d,%d", &data_ell_size, &cols_ell_size);

    cudaMallocManaged(&data_ell, data_ell_size * sizeof(float));
    cudaMallocManaged(&cols_ell,cols_ell_size * sizeof(int));
    cudaMallocManaged(&vector, matrix_dimension * sizeof(float));
    readFloatRowFormFile(fp, data_ell_size, &data_ell);
    readIntRowFromFile(fp, cols_ell_size, &cols_ell);

    printf("Dimension: %d\nELL Data Size: %d\nELL Cols Size: %d\n", matrix_dimension, data_ell_size, cols_ell_size);
    printf("ELL Row Size: %d\n", size_of_ell_row);
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
    fscanf(fp, "%d,%d", &data_coo_size, &size_of_coo_row);
    cudaMallocManaged(&data_coo, data_coo_size * sizeof(float));
    cudaMallocManaged(&rows_coo, data_coo_size * sizeof(int));
    cudaMallocManaged(&cols_coo, data_coo_size * sizeof(int));
    readFloatRowFormFile(fp, data_coo_size, &data_coo);
    readIntRowFromFile(fp, data_coo_size, &rows_coo);
    readIntRowFromFile(fp, data_coo_size, &cols_coo);

    printf("COO Data Size: %d\n", data_coo_size);
    printf("COO Row Size: %d\n", size_of_coo_row);
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
    readFloatRowFormFile(fp, matrix_dimension, &vector);
    printf("Vector: ");
    for (int i = 0; i < matrix_dimension; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");

    fclose(fp);
    return;
}

void readFloatRowFormFile(FILE* fp, int size_of_row, float** data) {
    int valuesRead = 0;
    char* row = (char*)malloc((size_of_row * 8 + size_of_row) * sizeof(char));
    
    fscanf(fp, "%s\n", row);
    char* ptr = strtok(row, ",");
    
    while (ptr != NULL)
    {
        (*data)[valuesRead] = atof(ptr);
        valuesRead++;
        ptr = strtok(NULL, ",");
    }

    free(row);
    return;
}

void readIntRowFromFile(FILE* fp, int size_of_row, int** data) {
    int valuesRead = 0;
    char* row = (char*)malloc((size_of_row * 6 + size_of_row) * sizeof(char));

    fscanf(fp, "%s\n", row);
    char* ptr = strtok(row, ",");
   
    while (ptr != NULL)
    {
        (*data)[valuesRead] = atoi(ptr);
        valuesRead++;
        ptr = strtok(NULL, ",");
    }
    free(row);
    return;
}