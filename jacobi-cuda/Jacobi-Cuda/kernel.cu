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
void offset(int* offset_array, int* rows_coo, int data_coo_size, int size_of_coo_row);
void evaluateSolution();
__global__ void jacobi(int matrix_dimension, int* offset_array, float* data_ell, int* cols_ell, float* x, float* y, int data_ell_size, float* vector, int data_coo_size, int size_of_coo_row, float* data_coo, int* cols_coo);
__global__ void init_vector_with_zero(int matrix_dimension, float* vector);
__global__ void check_iteration(int matrix_dimension, float EPSILON, float* x, float* y, bool* result);
__global__ void matrix_vector_mult(int matrix_dimension, int* offset_array, float* data_ell, int* cols_ell, float* x, float* y, int data_ell_size, int data_coo_size, int size_of_coo_row, float* data_coo, int* cols_coo);

const char* matrix_file_name = "matrix_ell_coo_4.csv";
int matrix_dimension = 0;
// Die Dimension des Grids bei einer Blocksize von 1024
int grid_dimension = 0;

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

// Array das angibt, ob eine Zeile eine COO oder ELL Zeile ist und wie groß der Offset der Zeile ist
int* offset_array = NULL;

// Vektoren für die Zwischenergebnisse
float* x = NULL;
float* y = NULL;

// Schwellwert für Änderungen (Hatte keinen großen Einfluss auf die Anzahl der Iterationen)
float EPSILON = 0.0001;

/* Gibt an ob sich mindestens ein Wert aus der letzten Iteration um mindestens Epsilon verändert hat
*  Bei true sollte eine weitere Iteration gestartet werden, bei false kann abbgebrochen werden
*/
bool* did_iteration_change_more_than_epsilon;

cudaError_t error;
cudaEvent_t start, stop;
float milliseconds = 0;

int main()
{
    cudaDeviceProp cdp;
    cudaGetDeviceProperties(&cdp, 0);
    printf("Device name:           %s\n", cdp.name);
    printf("Max Threads per Block: %d\n", cdp.maxThreadsPerBlock);

    // Lese Matrix und Vektor aus der Eingabedatei
    readMatrixAndVectorFromFile();

    // Berechne wie groß die Grid Dimension sein muss (bei BlockDim 1024)
    grid_dimension = calculate_grid_dimension(matrix_dimension);

    // Initialisiere den Ergebnisvektor
    cudaMallocManaged(&x, matrix_dimension * sizeof(float));
    init_vector_with_zero <<<grid_dimension, 1024 >>>(matrix_dimension, x);
    cudaDeviceSynchronize();

    // Vektor für das Zwischenergebnis
    cudaMallocManaged(&y, matrix_dimension * sizeof(float));
    init_vector_with_zero << <grid_dimension, 1024 >> > (matrix_dimension, y);
    cudaDeviceSynchronize();
    cudaMallocHost(&did_iteration_change_more_than_epsilon, sizeof(bool));

    // Initializiere und berechne Offset Array
    cudaMallocManaged(&offset_array, matrix_dimension * sizeof(int));
    offset(offset_array, rows_coo, data_coo_size, size_of_coo_row);

    // Events zum Messen der Zeit
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start der Zeitmessung
    cudaEventRecord(start);

    // Starte die Jacobi Iterationen
    int k;
    for (k = 1; k < 100; k++) {
        jacobi<<<grid_dimension, 1024>>> (matrix_dimension, offset_array, data_ell, cols_ell, x, y, data_ell_size, vector, data_coo_size, size_of_coo_row, data_coo, cols_coo);
        error = cudaDeviceSynchronize();
        check_iteration<<<grid_dimension, 1024 >>>(matrix_dimension, EPSILON, x, y, did_iteration_change_more_than_epsilon);
        error = cudaDeviceSynchronize();
        if (!(*did_iteration_change_more_than_epsilon)) {
            break;
        }
        else {
            *did_iteration_change_more_than_epsilon = false;
        }
        float* tmp = x;
        x = y;
        y = tmp;
        error = cudaDeviceSynchronize();
    }

    // Ende der Zeitmessung
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Computation finished after %f milliseconds and took %d iteration(s)\n", milliseconds, k);
    // Das Ergebnis steht in y, da in der letzten Iteration x und y nicht mehr getauscht werden
    evaluateSolution();

    cudaFree(data_ell);
    cudaFree(cols_ell);
    cudaFree(data_coo);
    cudaFree(rows_coo);
    cudaFree(cols_coo);
    cudaFree(vector);
    cudaFree(x);
    cudaFree(y);
    cudaFreeHost(did_iteration_change_more_than_epsilon);

    return 0;
}

__global__ void jacobi(int matrix_dimension, int* offset_array, float* data_ell, int* cols_ell, float* x, float* y, int data_ell_size, float* vector, int data_coo_size, int size_of_coo_row, float* data_coo, int* cols_coo)
{
    // Jeder Thread berechnet eine Zeile
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < matrix_dimension) {
        int index_of_diagonal_element;

        y[idx] = 0.0;

        // Wenn der Wert des offset Arrays < 0 ist, dann ist es eine COO Zeile und der Wert (+1) gibt an die wie vielte COO Zeile es ist.
        if (offset_array[idx] < 0) {
            int index_of_coo_row = (offset_array[idx] + 1) * -1;
            for (int m = index_of_coo_row * size_of_coo_row; m < (index_of_coo_row * size_of_coo_row) + size_of_coo_row; m++) {

                if (idx != cols_coo[m]) {
                    y[idx] -= data_coo[m] * x[cols_coo[m]];
                }
                else {
                    y[idx] += vector[idx];
                    index_of_diagonal_element = m;
                }
            }
            y[idx] = y[idx] / data_coo[index_of_diagonal_element];
        }
        else {
            // // Wenn der Wert des offset Arrays >= 0 ist, dann ist es eine ELL Zeile und der Wert gibt den Offset der Zeile an
            for (int i = idx - offset_array[idx]; i < data_ell_size; i = i + matrix_dimension - (data_coo_size / size_of_coo_row)) {
                if (idx != cols_ell[i]) {
                    y[idx] -= data_ell[i] * x[cols_ell[i]];
                }
                else {
                    y[idx] += vector[idx];
                    index_of_diagonal_element = i;
                }
            }
            y[idx] = y[idx] / data_ell[index_of_diagonal_element];
        }
    }
}

/*
        Das Ergebnis des Jacobi Algorithmus steht in y
        Berechne x = A * y (Matrix A mal berechneten Vektor y) mit Kernel
        Vergleiche x und vector und bewerte die Lösung
*/
void evaluateSolution() {
    matrix_vector_mult<<<grid_dimension, 1024>>> (matrix_dimension, offset_array, data_ell, cols_ell, x, y, data_ell_size, data_coo_size, size_of_coo_row, data_coo, cols_coo);
    error = cudaDeviceSynchronize();

    float max_difference = -10000.0;
    float min_difference = 10000.0;;
    float average_difference = 0.0;
    float euclidian_distance = 0.0;

    for (int i = 0; i < matrix_dimension; i++)
    {
        //printf("x[%d]=%f\n",i, x[i]);
        //printf("vector[%d]=%f\n", i, vector[i]);
        float absolute_difference = fabs(x[i] - vector[i]);
        if (absolute_difference > max_difference)
        {
            max_difference = absolute_difference;
        }
        else if (absolute_difference < min_difference)
        {
            min_difference = absolute_difference;
        }
        euclidian_distance += pow(x[i] - vector[i], 2);
        average_difference += absolute_difference;
    }

    average_difference = average_difference / matrix_dimension;
    euclidian_distance = sqrt(euclidian_distance);

    printf("Max difference:     %f\n", max_difference);
    printf("Min difference:     %f\n", min_difference);
    printf("Average difference: %f\n", average_difference);
    printf("Euclidian distance: %f\n", euclidian_distance);
}

void offset(int* offset_array, int* rows_coo, int data_coo_size, int size_of_coo_row) {

    int currentOffset = 0;
    int currentCooRowIndex = rows_coo[0];

    for (int i = 0; i < matrix_dimension; i++) {
        
        if (i == currentCooRowIndex)
        {
            offset_array[i] = (-1 * currentOffset) - 1;
            currentOffset++;
            if (currentOffset * size_of_coo_row < data_coo_size) {
                currentCooRowIndex = rows_coo[currentOffset * size_of_coo_row];
            }
        }
        else {
            offset_array[i] = currentOffset;
        }
    }
}

__global__ void check_iteration(int matrix_dimension, float EPSILON, float* x, float* y, bool* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < matrix_dimension) {
        if (fabs(x[idx] - y[idx]) > EPSILON) {
            *result = true;
        }
    }
}

__global__ void matrix_vector_mult(int matrix_dimension, int* offset_array, float* data_ell, int* cols_ell, float* x, float* y, int data_ell_size, int data_coo_size, int size_of_coo_row, float* data_coo, int* cols_coo) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < matrix_dimension) {
        x[idx] = 0.0;

        if (offset_array[idx] < 0) {
            int index_of_coo_row = (offset_array[idx] + 1) * -1;
            for (int m = index_of_coo_row * size_of_coo_row; m < (index_of_coo_row * size_of_coo_row) + size_of_coo_row; m++) {
                x[idx] += data_coo[m] * y[cols_coo[m]];
            }
        }
        else {
            for (int i = idx - offset_array[idx]; i < data_ell_size; i = i + matrix_dimension - (data_coo_size / size_of_coo_row)) {
                x[idx] += data_ell[i] * y[cols_ell[i]];
            }
        }
    }
}

int calculate_grid_dimension(int dimension) {
    return (int)ceil(dimension / static_cast<double>(1024));
}

__global__ void init_vector_with_zero(int matrix_dimension, float* vector) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < matrix_dimension) {
        vector[idx] = 0.0;
    }
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

    // Lese COO Format
    fscanf(fp, "%d,%d", &data_coo_size, &size_of_coo_row);
    cudaMallocManaged(&data_coo, data_coo_size * sizeof(float));
    cudaMallocManaged(&rows_coo, data_coo_size * sizeof(int));
    cudaMallocManaged(&cols_coo, data_coo_size * sizeof(int));
    readFloatRowFormFile(fp, data_coo_size, &data_coo);
    readIntRowFromFile(fp, data_coo_size, &rows_coo);
    readIntRowFromFile(fp, data_coo_size, &cols_coo);

    // Lese Vektor
    readFloatRowFormFile(fp, matrix_dimension, &vector);

    /*printf("Dimension: %d\nELL Data Size: %d\nELL Cols Size: %d\n", matrix_dimension, data_ell_size, cols_ell_size);
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
 
    printf("Vector: ");
    for (int i = 0; i < matrix_dimension; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");*/

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