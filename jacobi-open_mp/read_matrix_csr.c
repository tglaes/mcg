#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read_matrix_csr.h"

int readMatrixAndVectorFromFile(char *fileName, float **data, int **cols, int **row_ptr, float **vector, int *data_size, int *row_ptr_size)
{
    int dimension = 0;
    FILE *fp;
    fp = fopen(fileName, "r");
    fscanf(fp, "%d,%d,%d", &dimension, data_size, row_ptr_size);

    *data = malloc(*data_size * sizeof(float));
    *cols = malloc(*data_size * sizeof(int));
    *row_ptr = malloc(*row_ptr_size * sizeof(int));
    *vector = malloc(dimension * sizeof(float));

    int valuesRead = 0;

    // Read the data row
    char *row = malloc((*data_size * 10) * sizeof(char));
    fscanf(fp, "%s\n", row);

    char *ptr = strtok(row, ",");
    while (ptr != NULL)
    {
        (*data)[valuesRead] = atof(ptr);
        valuesRead++;
        ptr = strtok(NULL, ",");
    }

    // Read the cols row
    char *row_cols = malloc((*data_size * 10) * sizeof(char));
    fscanf(fp, "%s\n", row_cols);

    ptr = strtok(row_cols, ",");
    valuesRead = 0;
    while (ptr != NULL)
    {
        (*cols)[valuesRead] = atoi(ptr);
        valuesRead++;
        ptr = strtok(NULL, ",");
    }

    // Read the row_ptr row
    char *row_row_ptr = malloc((*row_ptr_size * 10) * sizeof(char));
    fscanf(fp, "%s\n", row_row_ptr);
    ptr = strtok(row_row_ptr, ",");
    valuesRead = 0;
    while (ptr != NULL)
    {
        (*row_ptr)[valuesRead] = atoi(ptr);
        valuesRead++;
        ptr = strtok(NULL, ",");
    }

    // Read the vector
    char *row_vector = malloc((dimension * 15) * sizeof(char));
    fscanf(fp, "%s\n", row_vector);
    ptr = strtok(row_vector, ",");
    valuesRead = 0;
    while (ptr != NULL)
    {
        (*vector)[valuesRead] = atof(ptr);
        valuesRead++;
        ptr = strtok(NULL, ",");
    }

    fclose(fp);
    free(row);
    free(row_cols);
    free(row_row_ptr);
    free(row_vector);

    return dimension;
}