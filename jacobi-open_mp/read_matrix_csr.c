#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read_matrix_csr.h"

int readMatrixAndVectorFromFile(char *fileName, float **data, int **cols, int **row_ptr, float **vector, int *data_size, int *row_ptr_size)
{
    int dimension = 0;
    int MAX_FLOAT_STRING_LENGTH = 8;
    FILE *fp;
    fp = fopen(fileName, "r");
    fscanf(fp, "%d,%d,%d", &dimension, data_size, row_ptr_size);

    *data = malloc(*data_size * sizeof(float));
    *cols = malloc(*data_size * sizeof(int));
    *row_ptr = malloc(*row_ptr_size * sizeof(int));
    *vector = malloc(dimension * sizeof(float));

    // Read the data row
    char *row = malloc((*data_size * MAX_FLOAT_STRING_LENGTH + *data_size - 1) * sizeof(char));
    fscanf(fp, "%s\n", row);
    int fromIndex = 0;
    int floatsRead = 0;
    for (int k = 0; k < strlen(row); k++)
    {
        if (row[k] == ',')
        {
            char *substr = malloc(MAX_FLOAT_STRING_LENGTH);
            strncpy(substr, row + fromIndex, k - fromIndex - 1);
            (*data)[floatsRead] = atof(substr);
            floatsRead++;
            fromIndex = k + 1;
        }
    }

    char *substr = malloc(MAX_FLOAT_STRING_LENGTH);
    strncpy(substr, row + fromIndex, strlen(row) - fromIndex - 1);
    (*data)[floatsRead] = atof(substr);

    // Read the cols row
    char *row_cols = malloc((*data_size * MAX_FLOAT_STRING_LENGTH + *data_size - 1) * sizeof(char));
    fscanf(fp, "%s\n", row_cols);
    fromIndex = 0;
    int intsRead = 0;
    for (int k = 0; k < strlen(row_cols); k++)
    {
        if (row_cols[k] == ',')
        {
            substr = malloc(MAX_FLOAT_STRING_LENGTH);
            strncpy(substr, row_cols + fromIndex, k - fromIndex);
            (*cols)[intsRead] = atoi(substr);
            intsRead++;
            fromIndex = k + 1;
        }
    }

    substr = malloc(MAX_FLOAT_STRING_LENGTH);
    strncpy(substr, row_cols + fromIndex, strlen(row_cols) - fromIndex);
    (*cols)[intsRead] = atoi(substr);

    // Read the row_ptr row
    char *row_row_ptr = malloc((*row_ptr_size * MAX_FLOAT_STRING_LENGTH + *row_ptr_size - 1) * sizeof(char));
    fscanf(fp, "%s\n", row_row_ptr);
    fromIndex = 0;
    intsRead = 0;
    for (int k = 0; k < strlen(row_row_ptr); k++)
    {
        if (row_row_ptr[k] == ',')
        {
            substr = malloc(MAX_FLOAT_STRING_LENGTH);
            strncpy(substr, row_row_ptr + fromIndex, k - fromIndex);
            (*row_ptr)[intsRead] = atoi(substr);
            intsRead++;
            fromIndex = k + 1;
        }
    }

    substr = malloc(MAX_FLOAT_STRING_LENGTH);
    strncpy(substr, row_row_ptr + fromIndex, strlen(row_row_ptr) - fromIndex);
    (*row_ptr)[intsRead] = atoi(substr);

    // Read the vector
    fscanf(fp, "%s\n", row);
    fromIndex = 0;
    floatsRead = 0;
    for (int k = 0; k < strlen(row); k++)
    {
        if (row[k] == ',')
        {
            substr = malloc(MAX_FLOAT_STRING_LENGTH);
            strncpy(substr, row + fromIndex, k - fromIndex - 1);
            (*vector)[floatsRead] = atof(substr);
            floatsRead++;
            fromIndex = k + 1;
        }
    }
    substr = malloc(MAX_FLOAT_STRING_LENGTH);
    strncpy(substr, row + fromIndex, strlen(row) - fromIndex - 1);
    (*vector)[floatsRead] = atof(substr);

    fclose(fp);

    return dimension;
}