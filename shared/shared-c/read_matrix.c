#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read_matrix.h"

int readMatrixAndVectorFromFile(char *fileName, float **matrix, float **vector)
{
    int dimension = 0;
    int MAX_FLOAT_STRING_LENGTH = 8;
    FILE *fp;
    fp = fopen(fileName, "r");
    fscanf(fp, "%d", &dimension);

    char *row = malloc((dimension * MAX_FLOAT_STRING_LENGTH + dimension - 1) * sizeof(char));

    *matrix = malloc(dimension * dimension * sizeof(float));
    *vector = malloc(dimension * sizeof(float));

    for (int i = 0; i < dimension; i++)
    {
        fscanf(fp, "%s\n", row);
        int fromIndex = 0;
        int floatsRead = 0;
        for (int k = 0; k < strlen(row); k++)
        {
            if (row[k] == ',')
            {
                char *substr = malloc(MAX_FLOAT_STRING_LENGTH);
                strncpy(substr, row + fromIndex, k - fromIndex - 1);
                (*matrix)[i * dimension + floatsRead] = atof(substr);
                floatsRead++;
                fromIndex = k + 1;
            }
        }
        char *substr = malloc(MAX_FLOAT_STRING_LENGTH);
        strncpy(substr, row + fromIndex, strlen(row) - fromIndex - 1);
        (*matrix)[i * dimension + floatsRead] = atof(substr);
    }

    fscanf(fp, "%s\n", row);
    int fromIndex = 0;
    int floatsRead = 0;
    for (int k = 0; k < strlen(row); k++)
    {
        if (row[k] == ',')
        {
            char *substr = malloc(MAX_FLOAT_STRING_LENGTH);
            strncpy(substr, row + fromIndex, k - fromIndex - 1);
            (*vector)[floatsRead] = atof(substr);
            floatsRead++;
            fromIndex = k + 1;
        }
    }
    char *substr = malloc(MAX_FLOAT_STRING_LENGTH);
    strncpy(substr, row + fromIndex, strlen(row) - fromIndex - 1);
    (*vector)[floatsRead] = atof(substr);

    fclose(fp);
    return dimension;
}