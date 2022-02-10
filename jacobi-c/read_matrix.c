#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read_matrix.h"

int readMatrixAndVectorFromFile(char *fileName, float **matrix, float **vector)
{
    int dimension = 0;
    FILE *fp;
    fp = fopen(fileName, "r");
    fscanf(fp, "%d", &dimension);
    int valuesRead = 0;
    char *ptr;

    char *row = malloc((dimension * 10 + dimension - 1) * sizeof(char));
    *matrix = malloc(dimension * dimension * sizeof(float));
    *vector = malloc(dimension * sizeof(float));

    for (int i = 0; i < dimension; i++)
    {
        fscanf(fp, "%s\n", row);
        ptr = strtok(row, ",");
        valuesRead = 0;
        while (ptr != NULL)
        {
            (*matrix)[i * dimension + valuesRead] = atof(ptr);
            valuesRead++;
            ptr = strtok(NULL, ",");
        }
    }

    valuesRead = 0;
    fscanf(fp, "%s\n", row);
    ptr = strtok(row, ",");
    valuesRead = 0;
    while (ptr != NULL)
    {
        (*vector)[valuesRead] = atof(ptr);
        valuesRead++;
        ptr = strtok(NULL, ",");
    }

    fclose(fp);
    return dimension;
}