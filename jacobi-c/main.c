#include <stdio.h>
#include "read_matrix.h"

int main()
{
    float *matrix = NULL;
    float *vector = NULL;
    int dimension = readMatrixAndVectorFromFile("../shared/matrix-generation/matrix.csv", &matrix, &vector);

    for (int k = 0; k < dimension; k++)
    {
        for (int j = 0; j < dimension; j++)
        {
            printf("%.6f ", matrix[k * dimension + j]);
        }
        printf("\n");
    }

    for (int k = 0; k < dimension; k++)
    {
        printf("%.6f ", vector[k]);
    }
    printf("\n");

    float x[3] = {0, 0, 0};
    float y[3] = {0, 0, 0};

    for (int i = 0; i < 10000; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float row_result = 0;
            for (int k = 0; k < 3; k++)
            {
                if (j != k)
                {
                    row_result -= (matrix[j * 3 + k] * x[j]) / matrix[j * 3 + j];
                }
                else
                {
                    row_result += vector[j] / matrix[j * 3 + j];
                }
            }
            y[j] = row_result;
        }
        x[0] = y[0];
        x[1] = y[1];
        x[2] = y[2];
    }

    for (int i = 0; i < 3; i++)
    {
        printf("%f ", x[i]);
    }
    printf("\n");
    return 0;
}