#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "read_matrix.h"

void printMatrixAndVector(float *matrix, float *vector);
void printSolution(float *vector);
int checkIteration(float *x, float *y);

clock_t begin, end;
double delta;
float EPSILON = 0.0001;
int dimension = 0;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("You need to provide a matrix as argument!\n");
        return 1;
    }

    float *matrix = NULL;
    float *vector = NULL;
    dimension = readMatrixAndVectorFromFile(argv[1], &matrix, &vector);

    //printMatrixAndVector(matrix, vector, dimension);

    float *x = calloc(dimension, sizeof(float));
    float *y = calloc(dimension, sizeof(float));

    begin = clock();
    int k;
    for (k = 0; k < 1000; k++)
    {
        for (int i = 0; i < dimension; i++)
        {
            float row_result = 0;
            for (int j = 0; j < dimension; j++)
            {
                if (i != j)
                {
                    row_result -= (matrix[i * dimension + j] * x[i]) / matrix[i * dimension + i];
                }
                else
                {
                    row_result += vector[i] / matrix[i * dimension + i];
                }
            }
            y[i] = row_result;
        }
        // Check if values changed by more than EPLISON
        int iterationCheck = checkIteration(x, y);
        memcpy(x, y, sizeof(float) * dimension);
        if (iterationCheck != 0)
        {
            break;
        }
        //printf("%f\n", x[0]);
    }

    end = clock();
    delta = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Computation finished after %d iterations and took %f seconds\n", k, delta);

    printSolution(x);

    //evaluateSolution();

    return 0;
}

int checkIteration(float *x, float *y)
{
    for (int i = 0; i < dimension; i++)
    {
        if (fabs((fabs(x[i]) - fabs(y[i]))) > EPSILON)
        {
            //printf("%d ", i);
            //printf("%f, %f, %f\n", fabs(x[i]), fabs(y[i]), fabs(x[i]) - fabs(y[i]));
            return 0;
        }
    }
    return 1;
}

void printMatrixAndVector(float *matrix, float *vector)
{
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
}

void printSolution(float *vector)
{
    for (int i = 0; i < dimension; i++)
    {
        printf("%f ", vector[i]);
    }
    printf("\n");
}