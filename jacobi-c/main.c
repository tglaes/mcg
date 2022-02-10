// gcc main.c read_matrix.c -o main -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "read_matrix.h"

void printMatrixAndVector(float *matrix, float *vector);
void printSolution(float *vector);
int checkIteration(float *x, float *y);
void evaluateSolution(float *matrix, float *vector, float *x);

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

    // printMatrixAndVector(matrix, vector);

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
                    row_result -= (matrix[i * dimension + j] * x[i]);
                }
                else
                {
                    row_result += vector[i];
                }
            }
            y[i] = row_result / matrix[i * dimension + i];
        }
        // Check if values changed by more than EPLISON
        int iterationCheck = checkIteration(x, y);
        memcpy(x, y, sizeof(float) * dimension);
        if (iterationCheck != 0)
        {
            break;
        }
        // printf("%f\n", x[0]);
    }

    end = clock();
    delta = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Computation finished after %d iterations and took %f seconds\n", k, delta);

    // printf("Solution:               ");
    // printSolution(x);

    evaluateSolution(matrix, vector, x);
    free(matrix);
    free(vector);
    free(x);
    free(y);
    return 0;
}

int checkIteration(float *x, float *y)
{
    for (int i = 0; i < dimension; i++)
    {
        if (fabs((fabs(x[i]) - fabs(y[i]))) > EPSILON)
        {
            // printf("%d ", i);
            // printf("%f, %f, %f\n", fabs(x[i]), fabs(y[i]), fabs(x[i]) - fabs(y[i]));
            return 0;
        }
    }
    return 1;
}

void evaluateSolution(float *matrix, float *vector, float *x)
{
    float *calculated_result = calloc(dimension, sizeof(float));

    // Calculate the result of matrix * x
    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            calculated_result[i] += matrix[i * dimension + j] * x[i];
        }
    }

    // Compare the calculated result and the actual result
    double max_difference = -10000.0;
    double min_difference = 10000.0;
    double average_difference = 0;
    double euclidian_distance = 0;

    for (int i = 0; i < dimension; i++)
    {
        double absolute_difference = fabs(calculated_result[i] - vector[i]);
        if (absolute_difference > max_difference)
        {
            max_difference = absolute_difference;
        }
        else if (absolute_difference < min_difference)
        {
            min_difference = absolute_difference;
        }
        euclidian_distance += pow(calculated_result[i] - vector[i], 2);
        average_difference += absolute_difference;
    }

    average_difference = average_difference / dimension;
    euclidian_distance = sqrt(euclidian_distance);
    // printf("Result vector:          ");
    // printSolution(vector);
    // printf("Calculated vector:      ");
    // printSolution(calculated_result);
    printf("Max difference:     %f\n", max_difference);
    printf("Min difference:     %f\n", min_difference);
    printf("Average difference: %f\n", average_difference);
    printf("Euclidian distance: %f\n", euclidian_distance);

    free(calculated_result);
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