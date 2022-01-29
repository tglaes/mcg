// gcc main.c read_matrix_csr.c -o main -fopenmp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include "read_matrix_csr.h"

int checkIteration(float *x, float *y);
void evaluateSolution();
void printData();
void freeResources();

int dimension = 0;
int data_size = 0;
int row_ptr_size = 0;
float *data = NULL;
int *cols = NULL;
int *row_ptr = NULL;
float *x;
float *y;
float *vector = NULL;
int k;
float EPSILON = 0.0001;

int main(int argc, char *argv[])
{
	printf("Jacobi OpenMP\n");
	printf("Number of threads=%d\n", omp_get_max_threads());

	if (argc != 2)
	{
		printf("You need to provide a matrix as argument!\n");
		return 1;
	}
	dimension = readMatrixAndVectorFromFile(argv[1], &data, &cols, &row_ptr, &vector, &data_size, &row_ptr_size);
	float *x = calloc(dimension, sizeof(float));
	float *y = calloc(dimension, sizeof(float));

	double time_t = omp_get_wtime();

	// Schleife für die Iterationen
	for (k = 1; k < 1000; k++)
	{
		for (int i = 0; i < row_ptr_size - 1; i++)
		{
			float row_result = 0;
			int index_of_diagonal_element;
			for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
			{
				//printf("%d ", j);
				//printf("%f ", data[j]);
				// i und cols[j] zusammen sind das Matrixelement matrix[i][cols[j]]
				if (i != cols[j])
				{
					row_result -= (data[j] * x[i]);
				}
				else
				{
					row_result += vector[i];
					// Der Index des Diagonalelements der aktuellen Zeile (Zeile i) im data Array
					index_of_diagonal_element = j;
					//printf("%d\n", j);
				}
			}

			// Teilen durch das Diagonalelement
			y[i] = row_result / data[index_of_diagonal_element];
			//printf("\n");
		}

		// Überprüfen, ob sich ein Wert mehr als EPSILON verändert hat
		int iterationCheck = checkIteration(x, y);
		memcpy(x, y, sizeof(float) * dimension);
		if (iterationCheck != 0)
		{
			break;
		}
		//printf("%f\n", x[0]);
	}

	double time_delta = omp_get_wtime() - time_t;
	printf("Computation finished after %d iteration(s) and took %f seconds\n", k, time_delta);

	for (int i = 0; i < dimension; i++)
	{
		printf("%f\n", y[i]);
	}

	freeResources();
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

void evaluateSolution()
{
}

void printData()
{
	for (int k = 0; k < data_size; k++)
	{
		printf("%f,", data[k]);
	}
	printf("\n");

	for (int k = 0; k < data_size; k++)
	{
		printf("%d,", cols[k]);
	}
	printf("\n");

	for (int k = 0; k < row_ptr_size; k++)
	{
		printf("%d,", row_ptr[k]);
	}
	printf("\n");

	for (int k = 0; k < dimension; k++)
	{
		printf("%f,", vector[k]);
	}
	printf("\n");
}

void freeResources()
{
	free(data);
	free(cols);
	free(row_ptr);
	free(x);
	free(y);
	free(vector);
}