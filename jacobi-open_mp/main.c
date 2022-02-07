// gcc main.c read_matrix_csr.c -o main -fopenmp -lm
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "read_matrix_csr.h"

int checkIteration(float *x, float *y);
void evaluateSolution();
void printData();
void freeResources();
void printCurrentTime();
void printSolution(float *vector);

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

	printCurrentTime();

	dimension = readMatrixAndVectorFromFile(argv[1], &data, &cols, &row_ptr, &vector, &data_size, &row_ptr_size);
	x = calloc(dimension, sizeof(float));
	y = calloc(dimension, sizeof(float));

	double time_t = omp_get_wtime();

	printCurrentTime();

	// Schleife für die Iterationen
	for (k = 1; k < 10000; k++)
	{
		//#pragma omp parallel for
		for (int i = 0; i < row_ptr_size - 1; i++)
		{
			float row_result = 0;
			int index_of_diagonal_element;
			for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
			{
				// printf("%d ", j);
				// printf("%f ", data[j]);
				//  i und cols[j] zusammen sind das Matrixelement matrix[i][cols[j]]
				if (i != cols[j])
				{
					row_result -= (data[j] * x[cols[j]]);
				}
				else
				{
					row_result += vector[i];
					// Der Index des Diagonalelements der aktuellen Zeile (Zeile i) im data Array
					index_of_diagonal_element = j;
					// printf("%d\n", j);
				}
			}

			// Teilen durch das Diagonalelement
			y[i] = row_result / data[index_of_diagonal_element];
			// printf("\n");
		}

		// Überprüfen, ob sich ein Wert mehr als EPSILON verändert hat
		int iterationCheck = checkIteration(x, y);
		memcpy(x, y, sizeof(float) * dimension);
		if (iterationCheck != 0)
		{
			break;
		}
		// printf("%f\n", x[0]);
	}

	double time_delta = omp_get_wtime() - time_t;
	printf("Computation finished after %d iteration(s) and took %f seconds\n", k, time_delta);
	evaluateSolution();
	/*for (int i = 0; i < dimension; i++)
	{
		printf("%f\n", y[i]);
	}*/

	freeResources();
}

int checkIteration(float *x, float *y)
{
	int counter = 0;
#pragma omp parallel for shared(counter)
	for (int i = 0; i < dimension; i++)
	{
		// TODO schauen ob (fabs(x[i]) - fabs(y[i]) weglassen kann
		// Counter
		if (fabs((fabs(x[i]) - fabs(y[i]))) > EPSILON)
		{
			// printf("%d ", i);
			// printf("%f, %f, %f\n", fabs(x[i]), fabs(y[i]), fabs(x[i]) - fabs(y[i]));
			counter++;
		}
	}
	return counter == 0 ? 1 : 0;
}

void evaluateSolution()
{
#pragma omp parallel for
	for (int i = 0; i < row_ptr_size - 1; i++)
	{
		for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
		{
			y[i] = y[i] + (data[j] * x[cols[j]]);
		}
	}

	// Der Vektor y ist die Lösung die man erhält, wenn man die berechneten Quoffizienten in das Gleichungssystem (Die Matrix) eingibt

	// Compare the calculated result and the actual result
	double max_difference = -10000.0;
	double min_difference = 10000.0;
	double average_difference = 0;
	double euclidian_distance = 0;

	for (int i = 0; i < dimension; i++)
	{
		double absolute_difference = fabs(y[i] - vector[i]);
		if (absolute_difference > max_difference)
		{
			max_difference = absolute_difference;
		}
		else if (absolute_difference < min_difference)
		{
			min_difference = absolute_difference;
		}
		euclidian_distance += pow(y[i] - vector[i], 2);
		average_difference += absolute_difference;
	}

	average_difference = average_difference / dimension;
	euclidian_distance = sqrt(euclidian_distance);
	printf("Result vector:          ");
	printSolution(vector);
	printf("Calculated vector:      ");
	printSolution(y);
	printf("Max difference:     %f\n", max_difference);
	printf("Min difference:     %f\n", min_difference);
	printf("Average difference: %f\n", average_difference);
	printf("Euclidian distance: %f\n", euclidian_distance);
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

void printCurrentTime()
{
	time_t current_time;
	char *c_time_string;

	/* Obtain current time. */
	current_time = time(NULL);

	if (current_time == ((time_t)-1))
	{
		(void)fprintf(stderr, "Failure to obtain the current time.\n");
		exit(EXIT_FAILURE);
	}

	/* Convert to local time format. */
	c_time_string = ctime(&current_time);

	if (c_time_string == NULL)
	{
		(void)fprintf(stderr, "Failure to convert the current time.\n");
		exit(EXIT_FAILURE);
	}

	char *substr = malloc(10);
	strncpy(substr, c_time_string + 11, 10);
	substr[8] = '\n';
	substr[9] = '\0';

	printf("Start time is %s", substr);
}

void printSolution(float *vector)
{
	for (int i = 0; i < dimension; i++)
	{
		printf("%f ", vector[i]);
	}
	printf("\n");
}