// gcc main.c read_matrix_csr.c -o main -fopenmp
#include <stdio.h>
#include <omp.h>
#include "read_matrix_csr.h"

int main(int argc, char *argv[])
{
	printf("Jacobi OpenMP\n");
	printf("Number of threads=%d\n", omp_get_max_threads());

	if (argc != 2)
	{
		printf("You need to provide a matrix as argument!\n");
		return 1;
	}

	int dimension = 0;
	int data_size = 0;
	int row_ptr_size = 0;
	float *data = NULL;
	int *cols = NULL;
	int *row_ptr = NULL;
	float *vector = NULL;
	dimension = readMatrixAndVectorFromFile(argv[1], &data, &cols, &row_ptr, &vector, &data_size, &row_ptr_size);

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
