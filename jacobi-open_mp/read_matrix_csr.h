#ifndef READ_MATRIX_CSR_H
#define READ_MATRIX_CSR_H

int readMatrixAndVectorFromFile(char *fileName, float **data, int **cols, int **row_ptr, float **vector, int *data_size, int *row_ptr_size);
#endif