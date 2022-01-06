#include<stdio.h>
#include<omp.h>

int main(int argc, char *argv[]) {
	printf("Number of threads=%d\n",omp_get_max_threads());
	printf("Jacobi open mp\n");
}
