#include <stdio.h>
#include <stdlib.h>
#include "mm-header.h"
#include <time.h>
#include <omp.h>
#include "mpi.h"

//Benjamin Creem
//May 23 2018
int main(int argc, char *argv[]){
    int n = 10; //matrixes are n x n
	
    //Allocating Memory and Assigning Values

    //Start MPI
    //Finding Matrix Product  and Printing
    int numranks, rank, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    double *mat1 = (double *)malloc(n*n*sizeof(double));
    double *mat2 = (double *)malloc(n*n*sizeof(double));

    double startTime;
    double endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &len);
   
    double *scatterMat = (double *)malloc((n+1)*sizeof(double));
    double *gatherMat = (double *)malloc((n+1)*sizeof(double));
    double *result = (double *)malloc(n*n*sizeof(double));

    if(rank == 0 && n % numranks != 0)
    {
        printf("N: %d, numranks: %d\n", n, numranks);
        exit(1);
    }

    

    if(rank == 0)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < n; j++)
            {
                mat1[i*n+j] = 2*i+j;
                mat2[i*n+j] = i+3*j;
            }
        }
        printf("Number of tasks is %d\n", numranks);
    }
    //Matrix B is copied to every processor
    MPI_Bcast(mat2, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //Print matricies
    //if(rank == 0)
    //{
       // printf("Matrix A\n");
       // printMat(mat1, n);
       // printf("Matrix B\n");
      //  printMat(mat2, n);
    //}
    
    //Matrix A is divided into blocks along the rows and distributed
    //among processors. 
    int root = 0;
    startTime = MPI_Wtime();
    for(int x = 0; x < (n/numranks); x++)
    {
        MPI_Scatter(&mat1[x*n], n, MPI_DOUBLE, scatterMat, n, MPI_DOUBLE, root, MPI_COMM_WORLD); 
        MPI_Barrier(MPI_COMM_WORLD);

	    double sum = 0.0;

    	//Multiply 
    	for(int i = 0; i < n; i++)
    	{
        	for(int j = 0; j < n; j++)
        	{
            		sum = sum + mat2[j*n+i] * scatterMat[j];
        	}
        	gatherMat[i] = sum;
        	sum = 0.0;
    	}

    	MPI_Gather(gatherMat, n, MPI_DOUBLE, &result[x*n], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    	MPI_Barrier(MPI_COMM_WORLD);
    }
    
    endTime = MPI_Wtime();
    if(rank == 0)
        printf("Time to complete %f\n", endTime - startTime);
    
    if(rank == 0)
    {
        printf("Result\n");
        printMat(result, n);
    }

    MPI_Finalize();

    return 0;
}


//Free memory used by first matrix
void freeMat(double** mat, int n)
{
	for(int i=0; i<n; i++)
	{
		free(mat[i]);
	}
	free(mat);
}

//Allocate memory for first matrix

//Print singular matrix
void printMat(double* mat, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%.2f\t", mat[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
}

//Calculate matrix product 
double* matMultiply(double *mat1, double* mat2, int n)
{
	//Return matrix
	double *result = (double *)malloc(n*n*sizeof(double*));
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            result[i*n+j] = 0;
            for(int k = 0; k < n; k++)
            {
                result[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
            }
        }
    }
    return result;
}

 
