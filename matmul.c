#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define SIZE 4
#define FROM_MASTER 1
#define FROM_WORKER 2
double a[SIZE][SIZE];
double b[SIZE][SIZE];
double c[SIZE][SIZE];

void initialize_matrix(void){

  int i,j;
  for(i=0; i < SIZE; i++){
    for(j=0; j < SIZE; j++){
      a[i][j]=2.0;
      b[i][j]=2.0;
    }
  }

}

void print_matrix(void){
  int i, j;
  printf("Final Result of Matrix Multiplication is\n");
  for (i=0; i < SIZE; i++){
    for (j=0; j < SIZE; j++){
      printf("%f\t", c[i][j]);
    }
      printf("\n");

  }

}



int main(int argc, char **argv){
  int rank, np;
  int rows;
  int i,j,k;
  int message, offset;
  double start_time, end_time;
  MPI_Comm comm=MPI_COMM_WORLD;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &rank);
  //****************************//
  // From Master
  
  if(!rank){
    printf("Total No. of Processor is=%d and SIZE=%d\n", np, SIZE);
    initialize_matrix();
    start_time=MPI_Wtime();
    rows = SIZE/np;
    message=FROM_MASTER;
    offset=rows;
    for (i=1; i < np; i++){
      printf("Sending %d Rows to rank %d\n", rows, i);
      MPI_Send(&rows, 1, MPI_INT, i, message, comm);
      MPI_Send(&offset, 1, MPI_INT, i, message, comm);
      MPI_Send(&a[offset][0], rows*SIZE, MPI_DOUBLE, i, message, comm);
      MPI_Send(&b, SIZE*SIZE, MPI_DOUBLE, i, message, comm);
      offset +=rows;
    }

    for (i=0; i < rows; i++){
      for(j=0; j < SIZE; j++){
	c[i][j]=0;
	for(k=0; k < SIZE; k++){
	  c[i][j]=c[i][j] + a[i][k]*b[k][j];
	  
       }
      }
    }

    message=FROM_WORKER;
    for(i=1; i < np; i++){
      MPI_Recv(&offset, 1, MPI_INT, i, message, comm, &status);
      MPI_Recv(&rows, 1, MPI_INT, i, message, comm, &status);
      MPI_Recv(&c[offset][0], rows*SIZE, MPI_DOUBLE, i, message, comm, &status);
      printf("Received %d rows from rank %d, offset=%d\n", rows, i, offset);

    }
    end_time=MPI_Wtime();
    print_matrix();

}else {
    // From Worker.......//

    message=FROM_MASTER;
      MPI_Recv(&rows, 1, MPI_INT, 0, message, comm, &status);
      MPI_Recv(&offset, 1, MPI_INT, 0, message, comm, &status);
      MPI_Recv(&a[offset][0], rows*SIZE, MPI_DOUBLE, 0, message, comm, &status);
      MPI_Recv(&b, SIZE*SIZE, MPI_DOUBLE, 0, message, comm, &status);

      for(i=offset; i < offset+rows; i++){
	for(j=0; j < SIZE; j++)
	{
	  c[i][j]=0.0;

	  for(k=0; k < SIZE; k++){
            c[i][j]=c[i][j] + a[i][k]*b[k][j];
	  }
	}
      }
      message=FROM_WORKER;
      MPI_Send(&rows, 1, MPI_INT, 0, message, comm);
      MPI_Send(&offset, 1, MPI_INT, 0, message, comm);
      MPI_Send(&c[offset][0], rows*SIZE, MPI_DOUBLE, 0, message, comm);

      


}   
  MPI_Finalize();
  return 0;

}
