#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {

    int size, rank, value, rc;

    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n"); 
        MPI_Abort(MPI_COMM_WORLD, rc); 
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size < 2) {
        fprintf(stdout, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }

    if (rank == 0) {
        value = 0;
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", 0, value);

        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received %d\n", 0, value);
    } else if (rank == size - 1) {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received %d\n", rank, value);

        value++;

        MPI_Send(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", rank, value);
    } else {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received %d\n", rank, value);

        value++;

        MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", rank, value);

        MPI_Recv(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d received %d\n", rank, value);

        value++;

        MPI_Send(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        printf("Rank %d sent %d\n", rank, value);
    }

    MPI_Finalize();

    return 0;
}
