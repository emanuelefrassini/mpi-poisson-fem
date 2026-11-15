/* Single-process MPI shim used ONLY for compilation/smoke tests without MPI. */
#ifndef TEST_MPI_SHIM_H
#define TEST_MPI_SHIM_H
#include <stddef.h>
#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL (-2)
#define MPI_PROC_NULL (-1)
#define MPI_DOUBLE 1
#define MPI_SUM 2
#define MPI_MAX 3
#define MPI_STATUS_IGNORE NULL
typedef int MPI_Comm;
int MPI_Init(int *, char ***);
int MPI_Finalize(void);
int MPI_Abort(MPI_Comm, int);
int MPI_Comm_size(MPI_Comm, int *);
int MPI_Comm_rank(MPI_Comm, int *);
int MPI_Dims_create(int, int, int *);
int MPI_Cart_create(MPI_Comm, int, const int *, const int *, int, MPI_Comm *);
int MPI_Cart_coords(MPI_Comm, int, int, int *);
int MPI_Cart_shift(MPI_Comm, int, int, int *, int *);
int MPI_Comm_free(MPI_Comm *);
int MPI_Sendrecv(const void *, int, int, int, int, void *, int, int, int, int, MPI_Comm, void *);
int MPI_Allreduce(const void *, void *, int, int, int, MPI_Comm);
int MPI_Barrier(MPI_Comm);
double MPI_Wtime(void);
#endif
