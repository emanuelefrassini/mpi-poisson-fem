#define _POSIX_C_SOURCE 200809L
#include "mpi_shim.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
int MPI_Init(int *a,char ***b){(void)a;(void)b;return 0;}
int MPI_Finalize(void){return 0;}
int MPI_Abort(MPI_Comm c,int e){(void)c;exit(e);}
int MPI_Comm_size(MPI_Comm c,int *n){(void)c;*n=1;return 0;}
int MPI_Comm_rank(MPI_Comm c,int *n){(void)c;*n=0;return 0;}
int MPI_Dims_create(int n,int d,int *dims){(void)n;(void)d;dims[0]=dims[1]=1;return 0;}
int MPI_Cart_create(MPI_Comm c,int n,const int *d,const int *p,int r,MPI_Comm *out){
 (void)c;(void)n;(void)d;(void)p;(void)r;*out=1;return 0;}
int MPI_Cart_coords(MPI_Comm c,int r,int d,int *coords){(void)c;(void)r;(void)d;coords[0]=coords[1]=0;return 0;}
int MPI_Cart_shift(MPI_Comm c,int d,int s,int *a,int *b){(void)c;(void)d;(void)s;*a=*b=MPI_PROC_NULL;return 0;}
int MPI_Comm_free(MPI_Comm *c){*c=MPI_COMM_NULL;return 0;}
int MPI_Sendrecv(const void *a,int b,int c,int d,int e,void *f,int g,int h,int i,int j,MPI_Comm k,void *l){
 (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;return 0;}
int MPI_Allreduce(const void *send,void *recv,int count,int datatype,int op,MPI_Comm comm){
 (void)op;(void)comm;if(datatype==MPI_DOUBLE)memcpy(recv,send,count*sizeof(double));return 0;}
int MPI_Barrier(MPI_Comm c){(void)c;return 0;}
double MPI_Wtime(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
