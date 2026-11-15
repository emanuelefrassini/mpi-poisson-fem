/*
 * Parallel Poisson solver on a structured triangulation of the unit square.
 * Continuous piecewise-linear (P1) finite elements, nodal mass-lumped load,
 * and matrix-free distributed Conjugate Gradient with MPI halo exchange.
 *
 * For right-triangle splitting of uniform rectangular cells, the assembled
 * P1 stiffness matrix has the same five-point interior stencil as the
 * finite-difference Laplacian after multiplying by cell area. This program
 * exploits that equivalence instead of storing the sparse FEM matrix.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#ifdef SERIAL_MPI_SHIM
#include "../tests/mpi_shim.h"
#else
#include <mpi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IDX(g, i, j) ((size_t)(j) * (size_t)((g)->lx + 2) + (size_t)(i))

typedef struct {
    int nx, ny, max_iter, px, py, case_mixed;
    double tol;
    const char *output_prefix;
} Options;

typedef struct {
    MPI_Comm comm;
    int rank, nprocs, coords[2], dims[2];
    int left, right, bottom, top;
    int lx, ly, off_x, off_y;
    double hx, hy, cx, cy, diag;
    double *send_left, *send_right, *recv_left, *recv_right;
    double *send_bottom, *send_top, *recv_bottom, *recv_top;
} Grid;

static void die(MPI_Comm comm, int rank, const char *message) {
    if (rank == 0) fprintf(stderr, "ERROR: %s\n", message);
    MPI_Abort(comm, 1);
    exit(1);
}

static void *checked_calloc(size_t count, size_t size, MPI_Comm comm, int rank) {
    void *ptr = calloc(count, size);
    if (!ptr) die(comm, rank, "Out of memory");
    return ptr;
}

static long parse_long(const char *text, const char *key, MPI_Comm comm, int rank) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || value <= 0 || value > 10000000L) {
        if (rank == 0) fprintf(stderr, "Invalid value for %s: %s\n", key, text);
        MPI_Abort(comm, 2);
        exit(2);
    }
    return value;
}

static void usage(void) {
    puts("MPI P1 FEM Poisson solver on [0,1]^2 with zero Dirichlet boundary\n"
         "Usage: poisson_mpi [--nx N] [--ny N] [--tol TOL] [--max-iter N]\n"
         "                   [--px P --py P] [--case mixed|sin]\n"
         "                   [--output-prefix PATH]\n"
         "  --nx, --ny: number of INTERIOR grid points per direction (default 63)\n"
         "  --case: smooth manufactured solution (mixed, default) or sin(pi*x)sin(pi*y)\n"
         "  --output-prefix: write rank-specific CSVs for visualisation\n"
         "  Output: one RESULT line with timings, residual and solution errors");
}

static Options parse_args(int argc, char **argv, int rank) {
    Options opt = {63, 63, 20000, 0, 0, 1, 1e-10, NULL};
    for (int k = 1; k < argc; ++k) {
        if (!strcmp(argv[k], "--help")) {
            if (rank == 0) usage();
            MPI_Finalize();
            exit(0);
        }
        if (k + 1 >= argc) die(MPI_COMM_WORLD, rank, "Missing command-line value");
        const char *flag = argv[k], *arg = argv[++k];
        if (!strcmp(flag, "--nx")) opt.nx = (int)parse_long(arg, flag, MPI_COMM_WORLD, rank);
        else if (!strcmp(flag, "--ny")) opt.ny = (int)parse_long(arg, flag, MPI_COMM_WORLD, rank);
        else if (!strcmp(flag, "--max-iter")) opt.max_iter = (int)parse_long(arg, flag, MPI_COMM_WORLD, rank);
        else if (!strcmp(flag, "--px")) opt.px = (int)parse_long(arg, flag, MPI_COMM_WORLD, rank);
        else if (!strcmp(flag, "--py")) opt.py = (int)parse_long(arg, flag, MPI_COMM_WORLD, rank);
        else if (!strcmp(flag, "--tol")) {
            char *end = NULL;
            errno = 0;
            opt.tol = strtod(arg, &end);
            if (errno || end == arg || *end || !isfinite(opt.tol) || opt.tol <= 0.0 || opt.tol >= 1.0)
                die(MPI_COMM_WORLD, rank, "--tol must be strictly between 0 and 1");
        } else if (!strcmp(flag, "--case")) {
            if (!strcmp(arg, "sin")) opt.case_mixed = 0;
            else if (!strcmp(arg, "mixed")) opt.case_mixed = 1;
            else die(MPI_COMM_WORLD, rank, "--case must be mixed or sin");
        } else if (!strcmp(flag, "--output-prefix")) opt.output_prefix = arg;
        else die(MPI_COMM_WORLD, rank, "Unknown argument (try --help)");
    }
    if ((opt.px == 0) != (opt.py == 0)) die(MPI_COMM_WORLD, rank, "Provide both --px and --py");
    return opt;
}

static void partition(int global_n, int parts, int coord, int *offset, int *length) {
    const int base = global_n / parts, remainder = global_n % parts;
    *length = base + (coord < remainder ? 1 : 0);
    *offset = base * coord + (coord < remainder ? coord : remainder);
}

static void setup_grid(Grid *g, Options opt) {
    int dims[2] = {opt.px, opt.py}, periodic[2] = {0, 0};
    MPI_Comm_size(MPI_COMM_WORLD, &g->nprocs);
    if (opt.px == 0) MPI_Dims_create(g->nprocs, 2, dims);
    if ((long long)dims[0] * dims[1] != g->nprocs)
        die(MPI_COMM_WORLD, 0, "px*py must equal MPI process count");
    if (dims[0] > opt.nx || dims[1] > opt.ny)
        die(MPI_COMM_WORLD, 0, "More process-grid partitions than interior grid points");
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &g->comm);
    if (g->comm == MPI_COMM_NULL) die(MPI_COMM_WORLD, 0, "MPI_Cart_create failed");
    MPI_Comm_rank(g->comm, &g->rank);
    memcpy(g->dims, dims, sizeof(dims));
    MPI_Cart_coords(g->comm, g->rank, 2, g->coords);
    MPI_Cart_shift(g->comm, 0, 1, &g->left, &g->right);
    MPI_Cart_shift(g->comm, 1, 1, &g->bottom, &g->top);
    partition(opt.nx, dims[0], g->coords[0], &g->off_x, &g->lx);
    partition(opt.ny, dims[1], g->coords[1], &g->off_y, &g->ly);
    g->hx = 1.0 / (opt.nx + 1.0);
    g->hy = 1.0 / (opt.ny + 1.0);
    g->cx = g->hy / g->hx;
    g->cy = g->hx / g->hy;
    g->diag = 2.0 * (g->cx + g->cy);
    const int xx = g->lx, yy = g->ly;
    g->send_left   = checked_calloc(yy, sizeof(double), g->comm, g->rank);
    g->send_right  = checked_calloc(yy, sizeof(double), g->comm, g->rank);
    g->recv_left   = checked_calloc(yy, sizeof(double), g->comm, g->rank);
    g->recv_right  = checked_calloc(yy, sizeof(double), g->comm, g->rank);
    g->send_bottom = checked_calloc(xx, sizeof(double), g->comm, g->rank);
    g->send_top    = checked_calloc(xx, sizeof(double), g->comm, g->rank);
    g->recv_bottom = checked_calloc(xx, sizeof(double), g->comm, g->rank);
    g->recv_top    = checked_calloc(xx, sizeof(double), g->comm, g->rank);
}

static void free_grid(Grid *g) {
    free(g->send_left); free(g->send_right); free(g->recv_left); free(g->recv_right);
    free(g->send_bottom); free(g->send_top); free(g->recv_bottom); free(g->recv_top);
    MPI_Comm_free(&g->comm);
}

static void exchange_halos(Grid *g, double *a) {
    for (int j = 1; j <= g->ly; ++j) {
        g->send_left[j-1] = a[IDX(g,1,j)];
        g->send_right[j-1] = a[IDX(g,g->lx,j)];
    }
    memset(g->recv_left, 0, g->ly * sizeof(double));
    memset(g->recv_right, 0, g->ly * sizeof(double));
    MPI_Sendrecv(g->send_right, g->ly, MPI_DOUBLE, g->right, 101,
                 g->recv_left, g->ly, MPI_DOUBLE, g->left, 101,
                 g->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(g->send_left, g->ly, MPI_DOUBLE, g->left, 102,
                 g->recv_right, g->ly, MPI_DOUBLE, g->right, 102,
                 g->comm, MPI_STATUS_IGNORE);
    for (int j = 1; j <= g->ly; ++j) {
        a[IDX(g,0,j)] = g->recv_left[j-1];
        a[IDX(g,g->lx+1,j)] = g->recv_right[j-1];
    }
    for (int i = 1; i <= g->lx; ++i) {
        g->send_bottom[i-1] = a[IDX(g,i,1)];
        g->send_top[i-1] = a[IDX(g,i,g->ly)];
    }
    memset(g->recv_bottom, 0, g->lx * sizeof(double));
    memset(g->recv_top, 0, g->lx * sizeof(double));
    MPI_Sendrecv(g->send_top, g->lx, MPI_DOUBLE, g->top, 103,
                 g->recv_bottom, g->lx, MPI_DOUBLE, g->bottom, 103,
                 g->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(g->send_bottom, g->lx, MPI_DOUBLE, g->bottom, 104,
                 g->recv_top, g->lx, MPI_DOUBLE, g->top, 104,
                 g->comm, MPI_STATUS_IGNORE);
    for (int i = 1; i <= g->lx; ++i) {
        a[IDX(g,i,0)] = g->recv_bottom[i-1];
        a[IDX(g,i,g->ly+1)] = g->recv_top[i-1];
    }
}

static void matvec(Grid *g, double *x, double *out) {
    exchange_halos(g, x);
    for (int j = 1; j <= g->ly; ++j)
        for (int i = 1; i <= g->lx; ++i)
            out[IDX(g,i,j)] = g->diag * x[IDX(g,i,j)]
                - g->cx * (x[IDX(g,i-1,j)] + x[IDX(g,i+1,j)])
                - g->cy * (x[IDX(g,i,j-1)] + x[IDX(g,i,j+1)]);
}

static double global_dot(const Grid *g, const double *a, const double *b) {
    double local = 0.0, total = 0.0;
    for (int j = 1; j <= g->ly; ++j)
        for (int i = 1; i <= g->lx; ++i)
            local += a[IDX(g,i,j)] * b[IDX(g,i,j)];
    MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, g->comm);
    return total;
}

static double exact_value(double x, double y, int mixed) {
    const double pi = acos(-1.0);
    double base = sin(pi*x) * sin(pi*y);
    double factor = mixed ? (1.0 + 0.3*x + 0.15*y + 0.2*x*y) : 1.0;
    return base * factor;
}

static double forcing(double x, double y, int mixed) {
    const double pi = acos(-1.0), sx = sin(pi*x), sy = sin(pi*y);
    if (!mixed) return 2.0 * pi * pi * sx * sy;
    const double v = 1.0 + 0.3*x + 0.15*y + 0.2*x*y;
    const double vx = 0.3 + 0.2*y, vy = 0.15 + 0.2*x;
    return 2.0*pi*pi*sx*sy*v - 2.0*pi*(cos(pi*x)*sy*vx + sx*cos(pi*y)*vy);
}

static void fill_load(const Grid *g, double *b, int mixed) {
    for (int j = 1; j <= g->ly; ++j)
        for (int i = 1; i <= g->lx; ++i) {
            double x = (g->off_x + i) * g->hx;
            double y = (g->off_y + j) * g->hy;
            b[IDX(g,i,j)] = (g->hx * g->hy) * forcing(x, y, mixed);
        }
}

typedef struct { int iterations, converged; double relative_residual, seconds; } SolverStats;

static SolverStats conjugate_gradient(Grid *g, double *solution, const double *b,
                                      int max_iter, double tol) {
    size_t count = (size_t)(g->lx + 2) * (size_t)(g->ly + 2);
    double *r = checked_calloc(count, sizeof(double), g->comm, g->rank);
    double *p = checked_calloc(count, sizeof(double), g->comm, g->rank);
    double *q = checked_calloc(count, sizeof(double), g->comm, g->rank);
    for (int j=1;j<=g->ly;++j)
        for(int i=1;i<=g->lx;++i) r[IDX(g,i,j)] = p[IDX(g,i,j)] = b[IDX(g,i,j)];
    double rr = global_dot(g,r,r);
    double norm_b = sqrt(rr);
    SolverStats stats = {0, (norm_b == 0.0), norm_b == 0.0 ? 0.0 : 1.0, 0.0};
    MPI_Barrier(g->comm);
    double begin = MPI_Wtime();
    for (int k=0;k<max_iter && !stats.converged;++k) {
        matvec(g,p,q);
        double pap = global_dot(g,p,q);
        if (!isfinite(pap) || pap <= 0.0) die(g->comm,g->rank,"CG breakdown: non-positive p^T A p");
        const double alpha = rr / pap;
        for (int j=1;j<=g->ly;++j) for (int i=1;i<=g->lx;++i) {
            size_t idx = IDX(g,i,j);
            solution[idx] += alpha*p[idx];
            r[idx] -= alpha*q[idx];
        }
        double rr_next = global_dot(g,r,r);
        stats.iterations = k+1;
        stats.relative_residual = sqrt(rr_next)/norm_b;
        if (stats.relative_residual <= tol) { stats.converged=1; break; }
        const double beta = rr_next/rr;
        for (int j=1;j<=g->ly;++j) for (int i=1;i<=g->lx;++i) {
            size_t idx = IDX(g,i,j);
            p[idx] = r[idx] + beta*p[idx];
        }
        rr=rr_next;
    }
    double elapsed = MPI_Wtime() - begin;
    MPI_Allreduce(&elapsed,&stats.seconds,1,MPI_DOUBLE,MPI_MAX,g->comm);
    free(r); free(p); free(q);
    return stats;
}

static void output_rank_csv(const Grid *g, const double *u, const Options *opt) {
    if (!opt->output_prefix) return;
    char path[1024];
    int n = snprintf(path,sizeof(path),"%s_rank%04d.csv",opt->output_prefix,g->rank);
    if(n<0 || n>=(int)sizeof(path)) die(g->comm,g->rank,"Output prefix too long");
    FILE *f=fopen(path,"w");
    if(!f) die(g->comm,g->rank,"Unable to open output file");
    fprintf(f,"ix,iy,x,y,u,exact\n");
    for (int j=1;j<=g->ly;++j) for (int i=1;i<=g->lx;++i) {
        int ix=g->off_x+i, iy=g->off_y+j;
        double x=ix*g->hx,y=iy*g->hy;
        fprintf(f,"%d,%d,%.16g,%.16g,%.16g,%.16g\n",ix,iy,x,y,u[IDX(g,i,j)],exact_value(x,y,opt->case_mixed));
    }
    fclose(f);
}

int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    Options opt=parse_args(argc,argv,world_rank);
    Grid grid = {0};
    setup_grid(&grid,opt);
    size_t count = (size_t)(grid.lx+2)*(size_t)(grid.ly+2);
    double *u=checked_calloc(count,sizeof(double),grid.comm,grid.rank);
    double *b=checked_calloc(count,sizeof(double),grid.comm,grid.rank);
    double *au=checked_calloc(count,sizeof(double),grid.comm,grid.rank);
    fill_load(&grid,b,opt.case_mixed);
    SolverStats stats=conjugate_gradient(&grid,u,b,opt.max_iter,opt.tol);
    matvec(&grid,u,au);
    double local_l2=0.0,local_exact2=0.0,local_max=0.0;
    double local_res2=0.0,local_b2=0.0;
    for(int j=1;j<=grid.ly;++j) for(int i=1;i<=grid.lx;++i) {
        double x=(grid.off_x+i)*grid.hx,y=(grid.off_y+j)*grid.hy;
        size_t idx=IDX(&grid,i,j);
        double e=u[idx]-exact_value(x,y,opt.case_mixed);
        double res=b[idx]-au[idx];
        local_l2+=e*e; local_exact2+=pow(exact_value(x,y,opt.case_mixed),2);
        if (fabs(e)>local_max) local_max=fabs(e);
        local_res2+=res*res;local_b2+=b[idx]*b[idx];
    }
    double l2=0.0,ex2=0.0,emax=0.0,res2=0.0,b2=0.0;
    MPI_Allreduce(&local_l2,&l2,1,MPI_DOUBLE,MPI_SUM,grid.comm);
    MPI_Allreduce(&local_exact2,&ex2,1,MPI_DOUBLE,MPI_SUM,grid.comm);
    MPI_Allreduce(&local_max,&emax,1,MPI_DOUBLE,MPI_MAX,grid.comm);
    MPI_Allreduce(&local_res2,&res2,1,MPI_DOUBLE,MPI_SUM,grid.comm);
    MPI_Allreduce(&local_b2,&b2,1,MPI_DOUBLE,MPI_SUM,grid.comm);
    if(grid.rank==0) {
        printf("RESULT nx=%d ny=%d ranks=%d px=%d py=%d case=%s iterations=%d converged=%d "
               "relative_residual=%.9e true_residual=%.9e l2_error=%.9e rel_l2_error=%.9e "
               "linf_error=%.9e time_sec=%.9e\n",
               opt.nx,opt.ny,grid.nprocs,grid.dims[0],grid.dims[1],
               opt.case_mixed?"mixed":"sin",stats.iterations,stats.converged,
               stats.relative_residual,sqrt(res2/b2),sqrt(grid.hx*grid.hy*l2),
               sqrt(l2/ex2),emax,stats.seconds);
        fflush(stdout);
    }
    output_rank_csv(&grid,u,&opt);
    free(u);free(b);free(au);
    free_grid(&grid);
    MPI_Finalize();
    return stats.converged ? 0 : 3;
}
