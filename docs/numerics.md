# Numerical formulation and validation

## Spatial discretisation

The unit square is partitioned into rectangles with spacings \(h_x=1/(n_x+1)\) and \(h_y=1/(n_y+1)\). Each rectangle is split consistently along the southwest-to-northeast diagonal into two right triangles. Continuous piecewise-linear (P1) shape functions are used, with the standard weak form:

\[
\int_{\Omega}\nabla u_h\cdot\nabla v_h\,dx=\int_{\Omega}f v_h\,dx
\quad \text{for all }v_h\in V_h^0.
\]

For this *specific* uniform right-triangle mesh, the element stiffness entries assemble to a five-point interior stencil. Horizontal neighbours have weight \(-h_y/h_x\), vertical neighbours \(-h_x/h_y\), and the central diagonal is \(2(h_y/h_x+h_x/h_y)\). The interior couplings along the triangle diagonal cancel because the opposite angles are right angles. Thus a matrix-free application of the stencil is algebraically equivalent to using the assembled P1 FEM stiffness matrix on this mesh. **This equivalence does not hold for an arbitrary triangulation or adapted grid.**

Instead of integrating the forcing exactly element by element, the implementation uses a *mass-lumped nodal right-hand side*, \(b_i=h_xh_y f(x_i,y_i)\), valid at interior nodes of the uniform mesh. It is a quadrature approximation (not exact P1 load integration).

All nodes on the boundary are prescribed \(u=0\), and only interior nodes are unknowns. Neighbour values beyond the global interior grid are identically zero.

## Manufactured solutions

With \(s_x=\sin(\pi x)\), \(s_y=\sin(\pi y)\), and \(v=1+0.3x+0.15y+0.2xy\), the default exact solution is \(u=s_xs_y v\). Let \(v_x=0.3+0.2y\), \(v_y=0.15+0.2x\). Then the PDE forcing is

\[
f=2\pi^2s_xs_yv-2\pi\bigl[\cos(\pi x)s_yv_x+s_x\cos(\pi y)v_y\bigr].
\]

A second test problem `--case sin` uses \(u=\sin(\pi x)\sin(\pi y)\) and \(f=2\pi^2u\). This is primarily a mathematical smoke test: it is also an eigenmode of the discrete stencil, so zero-initial-guess CG typically converges in one iteration; do not use it for performance benchmarking.

The discrete \(L^2\) error is computed by midpoint-weighted nodal values, \(\sqrt{h_xh_y\sum |u_{ij}-u_\mathrm{exact}(x_i,y_j)|^2}\). Also reported are relative nodal \(L^2\) and maximum error. Since the manufactured solution is smooth, approximately second-order convergence is expected in these grid-sampled errors.

## MPI decomposition

`MPI_Dims_create` produces a two-dimensional process grid unless a topology is specified. `MPI_Cart_create` and `MPI_Cart_shift` identify the immediate left/right/top/bottom neighbours. Interior grid points are divided into near-equal Cartesian blocks, including unequal splits when dimensions are not divisible by process counts. Each rank stores a `(local_ny+2) × (local_nx+2)` array with a one-cell halo.

Four `MPI_Sendrecv` calls exchange the two vertical and two horizontal faces. They use separate tags for each direction and send only face values, not corner values (the five-point stencil never accesses corners). At physical domain edges, halo values are zero. `MPI_Allreduce` computes global CG inner products and error norms; `MPI_MAX` reports the slowest rank's elapsed time.

The CG solver starts at zero and stops when its recursively updated residual satisfies \(\lVert r\rVert_2/\lVert b\rVert_2\leq\varepsilon\). To detect drift or accidental arithmetic errors, the program separately recomputes the **true residual** after convergence using one additional stencil application. It returns a nonzero exit code if the iteration limit is reached.

## Validation checklist

- Manufactured smooth solutions on square and rectangular meshes.
- L2 errors decrease by approximately four on doubling the grid resolution.
- `true_residual` agrees with the reported CG residual.
- 1, 2 and 4 MPI processes give matching solution errors to a specified tolerance.
- Cartesian shapes 2×1, 1×2, 2×2 and 4×1 produce matching results.
- Per-rank CSV outputs cover each global interior node exactly once.
- Strong scaling uses repeated **measured** MPI wall times on the same resolution, with per-run raw data retained.

## Limitations

The code is not an unstructured or adaptive-grid finite-element solver. There is no preconditioner (CG iterations increase with resolution), no nonzero Dirichlet support, and no overlap of halo exchange with computation. The two-dimensional structured topology connects only face neighbours. Real-MPI distributed correctness and performance need to be validated on a machine with an MPI runtime; the serial shim tests only the single-rank numerical path.
