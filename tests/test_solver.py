#!/usr/bin/env python3
"""Self-contained black-box tests for the MPI finite-element solver."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run_case(nproc, nx, ny, *, layout=None, problem="mixed", serial=False, output_prefix=None):
    binary = ROOT / "bin" / ("poisson_serial_smoke" if serial else "poisson_mpi")
    args = [str(binary), "--nx", str(nx), "--ny", str(ny),
            "--tol", "1e-10", "--case", problem]
    if layout:
        args += ["--px", str(layout[0]), "--py", str(layout[1])]
    if output_prefix:
        args += ["--output-prefix", str(output_prefix)]
    if not serial:
        launcher = shutil.which("mpiexec") or shutil.which("mpirun")
        if launcher is None:
            raise RuntimeError("MPI launcher missing: install OpenMPI or MPICH")
        args = [launcher, "-n", str(nproc)] + args
    env = os.environ.copy()
    env.update({"OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
                "OMPI_MCA_rmaps_base_oversubscribe": "1"})
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                            timeout=60, env=env)
    if result.returncode != 0:
        raise AssertionError(f"Failed ({result.returncode}): {' '.join(args)}\n"
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    results = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    if len(results) != 1:
        raise AssertionError(f"Expected one RESULT line.\n{result.stdout}\n{result.stderr}")
    parsed = {}
    for field in results[0].split()[1:]:
        key, value = field.split("=", 1)
        try:
            parsed[key] = int(value) if re.fullmatch(r"[+-]?\d+", value) else float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-only", action="store_true", help="only test single-process smoke binary")
    args = parser.parse_args()
    serial = args.serial_only
    print("Mode:", "single-rank shim (not MPI verification)" if serial else "real MPI")

    r15 = run_case(1, 15, 15, serial=serial)
    r31 = run_case(1, 31, 31, serial=serial)
    r63 = run_case(1, 63, 63, serial=serial)
    for value in (r15, r31, r63):
        check(value["converged"] == 1 and value["true_residual"] < 2e-9,
              f"CG converged with true relative residual < 2e-9 on {value['nx']}x{value['ny']}")
    ratio1 = r15["l2_error"] / r31["l2_error"]
    ratio2 = r31["l2_error"] / r63["l2_error"]
    check(3.5 < ratio1 < 4.5 and 3.5 < ratio2 < 4.5,
          f"second-order L2 error reduction under mesh refinement ({ratio1:.3f}, {ratio2:.3f})")
    rectangular = run_case(1, 31, 47, serial=serial)
    check(rectangular["converged"] == 1 and rectangular["l2_error"] < 0.001,
          "non-square mesh converged with small error")
    sin_case = run_case(1, 31, 31, serial=serial, problem="sin")
    check(sin_case["converged"] == 1 and sin_case["l2_error"] < 0.001,
          "sinusoidal manufactured solution passed")

    with tempfile.TemporaryDirectory(prefix="poisson_fem_") as dirname:
        prefix = Path(dirname) / "solution"
        small = run_case(1, 9, 7, serial=serial, output_prefix=prefix)
        files = sorted(Path(dirname).glob("solution_rank*.csv"))
        check(len(files) == 1 and len(files[0].read_text().splitlines()) == 9*7 + 1,
              "solution CSV contains each owned interior node exactly once")
        check(small["true_residual"] < 2e-9, "small-grid solution passes residual check")

    if not serial:
        baseline = run_case(1, 47, 39)
        for ranks, layout in [(2, (2, 1)), (2, (1, 2)), (4, (2, 2)), (4, (4, 1))]:
            result = run_case(ranks, 47, 39, layout=layout)
            check(result["ranks"] == ranks and result["px"] == layout[0]
                  and result["py"] == layout[1], f"MPI layout {layout} launched correctly")
            check(result["converged"] == 1 and result["true_residual"] < 2e-9,
                  f"MPI {layout} converged")
            check(abs(result["l2_error"] - baseline["l2_error"]) < 1e-8,
                  f"MPI {layout} matches single-rank numerical solution")
        with tempfile.TemporaryDirectory(prefix="poisson_fem_mpi_") as dirname:
            prefix = Path(dirname) / "solution"
            run_case(4, 11, 9, layout=(2, 2), output_prefix=prefix)
            files = sorted(Path(dirname).glob("solution_rank*.csv"))
            headers = [f.read_text().splitlines() for f in files]
            coords = [tuple(map(int, ln.split(",")[:2])) for rows in headers for ln in rows[1:]]
            check(len(files) == 4 and len(coords) == 11*9 and len(set(coords)) == 11*9,
                  "MPI writes each node exactly once across four rank outputs")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
