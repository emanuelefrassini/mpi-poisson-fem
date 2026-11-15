#!/usr/bin/env python3
"""Run strong-scaling benchmarks with genuine MPI execution (no fabricated timings)."""
import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=255)
    p.add_argument("--ny", type=int, default=255)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "strong_scaling.csv")
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    binary = ROOT / "bin" / "poisson_mpi"
    if not launcher or not binary.exists():
        raise SystemExit("First install MPI and run `make`; MPI benchmarks require mpiexec and bin/poisson_mpi")
    if args.nx < 3 or args.ny < 3 or args.repeats < 1 or any(v < 1 for v in args.ranks):
        raise SystemExit("Invalid benchmark dimensions, rank count or repetitions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    env = os.environ.copy()
    env.update({"OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
                "OMPI_MCA_rmaps_base_oversubscribe": "1"})
    for nproc in args.ranks:
        for repeat in range(args.repeats):
            cmd = [launcher, "-n", str(nproc), str(binary), "--nx", str(args.nx),
                   "--ny", str(args.ny), "--tol", "1e-10", "--case", "mixed"]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    cwd=ROOT, env=env, timeout=args.timeout)
            if result.returncode:
                raise SystemExit(f"MPI benchmark failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
            lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
            if len(lines) != 1:
                raise SystemExit(f"No unique RESULT line:\n{result.stdout}")
            data = dict(token.split("=", 1) for token in lines[0].split()[1:])
            if data["converged"] != "1":
                raise SystemExit(f"Nonconverged run: {data}")
            row = {"repeat": repeat + 1, **data}
            rows.append(row)
            print(f"ranks={nproc} repeat={repeat+1} seconds={float(data['time_sec']):.6f}"
                  f" iterations={data['iterations']}")
    columns = list(rows[0].keys())
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", args.output)
    times = {p: statistics.median(float(row["time_sec"]) for row in rows if int(row["ranks"]) == p)
             for p in args.ranks}
    baseline = times.get(1)
    if baseline:
        for count in args.ranks:
            print(f"MPI ranks {count:2}: median={times[count]:.6f}s "
                  f"speedup={baseline/times[count]:.3f}x efficiency={baseline/(count*times[count]):.3f}")
    else:
        print("No 1-rank baseline: speedups not calculated")


if __name__ == "__main__":
    main()
