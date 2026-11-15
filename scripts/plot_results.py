#!/usr/bin/env python3
"""Visualise actual solver output and strong-scaling measurements."""
import argparse
import csv
from pathlib import Path
import statistics


def plot_solution(prefix, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    files = sorted(prefix.parent.glob(prefix.name + "_rank*.csv"))
    if not files:
        raise SystemExit(f"No solver CSVs matching {prefix}_rank*.csv")
    rows = []
    for file in files:
        with file.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    nx = max(int(r["ix"]) for r in rows)
    ny = max(int(r["iy"]) for r in rows)
    u = np.zeros((ny+2,nx+2))
    exact = np.zeros_like(u)
    seen = set()
    for row in rows:
        x,y=int(row["ix"]),int(row["iy"])
        if (x,y) in seen:
            raise SystemExit("Duplicate global node detected")
        seen.add((x,y))
        u[y,x]=float(row["u"])
        exact[y,x]=float(row["exact"])
    if len(seen)!=nx*ny:
        raise SystemExit(f"Missing nodes: got {len(seen)}, expected {nx*ny}")
    output_dir.mkdir(parents=True,exist_ok=True)
    for field,name,cmap in [(u,"solution.png","viridis"),
                            (np.abs(u-exact),"absolute_error.png","magma")]:
        plt.figure(figsize=(7,5))
        plt.imshow(field,origin="lower",extent=(0,1,0,1),cmap=cmap,aspect="auto")
        plt.xlabel("x"); plt.ylabel("y")
        plt.title("Numerical solution" if name=="solution.png" else "Absolute error")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_dir/name,dpi=170)
        plt.close()
        print("Saved", output_dir/name)


def plot_scaling(csv_path, output_dir):
    import matplotlib.pyplot as plt
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    ranks = sorted({int(row["ranks"]) for row in rows})
    times = {p:statistics.median(float(row["time_sec"]) for row in rows
                                 if int(row["ranks"])==p) for p in ranks}
    output_dir.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(ranks,[times[n] for n in ranks],"o-")
    plt.xlabel("MPI processes"); plt.ylabel("Median elapsed seconds")
    plt.title("Strong scaling: solve time")
    plt.xticks(ranks); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(output_dir/"solve_time.png",dpi=170);plt.close()
    print("Saved",output_dir/"solve_time.png")
    if 1 in times:
        plt.figure(figsize=(7,5))
        plt.plot(ranks,[times[1]/times[n] for n in ranks],"o-",label="Measured")
        plt.plot(ranks,ranks,"--",label="Ideal linear speedup")
        plt.xlabel("MPI processes"); plt.ylabel("Speedup vs 1 process")
        plt.title("Strong scaling: speedup")
        plt.legend();plt.xticks(ranks);plt.grid(alpha=0.2)
        plt.tight_layout();plt.savefig(output_dir/"speedup.png",dpi=170);plt.close()
        print("Saved",output_dir/"speedup.png")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--solution-prefix",type=Path)
    p.add_argument("--scaling-csv",type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("figures"))
    args=p.parse_args()
    if not args.solution_prefix and not args.scaling_csv:
        p.error("Provide --solution-prefix or --scaling-csv")
    if args.solution_prefix:
        plot_solution(args.solution_prefix,args.output_dir)
    if args.scaling_csv:
        plot_scaling(args.scaling_csv,args.output_dir)


if __name__ == "__main__":
    main()
