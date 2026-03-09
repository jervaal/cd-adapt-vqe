"""
examples/dissociation_curve.py
================================
Compute the potential energy dissociation curve for a molecule using CD-ADAPT-VQE.

This script parallelizes the computation across bond distances using Python's
multiprocessing module, reproducing the dissociation curve results from
arXiv:2601.05973.

Supported molecules
-------------------
- LiH  : Li-H bond scan from 0.8 to 1.8 Å  (active space: 4e, 5o)
- BeH2 : Be-H bond scan                     (active space: 4e, 5o)
- HF   : H-F bond scan                      (active space: 4e, 5o)

Usage
-----
    # Run LiH dissociation curve (l=1, t=0.25)
    python dissociation_curve.py --molecule LiH --l_order 1 --time_point 0.25

    # Run HF with higher-order AGP (l=2, t=0.75)
    python dissociation_curve.py --molecule HF --l_order 2 --time_point 0.75

Output
------
An Excel file: <MOLECULE>_CD_ADAPT_l<l>_t<t>.xlsx with columns:
    - Distance (Å)
    - Exact Energy (Ha)
    - CD-ADAPT Energy (Ha)
    - Absolute Error (Ha)
    - Number of Operators
    - Final Ansatz Operators
    - Computation Time (s)
"""

import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from time import time

# --- Add parent directory to path if running directly ---
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cd_adapt_vqe import CD_ADAPT_Solver

# ── Molecule configurations ──────────────────────────────────────────────────

MOLECULE_CONFIGS = {
    "LiH": {
        "template": "Li 0.0 0.0 0.0; H 0.0 0.0 {dist}",
        "active_space": {"num_particles": 4, "num_spatial_orbitals": 5},
        "distances": list(np.linspace(0.8, 1.8, 10)),
    },
    "BeH2": {
        # Linear symmetric molecule: H-Be-H with equal bond lengths
        "template": "Be 0.0 0.0 0.0; H 0.0 0.0 {dist}; H 0.0 0.0 -{dist}",
        "active_space": {"num_particles": 4, "num_spatial_orbitals": 5},
        "distances": list(np.linspace(0.8, 1.8, 10)),
    },
    "HF": {
        "template": "H 0.0 0.0 0.0; F 0.0 0.0 {dist}",
        "active_space": {"num_particles": 4, "num_spatial_orbitals": 5},
        "distances": list(np.linspace(0.6, 1.3, 10)),
    },
}


# ── Worker function (runs in each parallel process) ──────────────────────────

def compute_single_distance(args):
    """
    Run CD-ADAPT-VQE for a single bond distance.

    This function is designed to be called in parallel via multiprocessing.Pool.
    It instantiates a fresh CD_ADAPT_Solver, generates the AGP pool, runs the
    ADAPT loop, and returns a result dictionary.

    Parameters
    ----------
    args : tuple
        (dist, geometry_template, basis, active_space_config,
         l_order, T_duration, time_point, max_iter, grad_threshold)

    Returns
    -------
    dict
        Results for this distance point, with NaN values on failure.
    """
    (
        dist,
        geometry_template,
        basis,
        active_space_config,
        l_order,
        T_duration,
        time_point,
        max_iter,
        grad_threshold,
    ) = args

    process_name = mp.current_process().name
    geometry = geometry_template.format(dist=dist)
    print(f"[{process_name}] Starting dist = {dist:.3f} Å ...")

    try:
        # Build solver (molecule setup + Hamiltonian construction)
        solver = CD_ADAPT_Solver(
            geometry,
            basis=basis,
            active_space=active_space_config,
        )

        # Reference energy by exact diagonalization
        exact_energy = solver.exact_energy()
        num_qubits = solver.num_qubits

        # Generate AGP operator pool
        pool = solver.compute_agp_pool(
            l_order=l_order,
            T_duration=T_duration,
            time_points=[time_point],
        )

        # Run ADAPT loop
        t0 = time()
        result = solver.run_cd_adapt(
            pool,
            max_iterations=max_iter,
            gradient_threshold=grad_threshold,
            verbose=False,
        )
        elapsed = time() - t0

        final_energy = result["energies"][-1]
        error = abs(final_energy - exact_energy)

        print(
            f"[{process_name}] Done dist = {dist:.3f} Å | "
            f"E = {final_energy:.6f} Ha | err = {error:.2e} Ha"
        )

        return {
            "Distance (Å)": dist,
            "Qubits": num_qubits,
            "Pool Size": len(pool),
            "Exact Energy (Ha)": exact_energy,
            "CD-ADAPT Energy (Ha)": final_energy,
            "Absolute Error (Ha)": error,
            "Number of Operators": result["num_ops"],
            "Ansatz Operators": ", ".join(result["ansatz_ops"]),
            "Computation Time (s)": elapsed,
        }

    except Exception as e:
        print(f"[{process_name}] FAILED at dist = {dist:.3f} Å: {e}")
        return {
            "Distance (Å)": dist,
            "Qubits": np.nan,
            "Pool Size": np.nan,
            "Exact Energy (Ha)": np.nan,
            "CD-ADAPT Energy (Ha)": np.nan,
            "Absolute Error (Ha)": np.nan,
            "Number of Operators": 0,
            "Ansatz Operators": f"FAILED: {e}",
            "Computation Time (s)": np.nan,
        }


# ── Main parallel runner ──────────────────────────────────────────────────────

def run_dissociation_curve(
    molecule: str = "LiH",
    basis: str = "sto-3g",
    l_order: int = 1,
    T_duration: float = 1.0,
    time_point: float = 0.25,
    max_iter: int = 40,
    grad_threshold: float = 1e-2,
    n_cores: int = None,
) -> pd.DataFrame:
    """
    Compute the full dissociation curve using parallel CD-ADAPT-VQE calculations.

    Parameters
    ----------
    molecule : str
        Molecule name. One of: 'LiH', 'BeH2', 'HF'.
    basis : str
        Basis set. Default: 'sto-3g'.
    l_order : int
        AGP approximation order. Default: 1.
    T_duration : float
        Adiabatic evolution duration for schedule evaluation. Default: 1.0.
    time_point : float
        Time point for pool construction (in [0, T_duration]). Default: 0.25.
    max_iter : int
        Maximum ADAPT iterations per distance point. Default: 40.
    grad_threshold : float
        ADAPT convergence threshold. Default: 1e-2.
    n_cores : int or None
        Number of parallel processes. None uses 80% of available cores.

    Returns
    -------
    pd.DataFrame
        Results table with one row per bond distance.
    """
    if molecule not in MOLECULE_CONFIGS:
        raise ValueError(f"Unknown molecule '{molecule}'. Choose from: {list(MOLECULE_CONFIGS)}")

    config = MOLECULE_CONFIGS[molecule]
    distances = config["distances"]

    if n_cores is None:
        n_cores = max(1, int(mp.cpu_count() * 0.8))

    print(f"\n{'='*60}")
    print(f"CD-ADAPT-VQE Dissociation Curve: {molecule}")
    print(f"Basis: {basis} | l={l_order} | t={time_point} | T={T_duration}")
    print(f"Distances: {len(distances)} points | Parallel cores: {n_cores}")
    print(f"{'='*60}\n")

    # Build argument list for starmap
    args_list = [
        (
            dist,
            config["template"],
            basis,
            config["active_space"],
            l_order,
            T_duration,
            time_point,
            max_iter,
            grad_threshold,
        )
        for dist in distances
    ]

    t_start = time()
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(compute_single_distance, args_list)
    t_total = time() - t_start

    print(f"\nAll calculations completed in {t_total/60:.1f} minutes.")

    df = pd.DataFrame(results).sort_values("Distance (Å)").reset_index(drop=True)
    return df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CD-ADAPT-VQE dissociation curve (parallelized)."
    )
    parser.add_argument("--molecule", default="LiH", choices=["LiH", "BeH2", "HF"])
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--l_order", type=int, default=1)
    parser.add_argument("--time_point", type=float, default=0.25)
    parser.add_argument("--T_duration", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=40)
    parser.add_argument("--grad_threshold", type=float, default=1e-2)
    parser.add_argument("--n_cores", type=int, default=None)
    args = parser.parse_args()

    df = run_dissociation_curve(
        molecule=args.molecule,
        basis=args.basis,
        l_order=args.l_order,
        T_duration=args.T_duration,
        time_point=args.time_point,
        max_iter=args.max_iter,
        grad_threshold=args.grad_threshold,
        n_cores=args.n_cores,
    )

    # Save results
    output_file = (
        f"{args.molecule}_CD_ADAPT_l{args.l_order}"
        f"_t{str(args.time_point).replace('.', '')}.xlsx"
    )
    df.to_excel(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(df.to_string(index=False))
