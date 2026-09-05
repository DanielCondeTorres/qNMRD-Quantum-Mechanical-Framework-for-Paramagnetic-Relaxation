"""
run_cases.py
============
Main driver script for the qNMRD spin-Hamiltonian benchmark cases.

Runs a B0 field sweep for a predefined set of (S, D, E) cases and saves:
  - Terminal output: eigenvalues and transition data
  - PNG figures: energy levels, transition frequencies, transition intensities
                 (all written to results/figures/)

Usage
-----
From the project root:

    python scripts/run_cases.py

Or run a single case by name:

    python scripts/run_cases.py --case Case_C
"""

import sys
import os
import argparse
import numpy as np

# Ensure the project root is on the Python path so `qnmrd` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnmrd.spin.hamiltonian import SpinHamiltonian
from qnmrd.spin.sweep import sweep_field
from qnmrd.visualization.sweep_plots import generate_sweep_plots

# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------
CASES = {
    # Case A: S=1/2, no ZFS (pure Zeeman — control case)
    "Case_A": {"S": 0.5,  "D": 0.0,    "E": 0.0},
    # Case B: S=5/2, no ZFS (Mn(II)-like Zeeman limit)
    "Case_B": {"S": 2.5,  "D": 0.0,    "E": 0.0},
    # Case C: S=5/2, axial ZFS only (Mn(II) with D)
    "Case_C": {"S": 2.5,  "D": 100.0,  "E": 0.0},
    # Case D: S=5/2, axial + rhombic ZFS (Mn(II) full)
    "Case_D": {"S": 2.5,  "D": 100.0,  "E": 20.0},
    # Case E: S=7/2, large ZFS (Gd(III)-like)
    "Case_E": {"S": 3.5,  "D": 3000.0, "E": 500.0},
}

GAMMA_E = -28024.95  # electron gyromagnetic ratio, MHz/T


def run_case(name: str, params: dict, verbose: bool = True):
    """
    Run eigenvalue analysis and generate figures for one benchmark case.

    Parameters
    ----------
    name : str
        Case label (e.g. 'Case_C').
    params : dict
        Keys: S (float), D (float, MHz), E (float, MHz).
    verbose : bool
        Print detailed transition information if True.
    """
    S, D, E = params["S"], params["D"], params["E"]
    ham = SpinHamiltonian(S)

    print(f"\n{'='*55}")
    print(f"  {name}  |  S={S}  D={D:.1f} MHz  E={E:.1f} MHz")
    print(f"{'='*55}")
    print(f"  Hilbert-space dimension : {ham.dim} × {ham.dim}")

    # --- ZFS validation ---
    H_zfs = ham.zfs_second_order(D_MHz=D, E_MHz=E)
    hermitian_ok = np.allclose(H_zfs, H_zfs.conj().T, atol=1e-10)
    trace_ok = np.isclose(np.trace(H_zfs).real, 0.0, atol=1e-10)
    print(f"  H_ZFS Hermitian         : {hermitian_ok}")
    print(f"  Tr(H_ZFS) = 0           : {trace_ok}")

    # --- Eigenvalues at key field values ---
    for label, B0 in [("B0 = 0 T (ZFS-dominated)", 0.0),
                       ("B0 = 0.1 T (intermediate)", 0.1),
                       ("B0 = 3.0 T (Zeeman-dominated)", 3.0)]:
        res = sweep_field(S, D, E, [B0], GAMMA_E)[0]
        print(f"\n  {label}")
        print(f"    Eigenvalues (MHz): {np.round(res['evals'], 3)}")

    # --- Top transitions at 0.1 T ---
    if verbose:
        res_int = sweep_field(S, D, E, [0.1], GAMMA_E)[0]
        major = sorted(
            [t for t in res_int["transitions"]
             if t["int_Sx"] + t["int_Sy"] + t["int_Sz"] > 0.1],
            key=lambda x: -(x["int_Sx"] + x["int_Sy"] + x["int_Sz"])
        )
        print(f"\n  Top transitions at B0=0.1 T (top 5 by intensity):")
        for t in major[:5]:
            total_int = t["int_Sx"] + t["int_Sy"] + t["int_Sz"]
            print(f"    |{t['i']}⟩ ↔ |{t['j']}⟩ : "
                  f"{t['freq_MHz']:.2f} MHz  "
                  f"(|Sx|²={t['int_Sx']:.3f}, |Sy|²={t['int_Sy']:.3f}, "
                  f"|Sz|²={t['int_Sz']:.3f}, total={total_int:.3f})")

    # --- Generate figures ---
    print(f"\n  Generating figures...")
    generate_sweep_plots(S, D, E, name)


def main():
    parser = argparse.ArgumentParser(
        description="qNMRD benchmark case runner"
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Run a specific case (e.g. --case Case_C). "
             "If omitted, all cases are run."
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation (faster, text output only)."
    )
    args = parser.parse_args()

    if args.case:
        if args.case not in CASES:
            print(f"ERROR: unknown case '{args.case}'. "
                  f"Valid options: {list(CASES.keys())}")
            sys.exit(1)
        cases_to_run = {args.case: CASES[args.case]}
    else:
        cases_to_run = CASES

    for name, params in cases_to_run.items():
        run_case(name, params, verbose=True)

    print("\n✅ All done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
