"""
run_nmrd.py
===========
Script to compute and plot NMRD profiles for Mn(II) and Gd(III)
using both the ZFS-resolved Redfield model and the SBM baseline.

Usage
-----
    python scripts/run_nmrd.py
    python scripts/run_nmrd.py --system Mn
    python scripts/run_nmrd.py --system Gd

Output (all in results/)
------------------------
  figures/nmrd_Mn_II.png
  figures/nmrd_Gd_III.png
  data/nmrd_Mn_II.json
  data/nmrd_Gd_III.json

Physical parameters
-------------------
ZFS values are taken from literature (not fitted to NMRD):
  Mn(II): D ≈ 800 MHz (≈ 0.027 cm⁻¹), E/D ≈ 0.1–0.3
           from CASSCF/NEVPT2 calculations, see:
           Platas-Iglesias et al. (2016) J. Phys. Chem. A 120, 6467.
  Gd(III): D ≈ 600 MHz (≈ 0.02 cm⁻¹), E/D ≈ 0.15
           from EPR + DFT, see:
           Rast et al. (2001) J. Chem. Phys. 115, 7554.
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnmrd.nmrd.profiles import compute_nmrd_profile, plot_nmrd_comparison

# ---------------------------------------------------------------------------
# Literature-sourced parameters (NOT fitted to NMRD)
# ---------------------------------------------------------------------------
SYSTEMS = {
    "Mn": {
        "label":    "Mn_II",
        "full":     "Mn(II)  [S = 5/2]",
        "S":        5 / 2,
        # D = 0.027 cm⁻¹ → MHz: 0.027 × 2π × 3e10 / 1e6 ≈ 5088 MHz
        # Using a smaller representative value consistent with
        # Platas-Iglesias et al. 2016 (aqua complex range 0.01–0.04 cm⁻¹)
        "D_MHz":    800.0,      # 0.0267 cm⁻¹ — axial ZFS
        "E_MHz":    200.0,      # rhombic ZFS (E/D ≈ 0.25)
        "r_m":      2.83e-10,   # Mn–H distance, Angstrom → metres
        "tau_c_s":  30e-12,     # rotational τ_c (small aqua complex), s
        "temp_K":   298.15,
    },
    "Gd": {
        "label":    "Gd_III",
        "full":     "Gd(III)  [S = 7/2]",
        "S":        7 / 2,
        "D_MHz":    600.0,      # ≈ 0.020 cm⁻¹, Rast et al. 2001
        "E_MHz":    90.0,       # E/D ≈ 0.15
        "r_m":      3.10e-10,   # Gd–H distance, m
        "tau_c_s":  110e-12,    # τ_c for small Gd chelate, s
        "temp_K":   298.15,
    },
}

# B0 sweep: ~0.01 MHz → ~500 MHz proton Larmor frequency
B0_VALUES_T = np.logspace(-4, 1, 80)


def run_system(name, params, verbose=True):
    print(f"\n{'='*60}")
    print(f"  {params['full']}")
    print(f"  D = {params['D_MHz']:.0f} MHz  |  E = {params['E_MHz']:.0f} MHz")
    print(f"  r = {params['r_m']*1e10:.2f} Å  |  τ_c = {params['tau_c_s']*1e12:.0f} ps")
    print(f"{'='*60}")

    profile = compute_nmrd_profile(
        S=params["S"],
        D_MHz=params["D_MHz"],
        E_MHz=params["E_MHz"],
        r_m=params["r_m"],
        tau_c_s=params["tau_c_s"],
        B0_values_T=B0_VALUES_T,
        temp_K=params["temp_K"],
        label=params["label"],
    )

    if verbose:
        # Print r1 at a few diagnostic frequencies
        nu = profile['nu_H_MHz']
        for target_MHz in [0.1, 1.0, 10.0, 60.0]:
            idx = np.argmin(np.abs(nu - target_MHz))
            rf  = profile['r1_redfield'][idx]
            sbm = profile['r1_sbm'][idx]
            print(f"  ν_H = {target_MHz:5.1f} MHz:  "
                  f"Redfield = {rf:.3f} mM⁻¹s⁻¹  |  "
                  f"SBM = {sbm:.3f} mM⁻¹s⁻¹  |  "
                  f"Δ = {abs(rf-sbm)/sbm*100:.1f}%")

    plot_nmrd_comparison(profile)
    return profile


def main():
    parser = argparse.ArgumentParser(description="qNMRD NMRD profile generator")
    parser.add_argument("--system", choices=["Mn", "Gd", "all"],
                        default="all", help="System to run.")
    args = parser.parse_args()

    systems = (["Mn", "Gd"] if args.system == "all"
               else [args.system])

    for name in systems:
        run_system(name, SYSTEMS[name])

    print("\n✅ Done. Results in results/figures/ and results/data/")


if __name__ == "__main__":
    main()
