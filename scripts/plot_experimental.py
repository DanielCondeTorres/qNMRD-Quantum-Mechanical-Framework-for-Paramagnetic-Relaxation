"""
plot_experimental.py
====================
Plots model predictions against an explicitly supplied experimental data set.

This module deliberately contains no embedded ``literature`` values.  A plot
cannot be called an experimental validation unless the numerical data, its
primary reference, and the data-processing procedure are supplied alongside
the calculation.  The expected CSV columns are::

    frequency_MHz,r1_mM_s,error_mM_s

Use only values digitised or obtained with permission from the cited primary
source, and archive the input CSV with the manuscript's Supporting
Information.
"""

import numpy as np
import os
import sys
import argparse

# Ensure the package is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnmrd.validation.sbm import nmrd_sbm

def load_experimental_data(csv_path):
    """Read a documented experimental NMRD data set from a CSV file."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    required = {"frequency_MHz", "r1_mM_s", "error_mM_s"}
    names = set(data.dtype.names or ())
    missing = required - names
    if missing:
        raise ValueError(
            f"{csv_path} is missing required CSV columns: {', '.join(sorted(missing))}"
        )
    return data


def generate_experimental_comparison(csv_path, reference, output_path=None):
    """Compare qNMRD with a traceable experimental data set."""
    import matplotlib.pyplot as plt
    from qnmrd.nmrd.profiles import compute_nmrd_profile

    print("Generating model/experiment comparison plot...")
    
    # 1. High-resolution sweep frequencies (0.01 to 100 MHz)
    B0_MHz_sweep = np.logspace(-2, 2, 50)
    
    # 2. Run the exact reference implementation.  This is the calculation
    # released and benchmarked by this repository; it is not hardware-VQE data.
    print("-> Running ZFS-Redfield reference sweep...")
    # Parameters for Gd(III)
    S = 3.5
    D_MHz = 600.0
    E_MHz = 90.0
    r_angstrom = 3.1
    tau_c_ps = 80.0
    
    B0_T_sweep = B0_MHz_sweep / 42.5774
    profile = compute_nmrd_profile(
        S=S, D_MHz=D_MHz, E_MHz=E_MHz,
        r_m=r_angstrom * 1e-10, tau_c_s=tau_c_ps * 1e-12,
        B0_values_T=B0_T_sweep, orientation_model="static_average",
        n_orientations=64, label="Gd_III_static_average"
    )
    from qnmrd.dynamics.outer_sphere import compute_R1_outer_sphere
    sbm_data = nmrd_sbm(B0_T_sweep, S, r_angstrom*1e-10, tau_c_ps*1e-12, proton_freq=False)
    r1_sbm_IS = sbm_data['r1']
    r1_sbm_OS = compute_R1_outer_sphere(B0_T_sweep, S, d_A=3.6, D_rel=2.2e-9, C_mM=1.0)
    r1_sbm = r1_sbm_IS + r1_sbm_OS
    r1_reference = profile['r1_redfield'] + r1_sbm_OS
    
    # 4. Load traceable experimental data.  Never substitute mock values here.
    experimental = load_experimental_data(csv_path)
    
    # 5. Plot
    plt.figure(figsize=(9, 6))
    
    # Plot experimental points
    plt.errorbar(experimental['frequency_MHz'], experimental['r1_mM_s'],
                 yerr=experimental['error_mM_s'], fmt='ko', markersize=8,
                 capsize=4, label=f'Experiment: {reference}', zorder=5)
    
    # Plot theoretical models
    plt.plot(B0_MHz_sweep, r1_reference, 'b-', linewidth=3,
             label='ZFS-Redfield (static orientation average)')
    plt.plot(B0_MHz_sweep, r1_sbm, 'r--', linewidth=2, label='Classical SBM (Zeeman only)')
    
    plt.xscale('log')
    plt.xlabel('Proton Larmor Frequency $\\nu_H$ (MHz)', fontsize=14)
    plt.ylabel('Relaxivity $r_1$ (mM$^{-1}$ s$^{-1}$)', fontsize=14)
    plt.title('qNMRD model comparison (Gd(III) complex, S=7/2)', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Save the figure
    if output_path is None:
        output_path = os.path.join("results", "figures", "model_experiment_comparison.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Validation plot saved to: {output_path}")

if __name__ == "__main__":
    # Suppress intermediate prints from quantum_pipeline
    import builtins
    original_print = builtins.print
    def dummy_print(*args, **kwargs): pass
    builtins.print = dummy_print
    
    parser = argparse.ArgumentParser(
        description="Compare qNMRD predictions with a traceable NMRD CSV data set."
    )
    parser.add_argument("data", help="CSV with frequency_MHz,r1_mM_s,error_mM_s columns")
    parser.add_argument("--reference", required=True,
                        help="Primary citation or DOI for the supplied data")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    try:
        generate_experimental_comparison(args.data, args.reference, args.output)
    finally:
        builtins.print = original_print
        print("Model/experiment comparison complete!")
