"""
sweep_plots.py
==============
Generate energy-level, transition-frequency, and transition-intensity
plots for a B0 field sweep.

All figures are saved to the ``results/figures/`` directory (created
automatically if it does not exist).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from qnmrd.spin.sweep import sweep_field

# Default output directory (relative to project root)
_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),          # qnmrd/visualization/
    "..", "..", "results", "figures"    # → results/figures/
)
FIGURES_DIR = os.path.normpath(_RESULTS_DIR)


def generate_sweep_plots(S, D_MHz, E_MHz, case_name,
                         figures_dir=None,
                         gamma_MHz_T=-28024.95):
    """
    Produce three plots for a given spin / ZFS case:

    1. Energy levels vs B0
    2. Transition frequencies vs B0
    3. Transition intensities vs B0

    Parameters
    ----------
    S : float
        Electronic spin quantum number.
    D_MHz : float
        Axial ZFS parameter in MHz.
    E_MHz : float
        Rhombic ZFS parameter in MHz.
    case_name : str
        Label used in titles and output filenames (e.g. 'Mn_II').
    figures_dir : str or None
        Directory for PNG output.  Defaults to ``results/figures/``.
    gamma_MHz_T : float
        Electron gyromagnetic ratio in MHz/T.
    """
    if figures_dir is None:
        figures_dir = FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)

    B0_values_T = np.logspace(-4, 1, 100)
    results = sweep_field(S, D_MHz, E_MHz, B0_values_T, gamma_MHz_T)

    B0 = [r['B0_T'] for r in results]
    evals = np.array([r['evals'] for r in results])

    # ---- 1. Energy levels ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(evals.shape[1]):
        ax.plot(B0, evals[:, i], label=f'Level {i}')
    ax.set_xscale('log')
    ax.set_xlabel('Magnetic Field $B_0$ (T)', fontsize=12)
    ax.set_ylabel('Energy (MHz)', fontsize=12)
    ax.set_title(f'Energy Levels — {case_name}  (S={S}, D={D_MHz} MHz, E={E_MHz} MHz)',
                 fontsize=11)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)
    out = os.path.join(figures_dir, f'energy_levels_{case_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---- 2. Transition frequencies ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        b0_val = r['B0_T']
        freqs = [t['freq_MHz'] for t in r['transitions'] if t['freq_MHz'] > 1e-3]
        if freqs:
            ax.scatter([b0_val] * len(freqs), freqs, s=2, color='steelblue', alpha=0.35)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Magnetic Field $B_0$ (T)', fontsize=12)
    ax.set_ylabel('Transition Frequency (MHz)', fontsize=12)
    ax.set_title(f'Transition Frequencies — {case_name}', fontsize=11)
    ax.grid(alpha=0.4)
    out = os.path.join(figures_dir, f'transition_freqs_{case_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---- 3. Transition intensities ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        b0_val = r['B0_T']
        ints = [t['int_Sx'] + t['int_Sy'] + t['int_Sz'] for t in r['transitions']]
        if ints:
            ax.scatter([b0_val] * len(ints), ints, s=2, color='tomato', alpha=0.35)
    ax.set_xscale('log')
    ax.set_xlabel('Magnetic Field $B_0$ (T)', fontsize=12)
    ax.set_ylabel('Total Intensity $|\\langle i|S_\\alpha|j\\rangle|^2$', fontsize=12)
    ax.set_title(f'Transition Intensities — {case_name}', fontsize=11)
    ax.grid(alpha=0.4)
    out = os.path.join(figures_dir, f'transition_intensities_{case_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

