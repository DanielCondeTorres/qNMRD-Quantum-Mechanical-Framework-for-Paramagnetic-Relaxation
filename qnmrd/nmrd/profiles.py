"""
profiles.py
===========
NMRD profile generator: ZFS-resolved Redfield vs SBM comparison.

Produces:
  - R1(B0) and r1(νH) arrays for both models
  - Figures saved to results/figures/
  - Data saved to results/data/ as JSON
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qnmrd.spin.hamiltonian import SpinHamiltonian
from qnmrd.dynamics.redfield import RedfieldR1
from qnmrd.dynamics.dipolar import GAMMA_H_RAD_S_T
from qnmrd.validation.sbm import nmrd_sbm, C_WATER_mM
from qnmrd.validation.orientation import static_orientation_average

# Default output directories
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FIGURES_DIR = os.path.normpath(os.path.join(_ROOT, "results", "figures"))
DATA_DIR    = os.path.normpath(os.path.join(_ROOT, "results", "data"))


def compute_nmrd_profile(S, D_MHz, E_MHz, r_m, tau_c_s,
                         B0_values_T, temp_K=298.15,
                         gamma_e_MHz_T=-28024.95,
                         orientation_model="aligned", n_orientations=64,
                         label="system"):
    """
    Compute both ZFS-resolved Redfield and SBM NMRD profiles.

    Parameters
    ----------
    S : float
        Electronic spin quantum number.
    D_MHz, E_MHz : float
        ZFS parameters in MHz.
    r_m : float
        Electron–proton distance in metres.
    tau_c_s : float
        Rotational correlation time in seconds.
    B0_values_T : array-like
        Magnetic field sweep in Tesla.
    temp_K : float
        Temperature in Kelvin.
    orientation_model : {"aligned", "static_average"}
        ``aligned`` reproduces the original calculation, with the ZFS
        principal axes aligned to the laboratory axes.  ``static_average``
        averages a deterministic SO(3) grid of static ZFS orientations.  The
        latter is a powder/reference average, not a dynamic SLE model.
    n_orientations : int
        Number of orientations when ``orientation_model="static_average"``.
    label : str
        System name for output filenames and titles.

    Returns
    -------
    profile : dict with keys:
        'B0_T', 'nu_H_MHz',
        'R1_redfield', 'r1_redfield',
        'R1_sbm',      'r1_sbm'
    """
    B0 = np.asarray(B0_values_T, dtype=float)
    nu_H = GAMMA_H_RAD_S_T * B0 / (2.0 * np.pi * 1e6)  # proton freq, MHz

    # --- Redfield (ZFS-resolved) ---
    ham = SpinHamiltonian(S)
    redfield = RedfieldR1(ham, r_m, tau_c_s, temp_K)
    if orientation_model == "aligned":
        R1_rf = redfield.sweep(B0, D_MHz, E_MHz, gamma_e_MHz_T)
    elif orientation_model == "static_average":
        R1_rf = np.array([
            static_orientation_average(
                redfield, field, D_MHz, E_MHz, gamma_e_MHz_T,
                n_orientations=n_orientations,
            )
            for field in B0
        ])
    else:
        raise ValueError(
            "orientation_model must be 'aligned' or 'static_average'."
        )
    r1_rf = R1_rf / C_WATER_mM

    # --- SBM baseline ---
    sbm = nmrd_sbm(B0, S, r_m, tau_c_s, proton_freq=False)
    R1_sbm = sbm['R1']
    r1_sbm = sbm['r1']

    return {
        'label':       label,
        'orientation_model': orientation_model,
        'n_orientations': n_orientations if orientation_model == 'static_average' else None,
        'B0_T':        B0,
        'nu_H_MHz':    nu_H,
        'R1_redfield': R1_rf,
        'r1_redfield': r1_rf,
        'R1_sbm':      R1_sbm,
        'r1_sbm':      r1_sbm,
    }


def plot_nmrd_comparison(profile, figures_dir=None, save_data=True,
                         data_dir=None, show=False):
    """
    Plot and save the NMRD comparison figure (Redfield vs SBM).

    Parameters
    ----------
    profile : dict
        Output of ``compute_nmrd_profile``.
    figures_dir : str or None
        Output directory for PNG. Defaults to results/figures/.
    save_data : bool
        If True, also save a JSON data file to results/data/.
    show : bool
        If True, call plt.show().
    """
    if figures_dir is None:
        figures_dir = FIGURES_DIR
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    label = profile['label']
    nu = profile['nu_H_MHz']
    r1_rf  = profile['r1_redfield']
    r1_sbm = profile['r1_sbm']

    fig, ax = plt.subplots(figsize=(9, 6))

    model_label = ('ZFS-Redfield (static orientation average)'
                   if profile.get('orientation_model') == 'static_average'
                   else 'ZFS-Redfield (aligned ZFS axes)')
    ax.semilogx(nu, r1_rf,  lw=2.5, color='#2563EB', label=model_label)
    ax.semilogx(nu, r1_sbm, lw=2.0, color='#DC2626', ls='--', label='SBM baseline')

    ax.set_xlabel(r'Proton Larmor frequency $\nu_H$ (MHz)', fontsize=13)
    ax.set_ylabel(r'Relaxivity $r_1$ (mM$^{-1}$ s$^{-1}$)', fontsize=13)
    ax.set_title(f'NMRD Profile — {label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    out_fig = os.path.join(figures_dir, f'nmrd_{label}.png')
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"  Figure saved: {out_fig}")
    if show:
        plt.show()
    plt.close(fig)

    # --- Save data ---
    if save_data:
        data = {
            'label':       label,
            'orientation_model': profile.get('orientation_model', 'aligned'),
            'n_orientations': profile.get('n_orientations'),
            'nu_H_MHz':    nu.tolist(),
            'r1_redfield': r1_rf.tolist(),
            'r1_sbm':      r1_sbm.tolist(),
            'B0_T':        profile['B0_T'].tolist(),
        }
        out_data = os.path.join(data_dir, f'nmrd_{label}.json')
        with open(out_data, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Data saved:   {out_data}")

    return fig
