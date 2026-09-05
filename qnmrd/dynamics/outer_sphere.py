"""
qnmrd/dynamics/outer_sphere.py
==============================
Hwang-Freed Outer-Sphere Translational Diffusion Model.

This module computes the relaxation contribution from water molecules
freely diffusing around the paramagnetic complex, which is critical
for accurate ab initio prediction of total relaxivity.
"""

import numpy as np

# Physical Constants
MU_0 = 4.0 * np.pi * 1e-7      # Magnetic permeability of free space (T·m/A)
H_BAR = 1.054571817e-34        # Reduced Planck constant (J·s)
GAMMA_H = 2.6752218744e8       # Proton gyromagnetic ratio (rad/s/T)
GAMMA_E = 1.76085963023e11     # Electron gyromagnetic ratio (rad/s/T)
N_A = 6.02214076e23            # Avogadro's number (mol⁻¹)


def j_os(omega, tau_d):
    """
    Hwang-Freed spectral density function for translational diffusion.
    
    Parameters
    ----------
    omega : ndarray or float
        Angular frequency (rad/s).
    tau_d : float
        Translational correlation time, tau_d = d^2 / D_rel (s).
        
    Returns
    -------
    j : ndarray or float
        Real part of the spectral density at frequency omega.
    """
    # z = sqrt(i * omega * tau_d)
    # i^0.5 = (1 + i)/sqrt(2)
    # So sqrt(i * w * td) = sqrt(w * td / 2) * (1 + 1j)
    w_td = omega * tau_d
    # Handle omega=0 safely
    w_td = np.maximum(w_td, 1e-20)
    
    z = np.sqrt(w_td / 2.0) * (1.0 + 1j)
    
    # J(w) = Re { [1 + 5/8 z + 1/8 z^2] / [1 + z + 4/9 z^2 + 1/9 z^3] }
    num = 1.0 + (5.0 / 8.0) * z + (1.0 / 8.0) * (z ** 2)
    den = 1.0 + z + (4.0 / 9.0) * (z ** 2) + (1.0 / 9.0) * (z ** 3)
    
    return np.real(num / den)


def compute_R1_outer_sphere(B0_T, S, d_A=3.6, D_rel=2.2e-9, C_mM=1.0):
    """
    Computes the Outer-Sphere longitudinal relaxation rate R1_OS (s⁻¹).
    
    Parameters
    ----------
    B0_T : array-like or float
        Magnetic field in Tesla.
    S : float
        Electronic spin quantum number (e.g., 3.5 for GdIII).
    d_A : float
        Distance of closest approach in Angstroms (default 3.6).
    D_rel : float
        Relative translational diffusion coefficient in m²/s 
        (default 2.2e-9 for Gd-DOTA in water at 298K).
    C_mM : float
        Concentration of the paramagnetic complex in mM (default 1.0).
        (Note: relaxivity r1_OS is obtained when C_mM = 1.0).
        
    Returns
    -------
    R1_OS : ndarray or float
        The outer-sphere relaxation rate (s⁻¹). 
        If C_mM = 1, this is equal to the outer-sphere relaxivity r1_OS.
    """
    B0_T = np.asarray(B0_T)
    d_m = d_A * 1e-10
    
    # Frequencies
    w_I = GAMMA_H * B0_T
    w_S = GAMMA_E * B0_T
    
    tau_d = (d_m ** 2) / D_rel
    
    # Pre-factor for Outer Sphere
    # C_OS = (32 pi / 405) * (mu_0 / 4 pi)^2 * gamma_I^2 * gamma_S^2 * h_bar^2 * N_A / 1000 * C_mM * S(S+1) / (d_m * D_rel)
    # Note: 1 mM = 1 mol/m^3. So N_A * C_mM = number of complexes per m^3.
    N_density = N_A * C_mM  # complexes / m^3
    
    C_OS = (32.0 * np.pi / 405.0) * ((MU_0 / (4.0 * np.pi)) ** 2) \
           * (GAMMA_H ** 2) * (GAMMA_E ** 2) * (H_BAR ** 2) \
           * N_density * S * (S + 1.0) / (d_m * D_rel)
           
    J_I = j_os(w_I, tau_d)
    J_S = j_os(w_S, tau_d)
    
    R1_OS = C_OS * (3.0 * J_I + 7.0 * J_S)
    return R1_OS
