"""
sbm.py
======
Solomon–Bloembergen–Morgan (SBM) baseline implementation.

This module provides a clean, independently implemented SBM calculation
for benchmarking against the ZFS-resolved Redfield model.

The SBM formula for paramagnetic longitudinal relaxation is:

    1/T1_SBM = (2/15) b_IS² S(S+1) [3 J(ω_I) + 7 J(ω_S)]

with the spectral density:

    J(ω) = τ_c / (1 + ω² τ_c²)

Note: this module uses the spectral density convention WITHOUT the factor
of 2 (i.e., one-sided), consistent with the Solomon (1955) paper.
The Redfield module uses J(ω) = 2τ/(1 + ω²τ²).  The factor is absorbed
into the 2/15 prefactor here vs 1/15 in Redfield implementations using
the two-sided convention.

Units
-----
All inputs and outputs are in SI units:
  - B0  : Tesla
  - r   : metres
  - τ_c : seconds
  - R1  : s⁻¹
  - r1  : mM⁻¹ s⁻¹  (relaxivity, concentration in mM)

References
----------
Solomon, I. (1955) Phys. Rev. 99, 559.
Bloembergen, N.; Morgan, L.O. (1961) J. Chem. Phys. 34, 842.
Merbach, A.E.; Helm, L.; Tóth, É. (Eds.) (2013) The Chemistry of
    Contrast Agents in Medical MRI. Wiley.
"""

import numpy as np
from ..dynamics.dipolar import (dipolar_prefactor, GAMMA_H_RAD_S_T,
                                 GAMMA_E_RAD_S_T)

# Water concentration (pure water, 298 K)  [mol/L = M]
C_WATER_M = 55.5   # M
C_WATER_mM = C_WATER_M * 1e3  # 55500 mM


def J_solomon(omega_rad_s, tau_c_s):
    """
    Spectral density in the Solomon convention (one-sided):

        J(ω) = τ_c / (1 + ω² τ_c²)

    Parameters
    ----------
    omega_rad_s : float or array-like
        Angular frequency in rad/s.
    tau_c_s : float
        Rotational correlation time in seconds.

    Returns
    -------
    float or ndarray  [seconds]
    """
    w = np.asarray(omega_rad_s, dtype=float)
    return tau_c_s / (1.0 + (w * tau_c_s) ** 2)


def R1_sbm(B0_T, S, r_m, tau_c_s,
           gamma_I=GAMMA_H_RAD_S_T,
           gamma_S=GAMMA_E_RAD_S_T):
    """
    Solomon–Bloembergen–Morgan nuclear longitudinal relaxation rate (s⁻¹).

        R1_SBM = (2/15) b_IS² S(S+1) [3 J(ω_I) + 7 J(ω_S)]

    Valid in the Zeeman-dominated limit where ZFS ≪ ℏ|γ_e|B0.

    Parameters
    ----------
    B0_T : float or array-like
        Magnetic field in Tesla.
    S : float
        Electronic spin quantum number.
    r_m : float
        Electron–proton distance in metres.
    tau_c_s : float
        Rotational correlation time in seconds.
    gamma_I : float
        Nuclear gyromagnetic ratio in rad/(s·T).
    gamma_S : float
        Electron gyromagnetic ratio magnitude in rad/(s·T).

    Returns
    -------
    R1 : float or ndarray
        Nuclear paramagnetic R1 in s⁻¹.
    """
    B0 = np.asarray(B0_T, dtype=float)
    b2 = dipolar_prefactor(r_m, gamma_S, gamma_I)
    omega_I = gamma_I * B0
    omega_S = gamma_S * B0
    return ((2.0 / 15.0) * b2 * S * (S + 1)
            * (3.0 * J_solomon(omega_I, tau_c_s)
               + 7.0 * J_solomon(omega_S, tau_c_s)))


def relaxivity_sbm(B0_T, S, r_m, tau_c_s,
                   gamma_I=GAMMA_H_RAD_S_T,
                   gamma_S=GAMMA_E_RAD_S_T):
    """
    Inner-sphere relaxivity r1 (mM⁻¹ s⁻¹) from SBM.

    Assumes q = 1 coordinated water molecule and T1m >> τ_m (fast exchange).

        r1 ≈ R1_param / [H₂O]_mM

    Parameters
    ----------
    B0_T : float or array-like
        Magnetic field in Tesla.
    S, r_m, tau_c_s : float
        Spin, distance (m), correlation time (s).

    Returns
    -------
    r1 : float or ndarray  (mM⁻¹ s⁻¹)
    """
    R1 = R1_sbm(B0_T, S, r_m, tau_c_s, gamma_I, gamma_S)
    return R1 / C_WATER_mM


def nmrd_sbm(B0_values_T, S, r_m, tau_c_s,
             proton_freq=True,
             gamma_I=GAMMA_H_RAD_S_T,
             gamma_S=GAMMA_E_RAD_S_T):
    """
    Full SBM NMRD profile.

    Parameters
    ----------
    B0_values_T : array-like
        Magnetic field values in Tesla.
    S, r_m, tau_c_s : float
        Spin, distance (m), correlation time (s).
    proton_freq : bool
        If True, also return proton Larmor frequency in MHz.

    Returns
    -------
    result : dict with keys:
        'B0_T'   : ndarray  (Tesla)
        'R1'     : ndarray  (s⁻¹)
        'r1'     : ndarray  (mM⁻¹ s⁻¹)
        'nu_H_MHz' : ndarray  (MHz, proton Larmor frequency)  if proton_freq=True
    """
    B0 = np.asarray(B0_values_T, dtype=float)
    R1 = R1_sbm(B0, S, r_m, tau_c_s, gamma_I, gamma_S)
    r1 = R1 / C_WATER_mM
    out = {'B0_T': B0, 'R1': R1, 'r1': r1}
    if proton_freq:
        out['nu_H_MHz'] = gamma_I * B0 / (2.0 * np.pi * 1e6)
    return out
