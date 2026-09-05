"""
spectral_density.py
===================
Rotational spectral density functions for paramagnetic relaxation.

Unit convention (enforced throughout)
--------------------------------------
- All angular frequencies ω are in rad/s.
- Correlation times τ are in seconds.
- J(ω) has units of seconds [s] (it integrates to give dimensionless products
  when multiplied by coupling constants in rad/s).

The Fourier transform convention used here is:

    J(ω) = ∫₋∞^∞ C(t) e^{iωt} dt  = 2 Re ∫₀^∞ C(t) e^{iωt} dt

For an isotropic rotational model with single-exponential correlation:

    C(t) = exp(-|t|/τ_R)

this gives:

    J(ω) = 2τ_R / (1 + ω²τ_R²)

This is the form used in the Bloembergen–Morgan and Redfield literature
(see e.g. Helm 2006, Merbach et al.).  Note: some texts include an extra
factor of 2/5 or 1/5 for the rank-2 spherical harmonic prefactor — that
prefactor belongs in the coupling constant, not in J(ω) itself.

References
----------
Helm, L. (2006) Prog. NMR Spectrosc. 49, 45–64.
Abragam, A. (1961) Principles of Nuclear Magnetism. Oxford.
Bloembergen, N.; Morgan, L.O. (1961) J. Chem. Phys. 34, 842.
"""

import numpy as np


def J_BPP(omega_rad_s, tau_c_s):
    """
    Isotropic single-exponential (BPP / Lorentzian) spectral density.

        J(ω) = 2τ_c / (1 + ω²τ_c²)

    Parameters
    ----------
    omega_rad_s : float or array-like
        Angular frequency in rad/s.
    tau_c_s : float
        Rotational correlation time in seconds.

    Returns
    -------
    float or ndarray
        J(ω) in seconds [s].

    Notes
    -----
    Normalisation check:  ∫₀^∞ J(ω) dω / π = τ_c  (integral of Lorentzian)
    """
    w = np.asarray(omega_rad_s, dtype=float)
    return 2.0 * tau_c_s / (1.0 + (w * tau_c_s) ** 2)


def J_Cole_Davidson(omega_rad_s, tau_c_s, beta):
    """
    Cole–Davidson spectral density for a distribution of correlation times.

        C(t) corresponds to a skewed distribution of τ values.

    This is an approximation; the exact expression requires numerical
    integration.  Not recommended for the primary calculations.

    Parameters
    ----------
    omega_rad_s : float or array-like
        Angular frequency in rad/s.
    tau_c_s : float
        Mean rotational correlation time in seconds.
    beta : float
        Stretching exponent in (0, 1].  beta=1 recovers BPP.

    Returns
    -------
    float or ndarray
        J(ω) in seconds [s].
    """
    w = np.asarray(omega_rad_s, dtype=float)
    phi = np.arctan(w * tau_c_s)
    denom = (1.0 + (w * tau_c_s) ** 2) ** (beta / 2.0)
    return 2.0 * tau_c_s * np.sin(beta * phi) / (w * denom + 1e-300)


def J_from_tau(omega_rad_s, tau_c_s, model='BPP', **kwargs):
    """
    Unified spectral density dispatcher.

    Parameters
    ----------
    omega_rad_s : float or array-like
        Angular frequency in rad/s.
    tau_c_s : float
        Rotational correlation time in seconds.
    model : str
        'BPP' (default) — single-exponential Lorentzian.
    **kwargs
        Extra model parameters (e.g. beta for Cole–Davidson).

    Returns
    -------
    float or ndarray
        J(ω) in seconds [s].
    """
    if model == 'BPP':
        return J_BPP(omega_rad_s, tau_c_s)
    elif model == 'Cole-Davidson':
        beta = kwargs.get('beta', 0.7)
        return J_Cole_Davidson(omega_rad_s, tau_c_s, beta)
    else:
        raise ValueError(f"Unknown spectral density model: '{model}'. "
                         f"Supported: 'BPP', 'Cole-Davidson'")


def MHz_to_rad_s(freq_MHz):
    """Convert MHz to rad/s:  ω = 2π·f."""
    return 2.0 * np.pi * freq_MHz * 1e6


def rad_s_to_MHz(omega_rad_s):
    """Convert rad/s to MHz:  f = ω / (2π)."""
    return omega_rad_s / (2.0 * np.pi * 1e6)


def larmor_angular_frequency(B0_T, gamma_MHz_T):
    """
    Larmor angular frequency in rad/s.

    Parameters
    ----------
    B0_T : float or array
        Magnetic field in Tesla.
    gamma_MHz_T : float
        Gyromagnetic ratio in MHz/T.

    Returns
    -------
    float or ndarray
        ω = |gamma| * B0 * 2π  in rad/s.
    """
    # gamma in MHz/T → multiply by 1e6 to get Hz/T → multiply by 2π for rad/s
    return np.abs(gamma_MHz_T) * 1e6 * 2.0 * np.pi * np.asarray(B0_T)
