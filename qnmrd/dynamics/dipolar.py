"""
dipolar.py
==========
Electron–nuclear magnetic dipole–dipole interaction.

Physical model
--------------
The secular part of the dipolar Hamiltonian between an electronic spin S
and a nuclear spin I, in a rigid-rotor model (isotropic tumbling), is:

    H_DD = b_IS * [ 3(S·r̂)(I·r̂) - S·I ]

where the coupling constant is:

    b_IS = -(μ0/4π) * (ℏ γ_S γ_I) / r³

Units
-----
Throughout this module energies are in rad/s (angular frequency).
The coupling constant C_dip returned by ``dipolar_prefactor`` has units
of (rad/s)² and is used directly in spectral density expressions.

Convention
----------
Orientational averaging for isotropic tumbling is applied analytically
through the rank-2 spherical harmonic correlation function
    ⟨Y₂ₘ*(Ω(t)) Y₂ₙ(Ω(0))⟩ = δₘₙ exp(-t/τ_R) / (4π)
which, when folded into the Redfield integrals, yields the standard
factor of 2/15 (for the outer Solomon equation) or 3/2 (for inner
Redfield cross-terms), depending on the observable.

References
----------
Abragam, A. (1961) Principles of Nuclear Magnetism. Oxford.
Solomon, I. (1955) Phys. Rev. 99, 559.
Helm, L. (2006) Prog. NMR Spectrosc. 49, 45–64.
"""

import numpy as np

# Physical constants (SI)
MU0_OVER_4PI = 1e-7          # T·m/A  (μ₀/4π)
HBAR          = 1.054571817e-34  # J·s
GAMMA_H_RAD_S_T = 2.6752219e8   # rad/(s·T)  — proton gyromagnetic ratio
GAMMA_E_RAD_S_T = 1.760860e11   # rad/(s·T)  — |electron gyromagnetic ratio|


def dipolar_coupling_rad_s(r_m, gamma_S_rad_s_T=GAMMA_E_RAD_S_T,
                            gamma_I_rad_s_T=GAMMA_H_RAD_S_T):
    """
    Scalar dipolar coupling constant  b_IS  (rad/s).

        b_IS = -(μ₀/4π) * (ℏ γ_S γ_I) / r³

    Parameters
    ----------
    r_m : float
        Electron–proton distance in metres.
    gamma_S_rad_s_T : float
        Electron gyromagnetic ratio magnitude in rad/(s·T).
    gamma_I_rad_s_T : float
        Nuclear gyromagnetic ratio in rad/(s·T).

    Returns
    -------
    b_IS : float
        Dipolar coupling constant in rad/s.
    """
    return -(MU0_OVER_4PI * HBAR * gamma_S_rad_s_T * gamma_I_rad_s_T
             / r_m**3)


def dipolar_prefactor(r_m, gamma_S_rad_s_T=GAMMA_E_RAD_S_T,
                      gamma_I_rad_s_T=GAMMA_H_RAD_S_T):
    """
    Squared dipolar coupling prefactor  b_IS²  [(rad/s)²].

    This is the quantity that appears multiplying the spectral density
    functions in the Solomon equation and the Redfield tensor.

    Parameters
    ----------
    r_m : float
        Electron–proton distance in metres (1 Å = 1e-10 m).

    Returns
    -------
    float
        b_IS² in (rad/s)².
    """
    b = dipolar_coupling_rad_s(r_m, gamma_S_rad_s_T, gamma_I_rad_s_T)
    return b ** 2


def transition_matrix_elements(evecs_S, Sx, Sy, Sz):
    """
    Compute spin transition matrix elements in the eigenbasis.

    Returns |⟨i|Sα|j⟩|² for α = x, y, z and all pairs (i,j).

    Parameters
    ----------
    evecs_S : ndarray, shape (dimS, dimS)
        Eigenvectors of the electronic Hamiltonian (columns).
    Sx, Sy, Sz : ndarray, shape (dimS, dimS)
        Spin matrices in the original |m⟩ basis.

    Returns
    -------
    dict with keys 'Sx', 'Sy', 'Sz', each a (dimS, dimS) array of
    |⟨i|Sα|j⟩|² values.
    """
    Sx_eig = evecs_S.conj().T @ Sx @ evecs_S
    Sy_eig = evecs_S.conj().T @ Sy @ evecs_S
    Sz_eig = evecs_S.conj().T @ Sz @ evecs_S
    return {
        'Sx': np.abs(Sx_eig) ** 2,
        'Sy': np.abs(Sy_eig) ** 2,
        'Sz': np.abs(Sz_eig) ** 2,
    }
