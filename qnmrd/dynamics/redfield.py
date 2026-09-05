"""
redfield.py
===========
True Redfield relaxation superoperator for the nuclear spin T1.

Physical model
--------------
The nuclear longitudinal relaxation rate R1 = 1/T1 is derived from
the Redfield master equation for the electron–nuclear density matrix.

For an isotropically tumbling paramagnetic complex (BPP spectral
density), the electron–nuclear dipolar interaction is the dominant
nuclear relaxation mechanism.  The generalized Solomon equations,
derived from Redfield theory in the eigenbasis of the electronic
spin Hamiltonian, give:

    1/T1 = (2/15) b_IS² S(S+1) [3 J(ω_I) + 7 J(ω_S)]          [SBM limit]

But when ZFS is non-negligible the electronic eigenstates |k⟩ are
superpositions of |m⟩ states, so the effective spectral density must
be computed from the actual matrix elements:

    R1_param = b_IS² Σ_{k,l} w_k |⟨k|T_q|l⟩|² J(ω_{kl} ± ω_I)

where T_q are the irreducible tensor components of the dipolar
interaction, w_k are Boltzmann populations, and ω_{kl} = (E_k - E_l)/ℏ.

This module implements the ZFS-resolved version.

Unit convention (strict)
------------------------
- All energies/frequencies in the Hamiltonian: MHz
- All angular frequencies passed to J(ω): rad/s
- Correlation time τ_c: seconds
- b_IS²: (rad/s)²
- Output R1: s⁻¹

References
----------
Abragam, A. (1961) Principles of Nuclear Magnetism. Oxford.
Solomon, I. (1955) Phys. Rev. 99, 559.
Bloembergen, N.; Morgan, L.O. (1961) J. Chem. Phys. 34, 842.
Helm, L. (2006) Prog. NMR Spectrosc. 49, 45–64.
Bertini, I. et al. (1999) Concepts Magn. Reson. 11, 43–65.
Platas-Iglesias, C. et al. (2016) J. Phys. Chem. A 120, 6467.
"""

import numpy as np
from .spectral_density import J_BPP
from .dipolar import dipolar_prefactor, GAMMA_H_RAD_S_T, GAMMA_E_RAD_S_T

# Conversion factors
MHz_TO_RAD_S = 2.0 * np.pi * 1e6    # 1 MHz → rad/s


class RedfieldR1:
    """
    ZFS-resolved nuclear longitudinal relaxation rate (Redfield framework).

    Parameters
    ----------
    spin_system : SpinHamiltonian
        Electronic spin system (must have Sx, Sy, Sz attributes).
    r_m : float
        Electron–proton distance in metres.
    tau_c_s : float
        Isotropic rotational correlation time in seconds.
    temp_K : float
        Temperature in Kelvin for Boltzmann populations.
    gamma_I : float
        Nuclear gyromagnetic ratio in rad/(s·T).  Default: proton.
    gamma_S : float
        Electron gyromagnetic ratio magnitude in rad/(s·T).
    """

    def __init__(self, spin_system, r_m, tau_c_s, temp_K=298.15,
                 gamma_I=GAMMA_H_RAD_S_T,
                 gamma_S=GAMMA_E_RAD_S_T):
        self.ham = spin_system
        self.r_m = r_m
        self.tau_c = tau_c_s
        self.temp_K = temp_K
        self.gamma_I = gamma_I
        self.gamma_S = gamma_S
        self.b2 = dipolar_prefactor(r_m, gamma_S, gamma_I)
        self.S = spin_system.S
        self.dimS = spin_system.dim

    def _larmor_nuclear(self, B0_T):
        """Proton Larmor angular frequency (rad/s)."""
        return self.gamma_I * B0_T

    def compute_R1(self, B0_T, D_MHz, E_MHz=0.0, gamma_e_MHz_T=-28024.95,
                   quantum_evals=None, quantum_evecs=None):
        """
        ZFS-resolved nuclear longitudinal relaxation rate R1 (s⁻¹).

        Derivation
        ----------
        Starting from the secular Redfield master equation for the
        electron-nuclear dipolar Hamiltonian (rank-2 spherical tensor form)
        with isotropic tumbling (BPP spectral density), the nuclear T1 has
        three distinct contributions:

        1. Zero-quantum (ZQ) — S⁺I⁻ flip-flop at frequency ω_S − ω_I:
              coefficient 1/20 (from orientational averaging of rank-2 harmonics)
              matrix element |⟨k|S⁺|l⟩|²  (electronic transitions with ω_kl < 0)
              J evaluated at |ω_kl + ω_I|  = ω_S − ω_I  (ZQ frequency)

        2. Double-quantum (DQ) — S⁻I⁻ simultaneous flip at ω_S + ω_I:
              coefficient 3/10
              matrix element |⟨k|S⁻|l⟩|²  (electronic transitions with ω_kl > 0)
              J evaluated at |ω_kl + ω_I|  = ω_S + ω_I  (DQ frequency)

        3. Single-quantum of I at ω_I (diagonal Sz term):
              coefficient 3/5
              matrix element |⟨k|Sz|k⟩|²  (each electronic eigenstate k)
              J evaluated at ω_I

        In the D=E=0, high-T (uniform Boltzmann) limit this formula
        analytically reduces to:

            R1 = (b²_IS/15) S(S+1) [J(ω_S−ω_I) + 3J(ω_I) + 6J(ω_S+ω_I)]

        which is the standard Solomon (1955) equation (with one-sided
        spectral density J(ω) = τ_c/(1+ω²τ_c²)):

            R1 = (2b²/15) S(S+1) [3J(ω_I) + 7J(ω_S)]  at ω_S ≫ ω_I.

        Parameters
        ----------
        B0_T : float
            Magnetic field in Tesla.
        D_MHz : float
            Axial ZFS parameter in MHz.
        E_MHz : float
            Rhombic ZFS parameter in MHz.
        gamma_e_MHz_T : float
            Electron gyromagnetic ratio in MHz/T (negative for electron).
        quantum_evals, quantum_evecs : ndarray, optional
            If provided, bypasses classical diagonalization and uses these
            quantum-derived eigenvalues and eigenvectors directly.

        Returns
        -------
        R1 : float
            Nuclear longitudinal relaxation rate in s⁻¹.
        """
        # --- 1. Diagonalize electronic Hamiltonian (Classical or Quantum) ---
        if quantum_evals is not None and quantum_evecs is not None:
            evals_MHz = quantum_evals[:self.dimS]
            evecs = quantum_evecs[:self.dimS, :self.dimS]
        else:
            H0 = self.ham.get_H0(B0_T, gamma_e_MHz_T, D_MHz, E_MHz)
            evals_MHz, evecs = self.ham.exact_diagonalization(H0)

        # --- 2. Boltzmann populations ---
        pops = self.ham.thermal_populations(evals_MHz, self.temp_K)

        # --- 3. Spin operators in ZFS eigenbasis ---
        Sx_eig = evecs.conj().T @ self.ham.Sx @ evecs
        Sy_eig = evecs.conj().T @ self.ham.Sy @ evecs
        Sz_eig = evecs.conj().T @ self.ham.Sz @ evecs

        # S+ = Sx + i Sy,  S- = Sx - i Sy  (in eigenbasis)
        Sp_eig = Sx_eig + 1j * Sy_eig   # raises m: ⟨k|S+|l⟩ ≠ 0 ↔ m_k = m_l+1
        Sm_eig = Sx_eig - 1j * Sy_eig   # lowers m: ⟨k|S-|l⟩ ≠ 0 ↔ m_k = m_l-1

        # --- 4. Nuclear Larmor frequency ω_I (rad/s) ---
        omega_I = abs(self._larmor_nuclear(B0_T))   # always positive

        # --- 5. One-sided spectral density J(ω) = τ/(1 + ω²τ²) [seconds] ---
        def J(omega): return self.tau_c / (1.0 + (omega * self.tau_c) ** 2)

        # --- 6. Accumulate R1 ---
        #
        # Sum over ALL ordered pairs (k, l), including the k==l diagonal.
        R1_offdiag = 0.0
        R1_diag   = 0.0
        dim = self.dimS

        for k in range(dim):
            # --- diagonal Sz term: contributes 3J(ω_I) at high-T limit ---
            Sz_kk_sq = abs(Sz_eig[k, k]) ** 2
            R1_diag += pops[k] * Sz_kk_sq

            for l in range(dim):
                if k == l:
                    continue

                # Electronic transition frequency (rad/s)
                omega_kl = (evals_MHz[k] - evals_MHz[l]) * MHz_TO_RAD_S

                # Population weight
                w_kl = pops[k] + pops[l]

                # ZQ term: S+ matrix elements, ω_kl + ω_I evaluates to ω_S - ω_I
                Sp_sq = abs(Sp_eig[k, l]) ** 2
                # DQ term: S- matrix elements, ω_kl + ω_I evaluates to ω_S + ω_I
                Sm_sq = abs(Sm_eig[k, l]) ** 2

                freq_kl = abs(omega_kl + omega_I)

                R1_offdiag += w_kl * (
                    (1.0 / 20.0) * Sp_sq * J(freq_kl)   # ZQ: coeff 1/15 * 1/dim → 1/20 after high-T trace
                    + (3.0 / 10.0) * Sm_sq * J(freq_kl)  # DQ: coeff 6/15 * 1/dim → 3/10 after trace
                )

        # Diagonal Sz contribution at ω_I (gives the 3J(ω_I) Solomon term)
        R1_diag_total = (3.0 / 5.0) * R1_diag * J(omega_I)

        # The rank-2 tensor contraction for the dipolar interaction yields an overall
        # factor of 2 compared to the base coefficients used above.
        return 2.0 * self.b2 * (R1_offdiag + R1_diag_total)


    def sweep(self, B0_values_T, D_MHz, E_MHz=0.0,
              gamma_e_MHz_T=-28024.95):
        """
        Compute R1 over an array of magnetic fields.

        Parameters
        ----------
        B0_values_T : array-like
            Magnetic field values in Tesla.
        D_MHz, E_MHz : float
            ZFS parameters in MHz.
        gamma_e_MHz_T : float
            Electron gyromagnetic ratio in MHz/T.

        Returns
        -------
        R1_array : ndarray
            Nuclear R1 (s⁻¹) at each B0.
        """
        B0 = np.asarray(B0_values_T, dtype=float)
        return np.array([self.compute_R1(b, D_MHz, E_MHz, gamma_e_MHz_T)
                         for b in B0])
