import numpy as np
from .operators import get_spin_matrices

# Boltzmann constant in MHz/K: k_B = 1.380649e-23 J/K, h = 6.62607015e-34 J·Hz
# k_B / h = 2.0836619e10 Hz/K = 20836.619 MHz/K
KB_MHz_PER_K = 2.0836619e4  # MHz / K


class SpinHamiltonian:
    """
    Electronic Spin Hamiltonian.

    Units:
    - Magnetic field B0 : Tesla (T)
    - Gyromagnetic ratio gamma : MHz / T  (gamma_e ≈ -28024.95 MHz/T)
    - ZFS parameters D, E : MHz
    - Hamiltonian energy : MHz

    Sign convention:
        H_Z = gamma * B0 * Sz
    For a free electron gamma_e < 0, so passing a negative gamma gives the
    physically correct ordering (m = +S is highest energy for an electron
    in a positive B0 field, anti-parallel to the field direction).
    """

    def __init__(self, S):
        self.S = S
        self.Sx, self.Sy, self.Sz = get_spin_matrices(S)
        self.dim = int(2 * S + 1)
        self.S_sq = self.Sx @ self.Sx + self.Sy @ self.Sy + self.Sz @ self.Sz

    # ------------------------------------------------------------------
    # Hamiltonian terms
    # ------------------------------------------------------------------

    def zeeman(self, B0_T, gamma_MHz_T):
        """Electronic Zeeman term: H_Z = gamma * B0 * Sz  [MHz]"""
        return gamma_MHz_T * B0_T * self.Sz

    def zfs_second_order(self, D_MHz, E_MHz=0.0):
        """
        Second-order ZFS: H_ZFS = D[Sz² - S(S+1)/3·I] + E[Sx² - Sy²]  [MHz]

        Properties enforced:
            Tr(H_ZFS) = 0
            H_ZFS = H_ZFS†
            E ≠ 0 mixes states with Δm = ±2
        """
        S_factor = self.S * (self.S + 1)
        term_D = D_MHz * (self.Sz @ self.Sz - (S_factor / 3.0) * np.eye(self.dim))
        term_E = E_MHz * (self.Sx @ self.Sx - self.Sy @ self.Sy)
        return term_D + term_E

    def get_H0(self, B0_T, gamma_MHz_T, D_MHz, E_MHz=0.0):
        """Full static electronic Hamiltonian H0 = H_Z + H_ZFS  [MHz]"""
        return self.zeeman(B0_T, gamma_MHz_T) + self.zfs_second_order(D_MHz, E_MHz)

    # ------------------------------------------------------------------
    # Diagonalization
    # ------------------------------------------------------------------

    def exact_diagonalization(self, H0):
        """
        Diagonalize H0 using eigh (assumes Hermitian).
        Returns eigenvalues and eigenvectors sorted by ascending energy.
        """
        self.check_hermitian(H0)
        evals, evecs = np.linalg.eigh(H0)
        idx = np.argsort(evals)
        return evals[idx], evecs[:, idx]

    # ------------------------------------------------------------------
    # Thermal state
    # ------------------------------------------------------------------

    def thermal_populations(self, evals_MHz, temp_K):
        """
        Boltzmann thermal populations at temperature temp_K.

        Uses energy-shifted exponents for numerical stability:
            P_i = exp(-(E_i - E_min) / k_B T) / Z

        Parameters
        ----------
        evals_MHz : array-like
            Eigenvalues in MHz (must be real).
        temp_K : float
            Temperature in Kelvin.

        Returns
        -------
        pops : ndarray, shape (dim,)
            Normalised thermal populations.  Sum = 1, all >= 0.
        """
        evals = np.asarray(evals_MHz, dtype=float)
        kT_MHz = KB_MHz_PER_K * temp_K          # k_B T in MHz
        shifted = evals - evals.min()            # shift so min = 0 (stability)
        weights = np.exp(-shifted / kT_MHz)
        return weights / weights.sum()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def check_hermitian(H, atol=1e-10):
        """
        Assert that H is Hermitian within tolerance atol.
        Raises ValueError if the check fails.
        """
        if not np.allclose(H, H.conj().T, atol=atol):
            max_err = np.max(np.abs(H - H.conj().T))
            raise ValueError(
                f"Hamiltonian is not Hermitian: max |H - H†| = {max_err:.2e}"
            )
        return True
