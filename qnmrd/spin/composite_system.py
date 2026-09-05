"""
composite_system.py
===================
Electron–nuclear composite spin system.

Builds the tensor-product Hilbert space H_S ⊗ H_I (dim = (2S+1)(2I+1)).

Convention
----------
All operators live in the composite space:
    A_S  →  A_S ⊗ I_I      (electronic operator)
    A_I  →  I_S ⊗ A_I      (nuclear operator)

Units
-----
All energies / Hamiltonians are in MHz (same convention as SpinHamiltonian).
"""

import numpy as np
from .operators import get_spin_matrices
from .hamiltonian import SpinHamiltonian

# Physical constants
GAMMA_E_MHz_T = -28024.95   # electron gyromagnetic ratio, MHz/T
GAMMA_H_MHz_T = 42.5774     # proton gyromagnetic ratio,   MHz/T


class CompositeSpinSystem:
    """
    Coupled electron (spin S) + nucleus (spin I) system.

    The composite Hilbert space has dimension dim_total = (2S+1)(2I+1).
    Operators are built by Kronecker products:

        Sα_total = Sα ⊗ I_I
        Iα_total = I_S ⊗ Iα

    Parameters
    ----------
    S : float
        Electronic spin quantum number (e.g. 5/2 for Mn²⁺, 7/2 for Gd³⁺).
    I : float
        Nuclear spin quantum number (e.g. 0.5 for ¹H).
    """

    def __init__(self, S, I=0.5):
        self.S = S
        self.I = I
        self.dimS = int(2 * S + 1)
        self.dimI = int(2 * I + 1)
        self.dim = self.dimS * self.dimI

        # --- Electronic spin matrices in composite space ---
        Sx, Sy, Sz = get_spin_matrices(S)
        II = np.eye(self.dimI)
        self.Sx = np.kron(Sx, II)
        self.Sy = np.kron(Sy, II)
        self.Sz = np.kron(Sz, II)

        # --- Nuclear spin matrices in composite space ---
        Ix, Iy, Iz = get_spin_matrices(I)
        IS = np.eye(self.dimS)
        self.Ix = np.kron(IS, Ix)
        self.Iy = np.kron(IS, Iy)
        self.Iz = np.kron(IS, Iz)

        # --- Raising/lowering operators (useful for T1 calculation) ---
        # S+ = Sx + i Sy,  S- = Sx - i Sy
        self.Sp = self.Sx + 1j * self.Sy
        self.Sm = self.Sx - 1j * self.Sy
        self.Ip = self.Ix + 1j * self.Iy
        self.Im = self.Ix - 1j * self.Iy

        # --- Electronic spin in bare space (for ZFS construction) ---
        self._ham_e = SpinHamiltonian(S)

    # ------------------------------------------------------------------
    # Static Hamiltonians
    # ------------------------------------------------------------------

    def H_zeeman_electron(self, B0_T, gamma_e_MHz_T=GAMMA_E_MHz_T):
        """
        Electronic Zeeman Hamiltonian in composite space  [MHz].

            H_Ze = gamma_e * B0 * Sz_total
        """
        return gamma_e_MHz_T * B0_T * self.Sz

    def H_zeeman_nuclear(self, B0_T, gamma_n_MHz_T=GAMMA_H_MHz_T):
        """
        Nuclear Zeeman Hamiltonian in composite space  [MHz].

            H_Zn = gamma_n * B0 * Iz_total
        """
        return gamma_n_MHz_T * B0_T * self.Iz

    def H_zfs(self, D_MHz, E_MHz=0.0):
        """
        Second-order ZFS in composite space  [MHz].

            H_ZFS = D[Sz² - S(S+1)/3·I] + E[Sx² - Sy²]

        Acts only on the electronic subspace (Kronecker with I_I already
        included in self.Sx, self.Sy, self.Sz).
        """
        S = self.S
        S_factor = S * (S + 1)
        term_D = D_MHz * (self.Sz @ self.Sz
                          - (S_factor / 3.0) * np.eye(self.dim))
        term_E = E_MHz * (self.Sx @ self.Sx - self.Sy @ self.Sy)
        return term_D + term_E

    def H0(self, B0_T, D_MHz, E_MHz=0.0,
           gamma_e_MHz_T=GAMMA_E_MHz_T,
           gamma_n_MHz_T=GAMMA_H_MHz_T,
           include_hyperfine=False, A_MHz=0.0):
        """
        Full static Hamiltonian  [MHz]:

            H0 = H_Ze + H_Zn + H_ZFS  [+ H_hf]

        Parameters
        ----------
        B0_T : float
            External magnetic field in Tesla.
        D_MHz : float
            Axial ZFS in MHz.
        E_MHz : float
            Rhombic ZFS in MHz.
        gamma_e_MHz_T : float
            Electron gyromagnetic ratio in MHz/T.
        gamma_n_MHz_T : float
            Nuclear gyromagnetic ratio in MHz/T.
        include_hyperfine : bool
            Whether to add isotropic hyperfine coupling A·(S·I).
        A_MHz : float
            Isotropic hyperfine coupling constant in MHz (if used).

        Returns
        -------
        H : ndarray, shape (dim, dim), complex
            Total static Hamiltonian in MHz.
        """
        H = (self.H_zeeman_electron(B0_T, gamma_e_MHz_T)
             + self.H_zeeman_nuclear(B0_T, gamma_n_MHz_T)
             + self.H_zfs(D_MHz, E_MHz))
        if include_hyperfine and A_MHz != 0.0:
            H_hf = A_MHz * (self.Sx @ self.Ix
                            + self.Sy @ self.Iy
                            + self.Sz @ self.Iz)
            H += H_hf
        return H

    # ------------------------------------------------------------------
    # Diagonalization
    # ------------------------------------------------------------------

    def diagonalize(self, H):
        """
        Diagonalize Hermitian H in composite space.

        Returns
        -------
        evals : ndarray, shape (dim,)
            Eigenvalues in ascending order [MHz].
        evecs : ndarray, shape (dim, dim)
            Columns are eigenvectors, ordered with evals.
        """
        SpinHamiltonian.check_hermitian(H)
        evals, evecs = np.linalg.eigh(H)
        idx = np.argsort(evals)
        return evals[idx], evecs[:, idx]

    def thermal_populations(self, evals_MHz, temp_K):
        """Boltzmann populations (delegates to SpinHamiltonian utility)."""
        return self._ham_e.thermal_populations(evals_MHz, temp_K)

    # ------------------------------------------------------------------
    # Nuclear reduced density matrix
    # ------------------------------------------------------------------

    def nuclear_rdm(self, rho_total):
        """
        Partial trace over the electronic degrees of freedom.

        rho_I = Tr_S [rho_total]

        Returns the (2I+1)×(2I+1) nuclear reduced density matrix.
        """
        rho = rho_total.reshape(self.dimS, self.dimI, self.dimS, self.dimI)
        # Tr_S: sum over electronic indices
        return np.einsum('iajb->ab', rho)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def check_dimensions(self):
        """Assert all composite operators have correct shape."""
        expected = (self.dim, self.dim)
        for name, op in [('Sx', self.Sx), ('Sy', self.Sy), ('Sz', self.Sz),
                         ('Ix', self.Ix), ('Iy', self.Iy), ('Iz', self.Iz)]:
            assert op.shape == expected, (
                f"Operator {name} has shape {op.shape}, expected {expected}")
        return True

    def check_commutators(self, atol=1e-10):
        """Verify electron–nuclear operator commutativity and individual algebras."""
        # [Sα, Iβ] = 0  (independent subsystems)
        for Sop, sname in [(self.Sx, 'Sx'), (self.Sy, 'Sy'), (self.Sz, 'Sz')]:
            for Iop, iname in [(self.Ix, 'Ix'), (self.Iy, 'Iy'), (self.Iz, 'Iz')]:
                comm = Sop @ Iop - Iop @ Sop
                assert np.allclose(comm, 0, atol=atol), (
                    f"[{sname}, {iname}] ≠ 0 — subsystems not commuting")

        # [Sx, Sy] = i Sz
        assert np.allclose(self.Sx @ self.Sy - self.Sy @ self.Sx,
                           1j * self.Sz, atol=atol)
        # [Ix, Iy] = i Iz
        assert np.allclose(self.Ix @ self.Iy - self.Iy @ self.Ix,
                           1j * self.Iz, atol=atol)
        return True
