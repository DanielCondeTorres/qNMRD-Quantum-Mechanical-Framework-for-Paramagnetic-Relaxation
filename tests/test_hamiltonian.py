import unittest
import numpy as np
from qnmrd.spin.operators import get_spin_matrices
from qnmrd.spin.hamiltonian import SpinHamiltonian

GAMMA_E = -28024.95  # MHz/T


class TestPhase1(unittest.TestCase):
    def test_spin_algebra(self):
        """Test 1 — Spin algebra: [Sx, Sy] = i Sz and S² = S(S+1)I"""
        for S in [0.5, 1.0, 1.5, 2.0, 2.5, 3.5]:
            Sx, Sy, Sz = get_spin_matrices(S)

            # Commutator [Sx, Sy] = i Sz
            comm = Sx @ Sy - Sy @ Sx
            np.testing.assert_allclose(comm, 1j * Sz, atol=1e-10,
                                       err_msg=f"[Sx,Sy]=iSz failed for S={S}")

            # S² = S(S+1) I
            S_sq = Sx @ Sx + Sy @ Sy + Sz @ Sz
            expected = S * (S + 1) * np.eye(int(2 * S + 1))
            np.testing.assert_allclose(S_sq, expected, atol=1e-10,
                                       err_msg=f"S²=S(S+1)I failed for S={S}")

    def test_zero_field(self):
        """Test 2 — B0=0 must recover the pure ZFS spectrum."""
        S = 2.5
        ham = SpinHamiltonian(S)
        H_zfs = ham.zfs_second_order(D_MHz=100.0, E_MHz=20.0)
        # B0=0 → Zeeman vanishes regardless of gamma
        H0 = ham.get_H0(B0_T=0.0, gamma_MHz_T=GAMMA_E, D_MHz=100.0, E_MHz=20.0)
        np.testing.assert_allclose(H0, H_zfs, atol=1e-10)

    def test_zero_zfs(self):
        """Test 3 — D=E=0 must recover pure Zeeman eigenvalues."""
        S = 3.5
        ham = SpinHamiltonian(S)
        B0_T = 0.5           # 0.5 T
        H_zeeman = ham.zeeman(B0_T=B0_T, gamma_MHz_T=GAMMA_E)
        H0 = ham.get_H0(B0_T=B0_T, gamma_MHz_T=GAMMA_E, D_MHz=0.0, E_MHz=0.0)
        np.testing.assert_allclose(H0, H_zeeman, atol=1e-10)

        # Eigenvalues should be m * gamma_e * B0 (sorted ascending)
        evals, _ = ham.exact_diagonalization(H0)
        m_values = np.array([-S + i for i in range(int(2 * S + 1))])
        expected_evals = np.sort(m_values * GAMMA_E * B0_T)
        np.testing.assert_allclose(evals, expected_evals, atol=1e-10)

    def test_thermal_populations(self):
        """Test 6 & 9 — Thermal populations: sum=1, positivity, high-T limit."""
        ham = SpinHamiltonian(S=2.5)
        evals = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

        pops = ham.thermal_populations(evals, temp_K=300)

        # Positivity
        self.assertTrue(np.all(pops >= 0), "Populations must be non-negative")
        # Normalisation
        np.testing.assert_allclose(np.sum(pops), 1.0, atol=1e-10,
                                   err_msg="Populations must sum to 1")
        # High-T limit: populations approach uniform (1/dim)
        pops_highT = ham.thermal_populations(evals, temp_K=1e8)
        np.testing.assert_allclose(pops_highT,
                                   np.ones(len(evals)) / len(evals),
                                   atol=1e-6,
                                   err_msg="High-T populations should be uniform")
        # Low-T limit: lowest energy level dominates when k_B T << gap
        # gap = 100 MHz, k_B * 1e-8 K = 2.08e-4 MHz << 100 MHz
        pops_lowT = ham.thermal_populations(evals, temp_K=1e-8)
        self.assertAlmostEqual(pops_lowT[0], 1.0, places=4,
                               msg="Low-T population should concentrate in ground state")


if __name__ == '__main__':
    unittest.main()
