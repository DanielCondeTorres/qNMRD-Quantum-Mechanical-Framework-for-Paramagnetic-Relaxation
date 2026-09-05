import unittest
import numpy as np
from qnmrd.spin.operators import get_spin_matrices
from qnmrd.spin.hamiltonian import SpinHamiltonian

class TestPhase1(unittest.TestCase):
    def test_spin_algebra(self):
        """Test 1 - Spin algebra: [Sx, Sy] = i Sz and S^2 = S(S+1)I"""
        for S in [0.5, 1.0, 1.5, 2.0, 2.5, 3.5]:
            Sx, Sy, Sz = get_spin_matrices(S)
            
            # Conmutador [Sx, Sy] = Sx Sy - Sy Sx
            comm = Sx @ Sy - Sy @ Sx
            expected = 1j * Sz
            np.testing.assert_allclose(comm, expected, atol=1e-10, err_msg=f"Commutator failed for S={S}")
            
            # S^2 = Sx^2 + Sy^2 + Sz^2 = S(S+1)I
            S_sq = Sx @ Sx + Sy @ Sy + Sz @ Sz
            expected_S_sq = S * (S + 1) * np.eye(int(2*S+1))
            np.testing.assert_allclose(S_sq, expected_S_sq, atol=1e-10, err_msg=f"S^2 failed for S={S}")

    def test_zero_field(self):
        """Test 2 - Zero field: B0=0 debe recuperar exclusivamente el espectro ZFS."""
        S = 2.5
        ham = SpinHamiltonian(S)
        H_zfs = ham.zfs_second_order(D_MHz=100.0, E_MHz=20.0)
        H0 = ham.get_H0(B0_z_MHz=0.0, D_MHz=100.0, E_MHz=20.0)
        np.testing.assert_allclose(H0, H_zfs, atol=1e-10)

    def test_zero_zfs(self):
        """Test 3 - Zero ZFS: D=E=0 debe recuperar comportamiento puramente Zeeman."""
        S = 3.5
        ham = SpinHamiltonian(S)
        H_zeeman = ham.zeeman(B0_z_MHz=500.0)
        H0 = ham.get_H0(B0_z_MHz=500.0, D_MHz=0.0, E_MHz=0.0)
        np.testing.assert_allclose(H0, H_zeeman, atol=1e-10)
        
        # Las energías deberían ser exactamente m * B0_z_MHz
        evals, _ = ham.exact_diagonalization(H0)
        # Como los autoestados están ordenados, los m van de -S a S
        expected_evals = np.array([-S + i for i in range(int(2*S+1))]) * 500.0
        np.testing.assert_allclose(evals, expected_evals, atol=1e-10)

    def test_thermal_populations(self):
        """Test 6 & 9 - Thermal equilibrium: Sum(P_i) = 1 and Positivity"""
        ham = SpinHamiltonian(S=2.5)
        evals = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        pops = ham.thermal_populations(evals, temp_K=300)
        
        # Positivity
        self.assertTrue(np.all(pops >= 0))
        # Sum = 1
        np.testing.assert_allclose(np.sum(pops), 1.0, atol=1e-10)

if __name__ == '__main__':
    unittest.main()
