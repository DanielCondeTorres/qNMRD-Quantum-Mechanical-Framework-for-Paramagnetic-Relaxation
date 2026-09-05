import unittest
import numpy as np
from qnmrd.spin.operators import get_spin_matrices
from qnmrd.spin.hamiltonian import SpinHamiltonian

class TestPhase1B(unittest.TestCase):
    def test_spin_algebra_extended(self):
        """Test A - Spin algebra extended"""
        for S in [0.5, 1.5, 2.5]:
            Sx, Sy, Sz = get_spin_matrices(S)
            S2 = Sx@Sx + Sy@Sy + Sz@Sz
            
            # [Sx, Sy] = i Sz
            np.testing.assert_allclose(Sx@Sy - Sy@Sx, 1j*Sz, atol=1e-10)
            
            # [S^2, S_alpha] = 0
            np.testing.assert_allclose(S2@Sx - Sx@S2, 0, atol=1e-10)
            np.testing.assert_allclose(S2@Sy - Sy@S2, 0, atol=1e-10)
            np.testing.assert_allclose(S2@Sz - Sz@S2, 0, atol=1e-10)
            
            # S^2 = S(S+1)I
            np.testing.assert_allclose(S2, S*(S+1)*np.eye(int(2*S+1)), atol=1e-10)

    def test_zfs_E_zero(self):
        """Test B - ZFS con E=0 coincide con analítico"""
        S = 2.5
        D = 100.0
        ham = SpinHamiltonian(S)
        H_zfs = ham.zfs_second_order(D, 0.0)
        
        # En la base diagonal Sz, los elementos H_ii = D * (m^2 - S(S+1)/3)
        for i in range(ham.dim):
            m = S - i
            expected = D * (m**2 - S*(S+1)/3.0)
            self.assertAlmostEqual(H_zfs[i,i].real, expected, places=10)

    def test_zfs_trace_and_hermiticity(self):
        """Test C y D - ZFS Trace = 0, H = H_dagger, and E!=0 connects Δm=±2"""
        S = 2.5
        ham = SpinHamiltonian(S)
        H_zfs = ham.zfs_second_order(100.0, 20.0)
        
        # Trace = 0
        self.assertAlmostEqual(np.trace(H_zfs), 0.0, places=10)
        
        # Hermiticity
        np.testing.assert_allclose(H_zfs, H_zfs.conj().T, atol=1e-10)
        
        # E!=0 conecta m con m±2
        # La base es |S>, |S-1>, ... El salto de index = 2.
        self.assertTrue(np.abs(H_zfs[0, 2]) > 0)
        self.assertTrue(np.abs(H_zfs[1, 3]) > 0)

    def test_zero_field_and_zfs(self):
        """Test E y F - B0=0 -> ZFS, D=E=0 -> Zeeman"""
        S = 1.5
        ham = SpinHamiltonian(S)
        H_zfs = ham.zfs_second_order(50.0, 10.0)
        H_zee = ham.zeeman(1.0, -28000.0)
        
        # B0 = 0
        np.testing.assert_allclose(ham.get_H0(0.0, -28000.0, 50.0, 10.0), H_zfs, atol=1e-10)
        
        # D=E=0
        np.testing.assert_allclose(ham.get_H0(1.0, -28000.0, 0.0, 0.0), H_zee, atol=1e-10)

if __name__ == '__main__':
    unittest.main()
