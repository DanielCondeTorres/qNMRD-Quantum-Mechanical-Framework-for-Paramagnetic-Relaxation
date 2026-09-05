"""Tests for explicit static-ZFS orientation handling."""

import unittest

import numpy as np

from qnmrd.dynamics.redfield import RedfieldR1
from qnmrd.spin.hamiltonian import SpinHamiltonian
from qnmrd.spin.orientations import fibonacci_orientations, rotate_zfs_tensor
from qnmrd.validation.orientation import static_orientation_average


class TestZFSOrientation(unittest.TestCase):
    def test_principal_tensor_recovers_de_hamiltonian(self):
        ham = SpinHamiltonian(3.5)
        D, E = 600.0, 90.0
        from_tensor = ham.zfs_from_tensor(ham.zfs_principal_tensor(D, E))
        np.testing.assert_allclose(from_tensor, ham.zfs_second_order(D, E), atol=1e-10)

    def test_rotations_preserve_tensor_invariants(self):
        D, E = 600.0, 90.0
        reference = np.linalg.eigvalsh(SpinHamiltonian.zfs_principal_tensor(D, E))
        for rotation in fibonacci_orientations(11):
            tensor = rotate_zfs_tensor(D, E, rotation)
            self.assertAlmostEqual(float(np.trace(tensor)), 0.0, places=10)
            np.testing.assert_allclose(np.linalg.eigvalsh(tensor), reference, atol=1e-10)

    def test_zero_zfs_orientation_average_is_identical(self):
        ham = SpinHamiltonian(3.5)
        redfield = RedfieldR1(ham, r_m=3.0e-10, tau_c_s=50e-12)
        direct = redfield.compute_R1(1.0, D_MHz=0.0, E_MHz=0.0)
        averaged = static_orientation_average(
            redfield, 1.0, D_MHz=0.0, E_MHz=0.0, n_orientations=8
        )
        self.assertAlmostEqual(direct, averaged, places=8)


if __name__ == "__main__":
    unittest.main()
