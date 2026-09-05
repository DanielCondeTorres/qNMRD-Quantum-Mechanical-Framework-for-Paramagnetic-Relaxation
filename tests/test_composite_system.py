"""
test_composite_system.py
========================
Unit tests for the composite electron–nuclear spin system (Phase 5).

Validation hierarchy levels covered:
  Level 4 — Composite system Hilbert-space dimensions
  Level 4 — Operator commutativity [Sα, Iβ] = 0
  Level 4 — Independent subalgebras [Sx,Sy]=iSz and [Ix,Iy]=iIz
  Level 4 — Composite H0 limits (B0=0, D=E=0)
"""

import unittest
import numpy as np
from qnmrd.spin.composite_system import CompositeSpinSystem


class TestCompositeDimensions(unittest.TestCase):
    """Hilbert-space dimension checks."""

    def test_mn_ii_dimension(self):
        """Mn(II): S=5/2, I=1/2 → dim = 6×2 = 12."""
        cs = CompositeSpinSystem(S=5/2, I=0.5)
        self.assertEqual(cs.dim, 12)
        self.assertTrue(cs.check_dimensions())

    def test_gd_iii_dimension(self):
        """Gd(III): S=7/2, I=1/2 → dim = 8×2 = 16."""
        cs = CompositeSpinSystem(S=7/2, I=0.5)
        self.assertEqual(cs.dim, 16)
        self.assertTrue(cs.check_dimensions())

    def test_arbitrary_dimensions(self):
        """Generic dimension (2S+1)(2I+1)."""
        for S, I in [(0.5, 0.5), (1.0, 1.0), (2.5, 0.5), (3.5, 0.5)]:
            cs = CompositeSpinSystem(S, I)
            expected = int(2*S+1) * int(2*I+1)
            self.assertEqual(cs.dim, expected,
                             f"Wrong dim for S={S}, I={I}")


class TestCompositeAlgebra(unittest.TestCase):
    """Spin commutator checks in composite space."""

    def setUp(self):
        self.cs = CompositeSpinSystem(S=2.5, I=0.5)

    def test_electron_nuclear_commutativity(self):
        """[Sα, Iβ] = 0 for all α, β (independent subsystems)."""
        self.assertTrue(self.cs.check_commutators())

    def test_electron_algebra(self):
        """[Sx, Sy] = i Sz in composite space."""
        cs = self.cs
        comm = cs.Sx @ cs.Sy - cs.Sy @ cs.Sx
        np.testing.assert_allclose(comm, 1j * cs.Sz, atol=1e-10)

    def test_nuclear_algebra(self):
        """[Ix, Iy] = i Iz in composite space."""
        cs = self.cs
        comm = cs.Ix @ cs.Iy - cs.Iy @ cs.Ix
        np.testing.assert_allclose(comm, 1j * cs.Iz, atol=1e-10)

    def test_S_squared(self):
        """S² = S(S+1)·I in composite space."""
        cs = self.cs
        S2 = cs.Sx @ cs.Sx + cs.Sy @ cs.Sy + cs.Sz @ cs.Sz
        expected = cs.S * (cs.S + 1) * np.eye(cs.dim)
        np.testing.assert_allclose(S2, expected, atol=1e-10)

    def test_I_squared(self):
        """I² = I(I+1)·I in composite space."""
        cs = self.cs
        I2 = cs.Ix @ cs.Ix + cs.Iy @ cs.Iy + cs.Iz @ cs.Iz
        expected = cs.I * (cs.I + 1) * np.eye(cs.dim)
        np.testing.assert_allclose(I2, expected, atol=1e-10)


class TestCompositeHamiltonian(unittest.TestCase):
    """H0 limit checks."""

    def test_zero_field_gives_zfs(self):
        """B0=0 → H0 = H_ZFS only."""
        cs = CompositeSpinSystem(S=2.5, I=0.5)
        H_zfs = cs.H_zfs(D_MHz=100.0, E_MHz=20.0)
        H0 = cs.H0(B0_T=0.0, D_MHz=100.0, E_MHz=20.0)
        np.testing.assert_allclose(H0, H_zfs, atol=1e-10)

    def test_zero_zfs_gives_zeeman(self):
        """D=E=0 → H0 = H_Ze + H_Zn."""
        cs = CompositeSpinSystem(S=2.5, I=0.5)
        H_Ze = cs.H_zeeman_electron(B0_T=1.0)
        H_Zn = cs.H_zeeman_nuclear(B0_T=1.0)
        H0 = cs.H0(B0_T=1.0, D_MHz=0.0, E_MHz=0.0)
        np.testing.assert_allclose(H0, H_Ze + H_Zn, atol=1e-10)

    def test_h0_hermitian(self):
        """H0 must be Hermitian."""
        cs = CompositeSpinSystem(S=2.5, I=0.5)
        H0 = cs.H0(B0_T=0.5, D_MHz=100.0, E_MHz=30.0)
        np.testing.assert_allclose(H0, H0.conj().T, atol=1e-10)

    def test_nuclear_rdm_trace(self):
        """Tr(ρ_I) = 1 when ρ_total is normalised."""
        cs = CompositeSpinSystem(S=2.5, I=0.5)
        # Build a random Hermitian positive definite ρ
        rho_total = np.eye(cs.dim, dtype=complex) / cs.dim
        rho_I = cs.nuclear_rdm(rho_total)
        self.assertEqual(rho_I.shape, (cs.dimI, cs.dimI))
        np.testing.assert_allclose(np.trace(rho_I), 1.0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
