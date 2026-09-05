"""
test_redfield.py
================
Unit tests for the Redfield R1 calculation and SBM baseline.

Validation hierarchy levels covered:
  Level 6 — J(ω) normalisation and limits
  Level 7 — Redfield R1: D=E=0 limit must approach SBM
  Level 7 — R1 > 0 for all physical inputs
  Level 8 — SBM vs Redfield at high field (ZFS negligible)
"""

import unittest
import numpy as np
from qnmrd.spin.hamiltonian import SpinHamiltonian
from qnmrd.dynamics.spectral_density import J_BPP
from qnmrd.dynamics.redfield import RedfieldR1
from qnmrd.validation.sbm import R1_sbm
from qnmrd.validation.zeeman_limit import zeeman_limit_errors


class TestSpectralDensity(unittest.TestCase):
    """Level 6: J(ω) normalisation and limits."""

    def test_jbpp_zero_frequency(self):
        """J(0) = 2τ_c."""
        tau = 50e-12
        self.assertAlmostEqual(J_BPP(0.0, tau), 2.0 * tau, places=20)

    def test_jbpp_high_frequency_decay(self):
        """J(ω) → 0 as ω → ∞."""
        tau = 50e-12
        high_omega = 1e15  # far above 1/τ_c
        self.assertLess(J_BPP(high_omega, tau), 1e-15)

    def test_jbpp_positive(self):
        """J(ω) >= 0 for all ω."""
        omegas = np.logspace(6, 12, 50)
        for tau in [10e-12, 100e-12, 500e-12]:
            vals = J_BPP(omegas, tau)
            self.assertTrue(np.all(vals >= 0),
                            f"J(ω) < 0 for τ_c={tau}")

    def test_jbpp_symmetry(self):
        """J(ω) = J(-ω)  (even function)."""
        omega = np.array([1e8, 1e9, 1e10])
        tau = 80e-12
        np.testing.assert_allclose(J_BPP(omega, tau), J_BPP(-omega, tau))


class TestSBM(unittest.TestCase):
    """SBM baseline sanity checks."""

    def test_r1_positive(self):
        """R1 > 0 for all physical B0."""
        B0 = np.logspace(-4, 1, 30)
        R1 = R1_sbm(B0, S=2.5, r_m=2.83e-10, tau_c_s=30e-12)
        self.assertTrue(np.all(R1 > 0))

    def test_r1_decreases_at_high_field(self):
        """R1 should decrease at high field (ω_S τ_c >> 1 dispersion)."""
        tau = 30e-12
        R1_low  = R1_sbm(1e-3, S=2.5, r_m=2.83e-10, tau_c_s=tau)
        R1_high = R1_sbm(10.0, S=2.5, r_m=2.83e-10, tau_c_s=tau)
        # At high field, dispersion reduces R1
        self.assertGreater(float(R1_low), float(R1_high))

    def test_r1_scales_with_S(self):
        """R1 ∝ S(S+1): R1(S=3.5)/R1(S=2.5) ≈ (3.5×4.5)/(2.5×3.5)."""
        B0 = 0.5
        r_m = 3e-10
        tau = 50e-12
        R1_S25 = R1_sbm(B0, S=2.5, r_m=r_m, tau_c_s=tau)
        R1_S35 = R1_sbm(B0, S=3.5, r_m=r_m, tau_c_s=tau)
        ratio_expected = (3.5 * 4.5) / (2.5 * 3.5)
        ratio_computed = float(R1_S35) / float(R1_S25)
        self.assertAlmostEqual(ratio_computed, ratio_expected, places=6)


class TestRedfieldVsSBM(unittest.TestCase):
    """Level 7 & 8: Redfield must recover SBM when D=E=0 at high field."""

    def _run_comparison(self, S, B0_T):
        ham = SpinHamiltonian(S)
        r_m = 3.0e-10
        tau = 50e-12
        rf = RedfieldR1(ham, r_m, tau, temp_K=298.15)
        R1_redfield = rf.compute_R1(B0_T, D_MHz=0.0, E_MHz=0.0)
        R1_baseline = float(R1_sbm(B0_T, S, r_m, tau))
        return float(R1_redfield), R1_baseline

    def test_d_zero_mn_high_field(self):
        """Redfield with D=E=0 must be within 15% of SBM at B0=2T (Mn)."""
        R1_rf, R1_sbm_val = self._run_comparison(S=2.5, B0_T=2.0)
        rel_err = abs(R1_rf - R1_sbm_val) / R1_sbm_val
        self.assertLess(rel_err, 0.15,
                        f"Redfield/SBM deviation too large: {rel_err:.2%}")

    def test_d_zero_gd_high_field(self):
        """Redfield with D=E=0 must be within 15% of SBM at B0=2T (Gd)."""
        R1_rf, R1_sbm_val = self._run_comparison(S=3.5, B0_T=2.0)
        rel_err = abs(R1_rf - R1_sbm_val) / R1_sbm_val
        self.assertLess(rel_err, 0.15,
                        f"Redfield/SBM deviation too large: {rel_err:.2%}")

    def test_r1_positive_with_zfs(self):
        """R1 > 0 with non-zero ZFS at any B0."""
        ham = SpinHamiltonian(S=2.5)
        rf = RedfieldR1(ham, r_m=2.83e-10, tau_c_s=30e-12, temp_K=298.15)
        for B0 in [0.001, 0.1, 1.0, 5.0]:
            R1 = rf.compute_R1(B0, D_MHz=800.0, E_MHz=200.0)
            self.assertGreater(R1, 0.0,
                               f"R1 ≤ 0 at B0={B0}T")

    def test_zeeman_limit_benchmark_over_grid(self):
        """The independent SBM oracle is recovered over the release grid."""
        fields = np.array([0.1, 1.0, 2.0, 10.0])
        for spin in (2.5, 3.5):
            benchmark = zeeman_limit_errors(spin, fields)
            self.assertLess(float(benchmark["relative_error"].max()), 0.01)


if __name__ == '__main__':
    unittest.main()
