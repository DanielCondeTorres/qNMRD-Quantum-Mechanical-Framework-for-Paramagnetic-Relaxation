"""Numerical validation of the ZFS-resolved kernel in the Zeeman limit.

The only analytic oracle used here is the independently implemented
Solomon--Bloembergen--Morgan (SBM) expression.  This module intentionally
does not fit any parameter: it reports the relative error obtained after
setting D = E = 0.
"""

from __future__ import annotations

import numpy as np

from qnmrd.dynamics.redfield import RedfieldR1
from qnmrd.spin.hamiltonian import SpinHamiltonian
from qnmrd.validation.sbm import R1_sbm


def zeeman_limit_errors(S, B0_T, r_m=3.0e-10, tau_c_s=50.0e-12,
                        temp_K=298.15):
    """Return ZFS-Redfield, SBM, and relative errors for a field grid.

    Parameters are deliberately generic and do not represent a fitted
    molecular system.  ``B0_T`` may be a scalar or a one-dimensional array.
    """
    fields = np.atleast_1d(np.asarray(B0_T, dtype=float))
    if np.any(fields <= 0.0):
        raise ValueError("B0_T must contain strictly positive fields.")

    redfield = RedfieldR1(SpinHamiltonian(S), r_m, tau_c_s, temp_K)
    r1_redfield = np.array([
        redfield.compute_R1(field, D_MHz=0.0, E_MHz=0.0)
        for field in fields
    ])
    r1_sbm = np.asarray(R1_sbm(fields, S, r_m, tau_c_s), dtype=float)
    rel_error = np.abs(r1_redfield - r1_sbm) / r1_sbm
    return {
        "B0_T": fields,
        "R1_redfield_s-1": r1_redfield,
        "R1_sbm_s-1": r1_sbm,
        "relative_error": rel_error,
    }
