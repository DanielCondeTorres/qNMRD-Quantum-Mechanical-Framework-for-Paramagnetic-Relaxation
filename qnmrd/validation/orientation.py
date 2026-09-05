"""Static-orientation reference calculations for the ZFS-Redfield kernel."""

from __future__ import annotations

import numpy as np

from qnmrd.spin.orientations import fibonacci_orientations, rotate_zfs_tensor


def static_orientation_average(redfield, B0_T, D_MHz, E_MHz=0.0,
                               gamma_e_MHz_T=-28024.95,
                               n_orientations=64):
    """Return an equal-weight static SO(3) average of ``R1``.

    The calculation averages the molecular ZFS principal axes over a
    quasi-uniform orientation grid.  It does not model time-dependent
    rotational diffusion or transient ZFS and is therefore a controlled
    reference for orientational sensitivity, not an SLE replacement.
    """
    rates = []
    for rotation in fibonacci_orientations(n_orientations):
        tensor = rotate_zfs_tensor(D_MHz, E_MHz, rotation)
        rates.append(redfield.compute_R1(
            B0_T, D_MHz, E_MHz, gamma_e_MHz_T,
            zfs_tensor_MHz=tensor,
        ))
    return float(np.mean(rates))
