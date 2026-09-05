"""Orientation handling for static ZFS tensors.

The original ``D``/``E`` Hamiltonian is defined in a molecular principal-axis
system.  This module rotates that tensor into the laboratory frame and supplies
a deterministic quasi-uniform SO(3) grid for static-orientation averaging.

This is a *static ensemble* average.  It must not be confused with the
Stochastic Liouville Equation (SLE), which additionally propagates rotational
and transient-ZFS dynamics.  Keeping the distinction explicit prevents a
powder average from being misreported as a solution-state slow-motion model.
"""

from __future__ import annotations

import numpy as np

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def rotation_matrix_zyz(alpha, beta, gamma):
    """Return an active Z-Y-Z Euler rotation matrix."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rz_alpha = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry_beta = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz_gamma = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
    return rz_alpha @ ry_beta @ rz_gamma


def fibonacci_orientations(n_orientations):
    """Generate quasi-uniform orientations and equal weights on SO(3).

    A Fibonacci grid distributes the molecular ``z`` axis over the sphere;
    the third Euler angle is advanced with a second irrational rotation.  The
    rule is deterministic, making convergence studies reproducible.
    """
    if int(n_orientations) != n_orientations or n_orientations < 1:
        raise ValueError("n_orientations must be a positive integer.")
    count = int(n_orientations)
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    beta = np.arccos(np.clip(z, -1.0, 1.0))
    alpha = np.mod(index * _GOLDEN_ANGLE, 2.0 * np.pi)
    gamma = np.mod(index * np.pi * np.sqrt(2.0), 2.0 * np.pi)
    return [rotation_matrix_zyz(a, b, g) for a, b, g in zip(alpha, beta, gamma)]


def rotate_zfs_tensor(D_MHz, E_MHz, rotation):
    """Rotate a principal-frame ZFS tensor into the laboratory frame."""
    principal = np.diag([
        E_MHz - D_MHz / 3.0,
        -E_MHz - D_MHz / 3.0,
        2.0 * D_MHz / 3.0,
    ])
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10):
        raise ValueError("rotation must be an orthogonal 3-by-3 matrix.")
    return matrix @ principal @ matrix.T
