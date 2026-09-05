"""
qnmrd/vqe/mapping.py
====================
Maps the macroscopic Effective Spin Hamiltonian (ZFS + Zeeman) onto
a system of qubits for quantum computation.

For a spin S, there are 2S+1 states. These can be embedded into
n qubits where 2^n >= 2S+1.
For example:
  - S = 7/2 (Gd3+) has 8 states -> maps perfectly to 3 qubits (2^3=8).
  - S = 5/2 (Mn2+) has 6 states -> maps to 3 qubits (the 2 highest states
    are either projected out or heavily penalized).
"""

import numpy as np
try:
    from qiskit.quantum_info import SparsePauliOp, Operator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def spin_to_qubit_hamiltonian(spin_hamiltonian, B0_T, D_MHz, E_MHz=0.0, gamma_e_MHz_T=-28024.95):
    """
    Constructs the effective spin Hamiltonian matrix classically and then
    decomposes it exactly into a sum of Pauli strings using Qiskit's Operator.

    Parameters
    ----------
    spin_hamiltonian : SpinHamiltonian
        Instance of the classical SpinHamiltonian.
    B0_T : float
        Magnetic field in Tesla.
    D_MHz : float
        Axial ZFS parameter in MHz.
    E_MHz : float
        Rhombic ZFS parameter in MHz.
    gamma_e_MHz_T : float
        Electron gyromagnetic ratio in MHz/T.

    Returns
    -------
    pauli_op : qiskit.quantum_info.SparsePauliOp
        The Hamiltonian mapped exactly to qubits.
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit is required to map the spin Hamiltonian to qubits.")

    # 1. Get the exact classical 2S+1 x 2S+1 matrix
    H0_matrix = spin_hamiltonian.get_H0(B0_T, gamma_e_MHz_T, D_MHz, E_MHz)

    dim = spin_hamiltonian.dim
    # 2. Determine number of qubits needed: 2^n >= dim
    num_qubits = int(np.ceil(np.log2(dim)))
    target_dim = 2**num_qubits

    # 3. If the spin dimension is not a power of 2 (e.g., S=5/2 -> dim=6),
    # pad the matrix with a large penalty energy to push non-physical states out.
    if dim < target_dim:
        padded_H0 = np.zeros((target_dim, target_dim), dtype=complex)
        padded_H0[:dim, :dim] = H0_matrix
        # Apply a massive penalty to the non-physical states
        penalty = 1e6  # 1 THz penalty
        for i in range(dim, target_dim):
            padded_H0[i, i] = penalty
        H0_matrix = padded_H0

    # 4. Decompose the matrix into Pauli strings
    op = Operator(H0_matrix)
    pauli_op = SparsePauliOp.from_operator(op)

    # 5. Filter out near-zero terms for sparsity (numerical noise)
    pauli_op = pauli_op.simplify(atol=1e-8)

    return pauli_op
