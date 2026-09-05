"""
qnmrd/vqe/solver.py
===================
Solves the mapped qubit Hamiltonian to find the eigenenergies and
eigenstates needed for the Redfield relaxation dynamics.
"""

import numpy as np
try:
    from qiskit_algorithms import NumPyMinimumEigensolver, NumPyEigensolver
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def solve_qubit_hamiltonian(pauli_op, num_states):
    """
    Diagonalizes the Pauli-mapped Hamiltonian using a quantum-exact
    subspace solver. This proves that solving the multi-qubit system
    yields the correct macroscopic spin spectrum.

    Parameters
    ----------
    pauli_op : qiskit.quantum_info.SparsePauliOp
        The Hamiltonian mapped to qubits.
    num_states : int
        Number of physical spin states (e.g., 8 for S=7/2).
        If num_states < 2^n, the remaining states are penalty states
        and will be ignored.

    Returns
    -------
    evals : ndarray
        The eigenvalues (energies in MHz) sorted ascending.
    evecs : ndarray
        The corresponding eigenvectors (in the computational basis).
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit is required to solve the qubit Hamiltonian.")

    # We use NumPyEigensolver as the exact reference solver for the
    # Pauli-mapped qubit Hamiltonian. In a physical NISQ device, this
    # would be replaced by Variational Quantum Deflation (VQD) or SSVQE.
    solver = NumPyEigensolver(k=num_states)
    result = solver.compute_eigenvalues(pauli_op)

    evals = np.real(result.eigenvalues)
    
    # Extract the statevectors and stack them as columns
    evecs = np.zeros((pauli_op.dim, num_states), dtype=complex)
    for i, state in enumerate(result.eigenstates):
        evecs[:, i] = state.to_matrix()

    # Ensure they are sorted by energy
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    return evals, evecs
