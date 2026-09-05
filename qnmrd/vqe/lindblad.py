"""
qnmrd/vqe/lindblad.py
=====================
Open Quantum Systems module for transient Zero-Field Splitting (ZFS).

Constructs the Lindblad Master Equation superoperator (Liouvillian) to
simulate the dissipative dynamics of the electron spin (T1e relaxation).
This is the quantum equivalent of the classical Stochastic Liouville
Equation (SLE) for a fluctuating ZFS interaction.
"""

import numpy as np

def commutator_superop(A):
    r"""
    Constructs the superoperator for the commutator [A, rho].
    L_comm = A \otimes I - I \otimes A^T
    r"""
    dim = A.shape[0]
    I = np.eye(dim)
    return np.kron(A, I) - np.kron(I, A.T)

def anticommutator_superop(A):
    r"""
    Constructs the superoperator for the anticommutator {A, rho}.
    L_anticomm = A \otimes I + I \otimes A^T
    r"""
    dim = A.shape[0]
    I = np.eye(dim)
    return np.kron(A, I) + np.kron(I, A.T)

def dissipator_superop(L):
    r"""
    Constructs the Lindblad dissipator superoperator for a jump operator L:
    D(rho) = L rho L^dagger - 1/2 {L^dagger L, rho}
    
    Vectorized form:
    D = L \otimes L^* - 1/2 * (L^\dagger L \otimes I + I \otimes (L^\dagger L)^T)
    r"""
    dim = L.shape[0]
    I = np.eye(dim)
    
    L_dag = L.conj().T
    L_dag_L = L_dag @ L
    
    term1 = np.kron(L, L.conj())
    term2 = 0.5 * anticommutator_superop(L_dag_L)
    
    return term1 - term2

def build_liouvillian(H0, jump_operators, rates):
    """
    Builds the full Liouvillian superoperator matrix L for the Lindblad equation:
    d_rho/dt = L * rho (where rho is vectorized).
    
    Parameters
    ----------
    H0 : ndarray
        Static Hamiltonian matrix (N x N).
    jump_operators : list of ndarray
        List of jump operator matrices L_k (N x N).
    rates : list of float
        Decay rates gamma_k corresponding to each jump operator.
        
    Returns
    -------
    L_super : ndarray
        The full Liouvillian superoperator matrix (N^2 x N^2).
    """
    # Coherent evolution: -i [H, rho]
    L_super = -1j * commutator_superop(H0)
    
    # Dissipative evolution
    for L_k, gamma_k in zip(jump_operators, rates):
        L_super += gamma_k * dissipator_superop(L_k)
        
    return L_super
