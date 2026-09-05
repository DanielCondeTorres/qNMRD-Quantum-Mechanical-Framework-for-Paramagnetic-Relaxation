"""
vqe_demo.py
===========
Demonstration of how the Variational Quantum Eigensolver (VQE) integrates
into the qNMRD framework.

Scientific Objective
--------------------
In a full quantum computing pipeline, the Zero-Field Splitting (ZFS)
tensors are NOT empirical inputs. Instead, they are extracted directly
from the ab initio electronic structure of the paramagnetic complex.

Pipeline:
1. Define the molecular geometry (e.g. Gd(III) DOTA).
2. Map the active space fermionic Hamiltonian to qubits.
3. Use VQE to find the low-lying multiplet states (ground and excited).
4. Project these states onto an effective Spin Hamiltonian to extract D and E.
5. Pass D and E into the qNMRD Redfield framework to predict relaxation.

This script demonstrates Step 4 conceptually: finding the ground state of
an effective spin Hamiltonian using VQE with Qiskit.
"""

import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer.primitives import Estimator
    from scipy.optimize import minimize
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def build_spin_ansatz():
    """
    Builds a simple parameterized quantum circuit (Ansatz) for 3 qubits.
    Since S=5/2 requires 6 states, 3 qubits (8 states) are sufficient.
    """
    if not HAS_QISKIT:
        return None
        
    qc = QuantumCircuit(3)
    
    # Simple Hardware-Efficient Ansatz (Ry + CX entanglements)
    # Layer 1
    qc.ry('θ0', 0)
    qc.ry('θ1', 1)
    qc.ry('θ2', 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    # Layer 2
    qc.ry('θ3', 0)
    qc.ry('θ4', 1)
    qc.ry('θ5', 2)
    
    return qc


def demonstrate_vqe():
    if not HAS_QISKIT:
        print("Qiskit is not installed. To run the VQE demo, install it via:")
        print("  pip install qiskit qiskit-aer")
        return

    print("=== Qiskit VQE Demonstration for qNMRD ===")
    
    # 1. Define the Hamiltonian
    # For a spin S=5/2 mapped to 3 qubits, the ZFS Hamiltonian D * Sz^2 
    # translates into a linear combination of Pauli strings (Z, ZZ).
    # Here is a toy observable representing a simplified ZFS mapping:
    # H = -1.0 * Z0 Z1 + 0.5 * Z1 Z2 + 0.2 * X0
    hamiltonian = SparsePauliOp.from_list([
        ("ZZI", -1.0),
        ("IZZ", 0.5),
        ("XII", 0.2)
    ])
    print(f"Mapped Spin Hamiltonian Observables:\n{hamiltonian}\n")

    # 2. Define the Ansatz
    ansatz = build_spin_ansatz()
    print("VQE Ansatz Circuit:")
    print(ansatz.draw())
    print("\n")

    # 3. Setup VQE (Estimator + classical optimizer)
    estimator = Estimator()
    
    def cost_function(params):
        # Bind parameters to the ansatz
        bound_circ = ansatz.assign_parameters(params)
        # Evaluate expectation value <Ψ(θ)| H |Ψ(θ)>
        result = estimator.run([bound_circ], [hamiltonian]).result()
        return result.values[0]

    # 4. Optimize
    initial_params = np.random.random(ansatz.num_parameters)
    print("Running VQE Optimization...")
    res = minimize(cost_function, initial_params, method='COBYLA', options={'maxiter': 100})
    
    print("\n=== VQE Results ===")
    print(f"Ground State Energy: {res.fun:.4f} a.u.")
    print("Optimal Parameters:", np.round(res.x, 3))
    print("\nThis ground state (and similarly computed excited states)")
    print("are used to extract the Zero-Field Splitting parameters (D, E)")
    print("which are then passed to the Redfield relaxation module.")


if __name__ == "__main__":
    demonstrate_vqe()
