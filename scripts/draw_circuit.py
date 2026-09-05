"""
draw_circuit.py
===============
Draws the Quantum Circuit (Ansatz) used for the Variational Quantum Eigensolver
(VQE) to diagonalize the 3-qubit mapped spin Hamiltonian.
"""

import os
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
import matplotlib.pyplot as plt

def generate_circuit_png():
    print("Generating Quantum Circuit PNG...")
    
    # S=7/2 has 8 states -> maps to exactly 3 qubits
    num_qubits = 3
    
    # Hardware Efficient Ansatz (Ry rotations + CX entanglement)
    # This is a standard ansatz for near-term quantum devices (NISQ).
    ansatz = efficient_su2(num_qubits, su2_gates=['ry', 'rz'], entanglement='linear', reps=1)
    
    # Draw the circuit using matplotlib
    fig = ansatz.draw('mpl', style='iqp')
    
    # Save the figure
    output_path = os.path.join("results", "figures", "vqe_ansatz_circuit.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Circuit saved to: {output_path}")

if __name__ == "__main__":
    generate_circuit_png()
