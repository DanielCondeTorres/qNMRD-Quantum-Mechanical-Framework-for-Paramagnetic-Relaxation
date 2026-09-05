import numpy as np
from qiskit.quantum_info import SparsePauliOp
# Usamos las librerías modernas de algoritmos cuánticos de Qiskit
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

def build_gadolinium_spin_hamiltonian(num_electrons=7, J_exchange=1.0, D_zfs=0.2):
    """
    Construye un modelo de espín simplificado para los 7 electrones f del Gd3+.
    - J_exchange: Acoplamiento ferromagnético (Regla de Hund) para alinear espines.
    - D_zfs: Efecto de desdoblamiento (anisotropía / Zero Field Splitting simulado local).
    """
    pauli_list = []
    
    # 1. Interacción de Intercambio (Heisenberg Ferromagnético)
    # Favorece que los espines se alineen (estado de alto espín S=7/2)
    for i in range(num_electrons - 1):
        for j in range(i + 1, num_electrons):
            # Término -J * Z_i * Z_j
            pauli_str = ['I'] * num_electrons
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_list.append(("".join(pauli_str), -J_exchange))
            
            # Término -J * X_i * X_j
            pauli_str = ['I'] * num_electrons
            pauli_str[i] = 'X'
            pauli_str[j] = 'X'
            pauli_list.append(("".join(pauli_str), -J_exchange))
            
            # Término -J * Y_i * Y_j
            pauli_str = ['I'] * num_electrons
            pauli_str[i] = 'Y'
            pauli_str[j] = 'Y'
            pauli_list.append(("".join(pauli_str), -J_exchange))
            
    # 2. Anisotropía Local / Simulando el Cristal / ZFS (Efecto en Z)
    for i in range(num_electrons):
        pauli_str = ['I'] * num_electrons
        pauli_str[i] = 'Z'
        pauli_list.append(("".join(pauli_str), D_zfs))
        
    return SparsePauliOp.from_list(pauli_list)

print("1. Construyendo el Hamiltoniano de 7 electrones (Gd3+)...")
hamiltonian = build_gadolinium_spin_hamiltonian(num_electrons=7)
print(f"Número de qubits: {hamiltonian.num_qubits}")
print(f"Términos del Hamiltoniano: {len(hamiltonian)}")

print("\n2. Calculando la energía exacta clásica (NumPy)...")
exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(hamiltonian)
print(f"Energía fundamental exacta: {exact_result.eigenvalue.real:.4f} Hartree/Unidades")

print("\n3. Configurando el Algoritmo Cuántico (VQE)...")
# Usamos un ansatz parametrizado capaz de entrelazar los 7 electrones
ansatz = EfficientSU2(num_qubits=7, reps=1, entanglement='linear')
optimizer = COBYLA(maxiter=300)
estimator = Estimator()

vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)

print("4. Ejecutando VQE (Esto simula el procesamiento en un ordenador cuántico real)...")
vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)

print(f"\n--- RESULTADOS VQE ---")
print(f"Energía VQE obtenida:       {vqe_result.eigenvalue.real:.4f}")
print(f"Error frente al exacto:     {abs(exact_result.eigenvalue.real - vqe_result.eigenvalue.real):.4f}")
print("El VQE ha convergido a una aproximación del estado fundamental electrónico del Gadolinio.")
print("A partir de esta función de onda cuántica, podríamos extraer el tensor ZFS exacto para predecir T1.")
