import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator

# Constantes Físicas Universales
gamma_I = 2.675e8
gamma_S = 1.76e11
hbar = 1.054e-34
mu0 = 4 * np.pi * 1e-7

def build_electron_hamiltonian(num_electrons, J_ex=1.0):
    pauli_list = []
    for i in range(num_electrons - 1):
        for j in range(i + 1, num_electrons):
            for axis in ['X', 'Y', 'Z']:
                p_str = ['I'] * num_electrons
                p_str[i] = axis
                p_str[j] = axis
                pauli_list.append(("".join(p_str), -J_ex))
    for i in range(num_electrons):
        p_str = ['I'] * num_electrons
        p_str[i] = 'Z'
        pauli_list.append(("".join(p_str), 0.1))
    return SparsePauliOp.from_list(pauli_list)

def run_pipeline(element_name, num_electrons, S, r_angstrom, tau_c_ps):
    print(f"=======================================")
    print(f"🔬 INICIANDO PIPELINE PARA: {element_name} (Espín {S})")
    print(f"=======================================")
    
    # FASE 1: VQE (Estructura Electrónica)
    hamiltonian = build_electron_hamiltonian(num_electrons)
    ansatz = EfficientSU2(num_qubits=num_electrons, reps=1, entanglement='linear')
    optimizer = COBYLA(maxiter=100)
    estimator = StatevectorEstimator()
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    
    print(f"[VQE] Buscando estado fundamental cuántico de {num_electrons} electrones...")
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
    print(f"[VQE] Energía del estado fundamental: {energy:.4f} Hartree")
    
    # FASE 2: EXTRACCIÓN AL MODELO DE RELAJACIÓN
    # La fuerza dipolar depende de la distancia y las constantes universales
    r_m = r_angstrom * 1e-10
    tc_s = tau_c_ps * 1e-12
    D_dipolar = (mu0 / (4 * np.pi)) * (gamma_I * gamma_S * hbar) / (r_m**3)
    
    # FASE 3: PREDICCIÓN CLINICA NMRD (Lindblad/Solomon)
    def calc_r1(freq_MHz):
        B0 = (freq_MHz * 2 * np.pi * 1e6) / gamma_I
        wI = gamma_I * B0
        wS = gamma_S * B0
        j_wI = tc_s / (1 + (wI * tc_s)**2)
        j_wS = tc_s / (1 + (wS * tc_s)**2)
        # Tasa de relajación usando la ecuación maestra
        R1 = (2/15) * (D_dipolar**2) * (S*(S+1)) * (3 * j_wI + 7 * j_wS)
        return R1 / 55500.0  # Asumiendo 1 molécula de agua (q=1)
    
    r1_20 = calc_r1(20.0)
    r1_60 = calc_r1(60.0)
    print(f"[NMRD] Predicción de Relaxividad en Hospital:")
    print(f"       -> A 20 MHz (0.47 T): {r1_20:.2f} mM⁻¹ s⁻¹")
    print(f"       -> A 60 MHz (1.41 T): {r1_60:.2f} mM⁻¹ s⁻¹\n")

# Ejecutar pruebas
# 1. Gadolinio (Gd3+) - Rey de los contrastes
run_pipeline("Gadolinio (Gd3+)", num_electrons=7, S=7/2, r_angstrom=3.1, tau_c_ps=110.0)

# 2. Manganeso (Mn2+) - Alternativa menos tóxica (ej. Teslascan)
# Suelen tener distancias un poco más cortas (2.8 A) y tau_c de ~50 ps
run_pipeline("Manganeso (Mn2+)", num_electrons=5, S=5/2, r_angstrom=2.8, tau_c_ps=50.0)

# 3. Cobre (Cu2+) - Parámagnetico débil
run_pipeline("Cobre (Cu2+)", num_electrons=1, S=1/2, r_angstrom=2.5, tau_c_ps=30.0)

