"""
quantum_sle_demo.py
===================
Proof-of-concept demonstration of Open Quantum Systems simulation
for Transient Zero-Field Splitting (Stochastic Liouville Equation).

This script simulates the decoherence and relaxation of the Gd(III)
electron spin (S=7/2, mapped to 3 qubits) using the Lindblad Master
Equation. The relaxation of magnetization <S_z(t)> corresponds to
the electronic longitudinal relaxation time (T1e).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Ensure the package is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnmrd.spin.hamiltonian import SpinHamiltonian
from qnmrd.vqe.lindblad import build_liouvillian

def run_open_quantum_system_simulation():
    print("======================================================")
    print("🌌 INITIATING OPEN QUANTUM SYSTEM (LINDBLAD) SIMULATION")
    print("======================================================")

    # 1. Define the 3-qubit (S=7/2) system
    S = 3.5
    ham = SpinHamiltonian(S)
    
    # Static parameters
    B0_T = 0.5
    D_MHz = 600.0
    gamma_e = -28024.95
    
    H0 = ham.get_H0(B0_T, gamma_e, D_MHz)
    
    # 2. Define the Jump Operators (Transient ZFS fluctuations)
    # The molecular distortions act on the spin components.
    # To simulate T1e/T2e relaxation, we use the spin operators Sx, Sy, Sz
    # as the noise channels (representing isotropic fluctuating fields).
    L_x = ham.Sx
    L_y = ham.Sy
    L_z = ham.Sz
    
    jump_ops = [L_x, L_y, L_z]
    
    # Fluctuation rate (proportional to transient ZFS amplitude squared * tau_v)
    # We choose an arbitrary decay rate for demonstration (~1 ns^-1 = 1e9 s^-1 -> 1e3 MHz)
    gamma_rate = 1e3  # MHz
    rates = [gamma_rate, gamma_rate, gamma_rate]
    
    print("-> Constructing the 64x64 Liouvillian Superoperator...")
    L_super = build_liouvillian(H0, jump_ops, rates)
    
    # 3. Initial State: Fully polarized in +z direction (|m = +7/2>)
    print("-> Initializing Qubit Register in |m=+7/2> state...")
    rho_0 = np.zeros((ham.dim, ham.dim), dtype=complex)
    rho_0[0, 0] = 1.0  # Top state is m=+7/2
    
    # Vectorize the density matrix
    rho_vec_0 = rho_0.flatten()
    
    # 4. Time Evolution (solve d(rho)/dt = L * rho)
    def lindblad_deriv(t, rho_vec):
        return L_super @ rho_vec
        
    t_span = (0, 0.005)  # Time in microseconds (us) since rates are in MHz
    t_eval = np.linspace(t_span[0], t_span[1], 200)
    
    print("-> Propagating the Lindblad Master Equation...")
    sol = solve_ivp(lindblad_deriv, t_span, rho_vec_0, t_eval=t_eval, method='RK45')
    
    # 5. Calculate Observables: Magnetization <S_z(t)>
    Sz_flat = ham.Sz.T.flatten()  # To compute Tr(Sz * rho) = sum(Sz^T_flat * rho_flat)
    
    magnetization = []
    for rho_vec in sol.y.T:
        # Trace of Sz * rho
        expect_z = np.dot(Sz_flat, rho_vec)
        magnetization.append(np.real(expect_z))
        
    magnetization = np.array(magnetization)
    
    # 6. Plotting the Decay (T1e)
    plt.figure(figsize=(8, 5))
    plt.plot(t_eval * 1000, magnetization, 'b-', linewidth=3, label=r'$\langle S_z(t) \rangle$ (Simulation)')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Time (ns)', fontsize=14)
    plt.ylabel('Magnetization (a.u.)', fontsize=14)
    plt.title('Transient ZFS: Quantum Decoherence of the Gd(III) Spin', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    output_path = os.path.join("results", "figures", "quantum_t1e_decay.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Simulation Complete! Magnetization decay plot saved to: {output_path}")

if __name__ == "__main__":
    run_open_quantum_system_simulation()
