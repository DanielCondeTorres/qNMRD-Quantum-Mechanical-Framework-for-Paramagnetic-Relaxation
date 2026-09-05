import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import curve_fit

def simulate_T1(r_dist, tau_c):
    C_dipolar = 100.0  
    J = C_dipolar / (r_dist**3)

    t_max = 50.0
    num_steps = 100
    t_values = np.linspace(0, t_max, num_steps)
    dt = t_max / (num_steps - 1)

    simulator = AerSimulator(method='density_matrix')
    noise_model = NoiseModel()
    
    p_error = 1.0 - np.exp(-dt / tau_c)
    error_channel = depolarizing_error(p_error, 1)
    noise_model.add_all_qubit_quantum_error(error_channel, ['id'])

    qc = QuantumCircuit(2)
    qc.x(1) 
    qc.save_density_matrix(label='rho_0')

    for i in range(1, num_steps):
        qc.rxx(2 * J * dt, 0, 1)
        qc.ryy(2 * J * dt, 0, 1)
        qc.id(0)
        qc.id(1)
        qc.save_density_matrix(label=f'rho_{i}')

    result = simulator.run(qc, noise_model=noise_model).result()

    prob_excited = np.zeros(num_steps)
    for i in range(num_steps):
        rho = result.data()[f'rho_{i}']
        prob = np.real(rho.data[1, 1] + rho.data[3, 3])
        prob_excited[i] = prob

    def exp_saturation(t, a, t1):
        return a * (1 - np.exp(-t / t1))

    try:
        popt, _ = curve_fit(exp_saturation, t_values, prob_excited, p0=[max(prob_excited), tau_c])
        T1_estimado = popt[1]
        return T1_estimado
    except:
        return None

if __name__ == "__main__":
    print("Prueba de Escala de Distancia (r):")
    print("Manteniendo tau_c = 20.0 ps")
    tau = 20.0
    t1_r3 = simulate_T1(3.0, tau)
    t1_r35 = simulate_T1(3.5, tau)
    t1_r4 = simulate_T1(4.0, tau)
    print(f"r = 3.0 A -> T1 = {t1_r3:.2f} ps")
    print(f"r = 3.5 A -> T1 = {t1_r35:.2f} ps")
    print(f"r = 4.0 A -> T1 = {t1_r4:.2f} ps")
    
    if t1_r3 and t1_r35 and t1_r4:
        # Teoría: T1 proporcional a r^6
        # Si t1(r) = k * r^6, entonces t1(4.0)/t1(3.0) deberia ser (4/3)^6 = 5.61
        print(f"Razón T1(4.0)/T1(3.0): {t1_r4/t1_r3:.2f} (Teórico r^6: {(4.0/3.0)**6:.2f})")
        print(f"Razón T1(3.5)/T1(3.0): {t1_r35/t1_r3:.2f} (Teórico r^6: {(3.5/3.0)**6:.2f})")

    print("\nPrueba de Escala de Tiempo de Correlación (tau_c):")
    print("Manteniendo r = 3.0 A")
    r_dist = 3.0
    t1_tau10 = simulate_T1(r_dist, 10.0)
    t1_tau20 = simulate_T1(r_dist, 20.0)
    t1_tau40 = simulate_T1(r_dist, 40.0)
    print(f"tau_c = 10.0 ps -> T1 = {t1_tau10:.2f} ps")
    print(f"tau_c = 20.0 ps -> T1 = {t1_tau20:.2f} ps")
    print(f"tau_c = 40.0 ps -> T1 = {t1_tau40:.2f} ps")
    
    if t1_tau10 and t1_tau20 and t1_tau40:
        # Teoría (límite de estrechamiento de movimiento): R1 = 1/T1 prop a tau_c -> T1 prop a 1/tau_c
        # T1(tau) prop 1/tau
        print(f"Razón T1(tau=20)/T1(tau=10): {t1_tau20/t1_tau10:.2f} (Teórico 1/x: 0.5)")
        print(f"Razón T1(tau=40)/T1(tau=20): {t1_tau40/t1_tau20:.2f} (Teórico 1/x: 0.5)")
