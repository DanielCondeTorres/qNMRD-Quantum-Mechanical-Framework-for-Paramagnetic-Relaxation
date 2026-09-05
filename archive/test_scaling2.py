import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import curve_fit

def simulate_T1(r_dist, tau_c, t_max=100.0):
    C_dipolar = 50.0  
    J = C_dipolar / (r_dist**3)

    num_steps = 100
    t_values = np.linspace(0, t_max, num_steps)
    dt = t_max / (num_steps - 1)

    simulator = AerSimulator(method='density_matrix')
    noise_model = NoiseModel()
    
    # Error aplicado SOLO al electrón (qubit 1)
    p_error = 1.0 - np.exp(-dt / tau_c)
    error_channel = depolarizing_error(p_error, 1)
    noise_model.add_quantum_error(error_channel, ['id'], [1])

    qc = QuantumCircuit(2)
    qc.x(1) 
    qc.save_density_matrix(label='rho_0')

    for i in range(1, num_steps):
        qc.rxx(2 * J * dt, 0, 1)
        qc.ryy(2 * J * dt, 0, 1)
        qc.id(1)  # Solo ponemos id en el electrón para que actúe el ruido
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
        # A veces 'a' es pequeño (no llega a 0.5 si la relajación es muy lenta)
        # Vamos a fijar p0 para ayudar al ajuste
        popt, _ = curve_fit(exp_saturation, t_values, prob_excited, p0=[0.5, 50.0], bounds=([0.0, 0.1], [1.0, 10000.0]))
        return popt[1]
    except:
        return None

print("Prueba de Escala de Distancia (r):")
r_vals = [2.5, 3.0, 3.5]
t1_vals = []
for r in r_vals:
    t1 = simulate_T1(r, 20.0, t_max=500.0)
    print(f"r = {r} -> T1 = {t1:.2f}")
    t1_vals.append(t1)

print(f"Ratio T1(3.0)/T1(2.5) = {t1_vals[1]/t1_vals[0]:.2f} (Teórico (3.0/2.5)^6 = {(3.0/2.5)**6:.2f})")
print(f"Ratio T1(3.5)/T1(3.0) = {t1_vals[2]/t1_vals[1]:.2f} (Teórico (3.5/3.0)^6 = {(3.5/3.0)**6:.2f})")

print("\nPrueba de Escala de tau_c:")
tau_vals = [10.0, 20.0, 40.0]
t1_tau = []
for tau in tau_vals:
    t1 = simulate_T1(3.0, tau, t_max=500.0)
    print(f"tau = {tau} -> T1 = {t1:.2f}")
    t1_tau.append(t1)

print(f"Ratio T1(tau=20)/T1(tau=10) = {t1_tau[1]/t1_tau[0]:.2f} (Teórico = 0.5)")

