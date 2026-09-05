import numpy as np
from qnmrd.spin.hamiltonian import SpinHamiltonian
from qnmrd.spin.sweep import sweep_field
from qnmrd.visualization.sweep_plots import generate_sweep_plots

cases = {
    'Case_A': {'S': 0.5, 'D': 0.0, 'E': 0.0},
    'Case_B': {'S': 2.5, 'D': 0.0, 'E': 0.0},
    'Case_C': {'S': 2.5, 'D': 100.0, 'E': 0.0},
    'Case_D': {'S': 2.5, 'D': 100.0, 'E': 20.0},
    'Case_E': {'S': 3.5, 'D': 3000.0, 'E': 500.0},
}

gamma = -28024.95 # MHz/T

for name, params in cases.items():
    print(f"================ {name} ================")
    S, D, E = params['S'], params['D'], params['E']
    ham = SpinHamiltonian(S)
    print(f"Dimensión del Hamiltoniano: {ham.dim} x {ham.dim}")
    
    H_zfs = ham.get_H0(0.0, gamma, D, E)
    print(f"Hermiticidad comprobada: {np.allclose(H_zfs, H_zfs.conj().T)}")
    print(f"Trace=0 comprobado: {np.isclose(np.trace(H_zfs), 0.0)}")
    
    # B0 = 0
    res_0 = sweep_field(S, D, E, [0.0], gamma)[0]
    print(f"Autovalores B0=0 T: {np.round(res_0['evals'], 2)}")
    
    # B0 = intermedio
    res_int = sweep_field(S, D, E, [0.1], gamma)[0]
    print(f"Autovalores B0=0.1 T: {np.round(res_int['evals'], 2)}")
    
    # B0 = alto
    res_high = sweep_field(S, D, E, [3.0], gamma)[0]
    print(f"Autovalores B0=3.0 T: {np.round(res_high['evals'], 2)}")
    
    # Frecuencias de transicion a 0.1T
    major_trans = sorted([t for t in res_int['transitions'] if t['int_Sx']+t['int_Sy']+t['int_Sz'] > 0.1], key=lambda x: -x['int_Sx'])
    print("Principales Frecuencias de Transición a 0.1T (Top 3 por intensidad Sx):")
    for t in major_trans[:3]:
        print(f"  |{t['i']}> <-> |{t['j']}> : Freq = {t['freq_MHz']:.2f} MHz (Int = Sx:{t['int_Sx']:.2f}, Sy:{t['int_Sy']:.2f}, Sz:{t['int_Sz']:.2f})")
    print("")
    
    # Visualización
    generate_sweep_plots(S, D, E, name)

