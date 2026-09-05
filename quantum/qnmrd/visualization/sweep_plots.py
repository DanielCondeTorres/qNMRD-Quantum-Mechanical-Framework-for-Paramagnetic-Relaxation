import numpy as np
import matplotlib.pyplot as plt
import os
from qnmrd.spin.sweep import sweep_field

def generate_sweep_plots(S, D_MHz, E_MHz, case_name):
    B0_values_T = np.logspace(-4, 1, 100)
    gamma_MHz_T = -28024.95
    
    results = sweep_field(S, D_MHz, E_MHz, B0_values_T, gamma_MHz_T)
    
    B0 = [r['B0_T'] for r in results]
    evals = np.array([r['evals'] for r in results])
    
    # 1. Energy levels
    plt.figure(figsize=(8,5))
    for i in range(evals.shape[1]):
        plt.plot(B0, evals[:, i], label=f'Nivel {i}')
    plt.xscale('log')
    plt.xlabel('Campo B0 (T)')
    plt.ylabel('Energía (MHz)')
    plt.title(f'Niveles de Energía ({case_name})')
    plt.grid(alpha=0.5)
    plt.savefig(f'energy_levels_{case_name}.png')
    plt.close()
    
    # 2. Transition frequencies
    plt.figure(figsize=(8,5))
    for r in results:
        b0_val = r['B0_T']
        freqs = [t['freq_MHz'] for t in r['transitions'] if t['freq_MHz'] > 1e-3]
        if len(freqs) > 0:
            plt.scatter([b0_val]*len(freqs), freqs, s=2, color='blue', alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Campo B0 (T)')
    plt.ylabel('Frecuencias de Transición (MHz)')
    plt.title(f'Frecuencias de Transición ({case_name})')
    plt.grid(alpha=0.5)
    plt.savefig(f'transition_freqs_{case_name}.png')
    plt.close()

    # 3. Transition intensities
    plt.figure(figsize=(8,5))
    for r in results:
        b0_val = r['B0_T']
        ints = [t['int_Sx'] + t['int_Sy'] + t['int_Sz'] for t in r['transitions']]
        if len(ints) > 0:
            plt.scatter([b0_val]*len(ints), ints, s=2, color='red', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('Campo B0 (T)')
    plt.ylabel('Intensidad Total (Sx+Sy+Sz)')
    plt.title(f'Intensidades de Transición ({case_name})')
    plt.grid(alpha=0.5)
    plt.savefig(f'transition_intensities_{case_name}.png')
    plt.close()
