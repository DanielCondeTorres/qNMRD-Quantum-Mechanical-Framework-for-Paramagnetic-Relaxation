import numpy as np
import matplotlib.pyplot as plt
from qnmrd.spin.hamiltonian import SpinHamiltonian

def plot_energy_levels(S, D_MHz, E_MHz, B0_range_MHz):
    ham = SpinHamiltonian(S)
    
    levels = []
    for B0 in B0_range_MHz:
        H0 = ham.get_H0(B0, D_MHz, E_MHz)
        evals, _ = ham.exact_diagonalization(H0)
        levels.append(evals)
        
    levels = np.array(levels)
    
    plt.figure(figsize=(10, 6))
    for i in range(levels.shape[1]):
        plt.plot(B0_range_MHz, levels[:, i], label=f'Nivel {i+1}')
        
    plt.xscale('log')
    plt.xlabel('Campo Magnético Equivalente $B_0$ (MHz)')
    plt.ylabel('Energía (MHz)')
    plt.title(f'Niveles de Energía vs B0 (S={S}, D={D_MHz} MHz, E={E_MHz} MHz)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Marcar los regímenes
    plt.axvline(x=abs(D_MHz), color='red', linestyle=':', label='ZFS ~ Zeeman Crossover')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('energy_levels_plot.png')
    print("Gráfica guardada como energy_levels_plot.png")

if __name__ == "__main__":
    # Test plot: S=5/2, D=3000 MHz (~0.1 cm-1), barrido 10 a 100000 MHz
    B0_range = np.logspace(1, 5, 100)
    plot_energy_levels(S=2.5, D_MHz=3000.0, E_MHz=500.0, B0_range_MHz=B0_range)
