import numpy as np
import matplotlib.pyplot as plt
from qnmrd.spin.hamiltonian import SpinHamiltonian

# Electron gyromagnetic ratio in MHz/T (|gamma_e| = 28024.95 MHz/T)
GAMMA_E_MHz_T = -28024.95


def plot_energy_levels(S, D_MHz, E_MHz, B0_values_T, gamma_MHz_T=GAMMA_E_MHz_T,
                       save_path=None, show=True):
    """
    Plot electronic energy levels vs. B0 (Tesla).

    Parameters
    ----------
    S : float
        Electronic spin quantum number.
    D_MHz : float
        Axial ZFS parameter in MHz.
    E_MHz : float
        Rhombic ZFS parameter in MHz.
    B0_values_T : array-like
        Magnetic field values in Tesla.
    gamma_MHz_T : float
        Gyromagnetic ratio in MHz/T.  Default: gamma_e = -28024.95 MHz/T.
    save_path : str or None
        If given, save the figure to this path.
    show : bool
        If True, call plt.show().
    """
    ham = SpinHamiltonian(S)
    B0 = np.asarray(B0_values_T)

    levels = []
    for b0 in B0:
        H0 = ham.get_H0(B0_T=b0, gamma_MHz_T=gamma_MHz_T,
                        D_MHz=D_MHz, E_MHz=E_MHz)
        evals, _ = ham.exact_diagonalization(H0)
        levels.append(evals)

    levels = np.array(levels)  # shape (n_B0, dim)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(levels.shape[1]):
        ax.plot(B0, levels[:, i], label=f'Level {i}')

    ax.set_xscale('log')
    ax.set_xlabel('Magnetic Field $B_0$ (T)', fontsize=13)
    ax.set_ylabel('Energy (MHz)', fontsize=13)
    ax.set_title(
        f'Energy Levels vs $B_0$  (S={S}, D={D_MHz:.1f} MHz, E={E_MHz:.1f} MHz)',
        fontsize=13
    )
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # Mark the ZFS/Zeeman crossover:  |gamma_e| * B0 = |D|
    if abs(D_MHz) > 0 and abs(gamma_MHz_T) > 0:
        B_crossover = abs(D_MHz) / abs(gamma_MHz_T)
        ax.axvline(x=B_crossover, color='red', linestyle=':',
                   label=f'|D|/|γ| ≈ {B_crossover:.4f} T')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    # Quick validation: S=5/2, D=3000 MHz (~0.1 cm⁻¹), sweep 1e-4 → 10 T
    B0_range = np.logspace(-4, 1, 120)
    plot_energy_levels(S=2.5, D_MHz=3000.0, E_MHz=500.0,
                       B0_values_T=B0_range,
                       save_path='energy_levels_plot.png', show=False)

