import numpy as np
from .operators import get_spin_matrices

class SpinHamiltonian:
    """
    Electronic Spin Hamiltonian.
    
    Unidades:
    - Campo Magnético B0: Tesla (T)
    - Gyromagnetic ratio gamma: MHz / T
    - Parámetros ZFS (D, E): MHz
    - Hamiltoniano devuelto: Energía en MHz
    
    Convención de signos:
    H_Z = gamma * B0_z * Sz
    Para un electrón libre, gamma_e es negativa (aprox -28024.95 MHz/T). 
    Por tanto, si pasas un gamma negativo, el Hamiltoniano tendrá el signo físicamente correcto, 
    colocando los autoestados de m=S en la menor energía (mayor alineación antiparalela al campo).
    """
    def __init__(self, S):
        self.S = S
        self.Sx, self.Sy, self.Sz = get_spin_matrices(S)
        self.dim = int(2 * S + 1)
        self.S_sq = self.Sx @ self.Sx + self.Sy @ self.Sy + self.Sz @ self.Sz
        
    def zeeman(self, B0_T, gamma_MHz_T):
        return gamma_MHz_T * B0_T * self.Sz

    def zfs_second_order(self, D_MHz, E_MHz=0.0):
        S_factor = self.S * (self.S + 1)
        term_D = D_MHz * (self.Sz @ self.Sz - (S_factor / 3.0) * np.eye(self.dim))
        term_E = E_MHz * (self.Sx @ self.Sx - self.Sy @ self.Sy)
        return term_D + term_E

    def get_H0(self, B0_T, gamma_MHz_T, D_MHz, E_MHz=0.0):
        return self.zeeman(B0_T, gamma_MHz_T) + self.zfs_second_order(D_MHz, E_MHz)

    def exact_diagonalization(self, H0):
        evals, evecs = np.linalg.eigh(H0)
        idx = np.argsort(evals)
        return evals[idx], evecs[:, idx]
