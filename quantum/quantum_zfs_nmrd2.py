import numpy as np
import scipy.linalg as la

gamma_I = 2.675e8
gamma_S = 1.76e11
hbar = 1.054e-34
mu0 = 4 * np.pi * 1e-7
cm_to_rad = 2 * np.pi * 3e10

def get_spin_matrices(S):
    dim = int(2*S + 1)
    Sz = np.zeros((dim, dim), dtype=complex)
    Sx = np.zeros((dim, dim), dtype=complex)
    Sy = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        m = S - i
        Sz[i, i] = m
        if i > 0:
            val = np.sqrt(S*(S+1) - m*(m+1))
            Sx[i-1, i] = val / 2.0
            Sx[i, i-1] = val / 2.0
            Sy[i-1, i] = -1j * val / 2.0
            Sy[i, i-1] = 1j * val / 2.0
    return Sx, Sy, Sz

def calc_r1(S, D_cm, r_A, tau_c_ps, B0_array):
    Sx, Sy, Sz = get_spin_matrices(S)
    dim = int(2*S + 1)
    D = D_cm * cm_to_rad
    r_m = r_A * 1e-10
    tc = tau_c_ps * 1e-12
    C_dip = (mu0 / (4*np.pi) * gamma_I * gamma_S * hbar / r_m**3)**2
    
    # H_ZFS = D * (Sz^2 - S(S+1)/3)
    S_sq = S*(S+1)
    H_ZFS = D * (Sz @ Sz - (S_sq/3)*np.eye(dim))
    
    r1_list = []
    for B0 in B0_array:
        wI = gamma_I * B0
        wS = gamma_S * B0
        H_elec = H_ZFS + wS * Sz
        evals, evecs = la.eigh(H_elec)
        
        Sx_eig = evecs.conj().T @ Sx @ evecs
        Sy_eig = evecs.conj().T @ Sy @ evecs
        Sz_eig = evecs.conj().T @ Sz @ evecs
        
        def J(omega): return 2 * tc / (1 + (omega * tc)**2)
            
        rate = 0.0
        for i in range(dim):
            for j in range(dim):
                delta_E = evals[i] - evals[j]
                w_trans = delta_E + wI
                
                rate += (3/10) * np.abs(Sx_eig[i, j])**2 * J(w_trans)
                rate += (3/10) * np.abs(Sy_eig[i, j])**2 * J(w_trans)
                rate += (6/10) * np.abs(Sz_eig[i, j])**2 * J(w_trans)
                
        R1 = C_dip * rate / dim
        r1_list.append(R1 / 55500.0)
    return r1_list

B0 = [0.0001]
r1_Mn = calc_r1(S=5/2, D_cm=0.00, r_A=2.7, tau_c_ps=30.0, B0_array=B0)[0]
r1_Gd = calc_r1(S=7/2, D_cm=0.00, r_A=3.1, tau_c_ps=30.0, B0_array=B0)[0]
print(f"Sin ZFS:")
print(f"Mn: {r1_Mn:.2f}, Gd: {r1_Gd:.2f}")
