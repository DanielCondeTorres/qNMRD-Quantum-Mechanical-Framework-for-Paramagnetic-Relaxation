import numpy as np
import scipy.linalg as la

# --- FÍSICA CONSTANTES ---
gamma_I = 2.675e8      # rad/(s T)
gamma_S = 1.76e11      # rad/(s T)
hbar = 1.054e-34       # J s
mu0 = 4 * np.pi * 1e-7 # T m / A

# Factor de conversión de cm-1 a rad/s
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

def calc_r1_zfs(S, D_cm, E_cm, r_A, tau_c_ps, B0_array):
    Sx, Sy, Sz = get_spin_matrices(S)
    dim = int(2*S + 1)
    
    # Parámetros
    D = D_cm * cm_to_rad
    E = E_cm * cm_to_rad
    r_m = r_A * 1e-10
    tc = tau_c_ps * 1e-12
    
    # Constante de acoplamiento dipolar al cuadrado
    C_dip = (mu0 / (4*np.pi) * gamma_I * gamma_S * hbar / r_m**3)**2
    
    # Hamiltoniano ZFS (Independiente de B0)
    # H_ZFS = D * (Sz^2 - S(S+1)/3) + E * (Sx^2 - Sy^2)
    S_sq = S*(S+1)
    H_ZFS = D * (Sz @ Sz - (S_sq/3)*np.eye(dim)) + E * (Sx @ Sx - Sy @ Sy)
    
    r1_list = []
    for B0 in B0_array:
        wI = gamma_I * B0
        wS = gamma_S * B0
        
        # Hamiltoniano Total del Electrón
        H_elec = H_ZFS + wS * Sz
        
        # Diagonalizar el Hamiltoniano
        # eigenvals en rad/s
        evals, evecs = la.eigh(H_elec)
        
        # Proyectar operadores de espín en la nueva base
        Sx_eig = evecs.conj().T @ Sx @ evecs
        Sy_eig = evecs.conj().T @ Sy @ evecs
        Sz_eig = evecs.conj().T @ Sz @ evecs
        
        # Función de densidad espectral
        def J(omega):
            return 2 * tc / (1 + (omega * tc)**2)
            
        rate = 0.0
        # Sumar sobre todas las transiciones posibles |i> -> |j>
        for i in range(dim):
            for j in range(dim):
                delta_E = evals[i] - evals[j]
                
                # Frecuencias de transición dipolar (Protón voltea)
                # w_trans = delta_E + wI
                w_trans = delta_E + wI
                
                # Elementos de matriz al cuadrado
                sx_sq = np.abs(Sx_eig[i, j])**2
                sy_sq = np.abs(Sy_eig[i, j])**2
                sz_sq = np.abs(Sz_eig[i, j])**2
                
                # Contribución a la relajación (Fórmula de Redfield generalizada)
                # Simplificación de los coeficientes dipolares promediados espacialmente
                rate += (3/10) * sx_sq * J(w_trans)
                rate += (3/10) * sy_sq * J(w_trans)
                rate += (6/10) * sz_sq * J(w_trans)
                
        # Factor final
        R1 = C_dip * rate / dim  # Promedio sobre los estados poblados (alta T)
        r1_list.append(R1 / 55500.0)
        
    return r1_list

# SIMULAR PERFILES NMRD
B0_range = np.logspace(-4, 0, 50)  # De 0.0001 T a 1 T
freq_MHz = B0_range * gamma_I / (2 * np.pi * 1e6)

# Mn(II): S=5/2, ZFS muy pequeño
r1_Mn = calc_r1_zfs(S=5/2, D_cm=0.01, E_cm=0.003, r_A=2.8, tau_c_ps=30.0, B0_array=B0_range)

# Fe(III): S=5/2, ZFS grande
r1_Fe = calc_r1_zfs(S=5/2, D_cm=0.15, E_cm=0.05, r_A=2.8, tau_c_ps=30.0, B0_array=B0_range)

# Gd(III): S=7/2, ZFS intermedio
r1_Gd = calc_r1_zfs(S=7/2, D_cm=0.04, E_cm=0.01, r_A=3.1, tau_c_ps=50.0, B0_array=B0_range)

print("Comparación de Relaxividad a Bajas Frecuencias (0.01 MHz):")
print(f"Mn(II) (ZFS pequeño): {r1_Mn[0]:.2f} mM-1 s-1")
print(f"Gd(III) (ZFS medio) : {r1_Gd[0]:.2f} mM-1 s-1")
print(f"Fe(III) (ZFS grande): {r1_Fe[0]:.2f} mM-1 s-1")

print("\nComparación de Relaxividad a Altas Frecuencias (20 MHz):")
idx_20 = np.argmin(np.abs(freq_MHz - 20.0))
print(f"Mn(II): {r1_Mn[idx_20]:.2f} mM-1 s-1")
print(f"Gd(III): {r1_Gd[idx_20]:.2f} mM-1 s-1")
print(f"Fe(III): {r1_Fe[idx_20]:.2f} mM-1 s-1")

