import numpy as np
import scipy.linalg as la


# Físicas
gamma_I = 2.675e8      # rad/(s T)
gamma_S = 1.76e11      # rad/(s T)
hbar = 1.054e-34       # J s
mu0 = 4 * np.pi * 1e-7 # T m / A
r = 3.1e-10            # 3.1 Angstrom
tau_c = 70e-12         # 70 ps

# Pre-factor dipolar D/hbar en rad/s
D = (mu0 / (4 * np.pi)) * (gamma_I * gamma_S * hbar) / (r**3)
# C = 3/10 * D^2
C = 0.3 * D**2

# Spin matrices para I = 1/2
Iz = np.array([[0.5, 0], [0, -0.5]])
Ip = np.array([[0, 1], [0, 0]])
Im = np.array([[0, 0], [1, 0]])

# Spin matrices para S = 7/2
S = 7/2
dimS = int(2*S + 1)
Sz = np.zeros((dimS, dimS))
Sp = np.zeros((dimS, dimS))
Sm = np.zeros((dimS, dimS))
for i in range(dimS):
    m = S - i
    Sz[i, i] = m
    if i > 0:
        Sp[i-1, i] = np.sqrt(S*(S+1) - m*(m+1))
    if i < dimS - 1:
        Sm[i+1, i] = np.sqrt(S*(S+1) - m*(m-1))

# Operadores de Kronecker para sistema 16x16
def kron(A, B):
    return np.kron(A, B)

II = np.eye(2)
IS = np.eye(dimS)

Iz_kron = kron(Iz, IS)
Ip_kron = kron(Ip, IS)
Im_kron = kron(Im, IS)

Sz_kron = kron(II, Sz)
Sp_kron = kron(II, Sp)
Sm_kron = kron(II, Sm)

# Función de densidad espectral
def J(omega, tc):
    return 2 * tc / (1 + (omega * tc)**2)

# Frecuencias de Larmor para un campo B0 (en Tesla)
def get_T1(B0):
    wI = gamma_I * B0
    wS = gamma_S * B0
    
    # Solomon equation para comparación (S=7/2)
    # 1/T1 = 2/15 * (mu0/4pi * gammaI * gammaS * hbar / r^3)^2 * S(S+1) * [3 J(wI) + 7 J(wS)]
    # Usando J(w) = tc / (1 + w^2 tc^2)
    # Nota: la definicion de J varía por un factor de 2 en la literatura. 
    # La ecuacion de Solomon clasica usa tc / (1 + w^2 tc^2).
    # Nuestro J(w) arriba tiene el 2. Ajustaremos a la ecuacion clasica.
    
    j_wI = tau_c / (1 + (wI * tau_c)**2)
    j_wS = tau_c / (1 + (wS * tau_c)**2)
    
    R1 = (2/15) * (D**2) * (S*(S+1)) * (3 * j_wI + 7 * j_wS)
    
    # Para relaxividad (r1), asumiendo q=1, y T1m = 1/R1
    # r1 = q / (55.5 * T1m) = R1 / 55.5
    r1 = R1 / 55.5
    return R1, r1

B0_vals = np.logspace(-4, 1, 50)
r1_vals = [get_T1(b)[1] for b in B0_vals]

# Datos experimentales aprox para Gd-DOTA a 37C (de literatura)
# 20 MHz (0.47 T) -> ~3.7 mM-1 s-1
# 60 MHz (1.41 T) -> ~3.2 mM-1 s-1
exp_B0 = [0.47, 1.41]
exp_r1 = [3.7, 3.2]

print(f"Predicción teórica/cuántica a 0.47 T (20 MHz): {get_T1(0.47)[1]:.2f} mM-1 s-1")
print(f"Predicción teórica/cuántica a 1.41 T (60 MHz): {get_T1(1.41)[1]:.2f} mM-1 s-1")

