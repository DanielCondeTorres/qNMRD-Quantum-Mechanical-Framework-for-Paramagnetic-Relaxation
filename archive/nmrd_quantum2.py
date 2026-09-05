import numpy as np
gamma_I = 2.675e8
gamma_S = 1.76e11
hbar = 1.054e-34
mu0 = 4 * np.pi * 1e-7
r = 3.1e-10
tau_c = 110e-12
D = (mu0 / (4 * np.pi)) * (gamma_I * gamma_S * hbar) / (r**3)
S = 7/2
def get_r1(B0):
    wI = gamma_I * B0
    wS = gamma_S * B0
    j_wI = tau_c / (1 + (wI * tau_c)**2)
    j_wS = tau_c / (1 + (wS * tau_c)**2)
    R1 = (2/15) * (D**2) * (S*(S+1)) * (3 * j_wI + 7 * j_wS)
    return R1 / 55500
print(f"20 MHz (0.47 T): {get_r1(0.47):.2f}")
print(f"60 MHz (1.41 T): {get_r1(1.41):.2f}")
