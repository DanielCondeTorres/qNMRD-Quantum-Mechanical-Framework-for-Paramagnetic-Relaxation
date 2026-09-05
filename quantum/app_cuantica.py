import streamlit as st
import numpy as np
import scipy.linalg as la
import plotly.graph_objects as go

st.set_page_config(page_title="Simulador Cuántico NMRD", page_icon="⚛️", layout="wide")
st.title("⚛️ Simulador Ab-Initio NMRD (Redfield ZFS)")

st.markdown("""
Esta herramienta computacional prescinde de la teoría clásica SBM. Resuelve numéricamente la **Ecuación Maestra de Redfield** para la matriz de densidad de un sistema de espín acoplado electrón-núcleo, incorporando el **Zero-Field Splitting (ZFS)** exacto. 
Esto permite predecir el comportamiento real a campos bajos, como la anomalía del Mn(II) frente al Gd(III).
""")

# --- FÍSICA CONSTANTES ---
gamma_I = 2.675e8      # rad/(s T)
gamma_S = 1.76e11      # rad/(s T)
hbar = 1.054e-34       # J s
mu0 = 4 * np.pi * 1e-7 # T m / A
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

@st.cache_data
def calc_nmrd_sweep(S, D_cm, r_A, tau_c_ps, freqs_MHz):
    Sx, Sy, Sz = get_spin_matrices(S)
    dim = int(2*S + 1)
    D = D_cm * cm_to_rad
    r_m = r_A * 1e-10
    tc = tau_c_ps * 1e-12
    C_dip = (mu0 / (4*np.pi) * gamma_I * gamma_S * hbar / r_m**3)**2
    
    S_sq = S*(S+1)
    H_ZFS = D * (Sz @ Sz - (S_sq/3)*np.eye(dim))
    
    B0_array = (freqs_MHz * 2 * np.pi * 1e6) / gamma_I
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

col1, col2, col3 = st.columns(3)
with col1:
    ion = st.selectbox("Ion Metálico", ["Mn(II) - Espín 5/2", "Gd(III) - Espín 7/2", "Fe(III) - Espín 5/2"])
with col2:
    if "Mn" in ion:
        default_D, default_r, default_tc = 0.005, 2.8, 200.0
    elif "Gd" in ion:
        default_D, default_r, default_tc = 0.030, 3.1, 200.0
    else:
        default_D, default_r, default_tc = 0.060, 2.8, 200.0
        
    D_cm = st.slider("Zero-Field Splitting D (cm⁻¹)", 0.0, 0.15, default_D, 0.005, format="%.3f")
with col3:
    r_A = st.slider("Distancia r (Å)", 2.0, 4.0, default_r, 0.1)
    tau_c = st.slider("Tiempo de correlación τ_c (ps)", 10.0, 500.0, default_tc, 10.0)

# Barrido de frecuencias
freqs_MHz = np.logspace(-2, 3, 50) # 0.01 a 1000 MHz
S_val = 7/2 if "Gd" in ion else 5/2

with st.spinner("Diagonalizando matrices cuánticas y resolviendo Redfield..."):
    r1_curve = calc_nmrd_sweep(S_val, D_cm, r_A, tau_c, freqs_MHz)

fig = go.Figure()
color_map = {"Mn": "black", "Gd": "blue", "Fe": "red"}
line_color = color_map["Mn"] if "Mn" in ion else (color_map["Gd"] if "Gd" in ion else color_map["Fe"])

fig.add_trace(go.Scatter(
    x=freqs_MHz, y=r1_curve, 
    mode='lines', 
    name=f"{ion.split()[0]} Simulación", 
    line=dict(color=line_color, width=4)
))

fig.update_layout(
    height=600,
    plot_bgcolor='white', paper_bgcolor='white',
    xaxis_type="log",
    xaxis_title="<b>Frecuencia de Larmor de Protón (MHz)</b>",
    yaxis_title="<b>Relaxividad r₁ (mM⁻¹ s⁻¹)</b>",
    xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgray', range=[-2, 3]),
    yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgray', range=[0, 16]),
    title=f"Barrido de Perfil NMRD para {ion}"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**¿Para qué usamos el VQE en un artículo real?** 
En esta web, tú ajustas el parámetro **$D$ (ZFS)** con una barra deslizante (arriba en el centro). En la vida real, ese valor de $D$ es desconocido para moléculas nuevas. Ahí es donde entra el algoritmo **VQE**. Usaríamos el VQE para calcular la estructura de los electrones de la molécula desde cero y obtener matemáticamente el valor de $D$. Una vez obtenido gracias al VQE, lo introducimos en este simulador de Redfield para imprimir la curva final.
""")
