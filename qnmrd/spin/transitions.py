import numpy as np

def transition_frequencies(evals):
    """
    Devuelve lista de transiciones con freq_MHz y diff_energy_MHz.
    """
    n = len(evals)
    transitions = []
    for i in range(n):
        for j in range(i+1, n):
            diff = evals[j] - evals[i]
            freq = np.abs(diff)
            transitions.append({
                'i': i, 'j': j,
                'diff_energy_MHz': diff,
                'freq_MHz': freq
            })
    return transitions

def transition_intensities(evecs, Sx, Sy, Sz):
    """
    Calcula |<i|S_alpha|j>|^2 para las transiciones.
    """
    n = evecs.shape[1]
    Sx_eig = evecs.conj().T @ Sx @ evecs
    Sy_eig = evecs.conj().T @ Sy @ evecs
    Sz_eig = evecs.conj().T @ Sz @ evecs
    
    intensities = {}
    for i in range(n):
        for j in range(i+1, n):
            ix = np.abs(Sx_eig[i, j])**2
            iy = np.abs(Sy_eig[i, j])**2
            iz = np.abs(Sz_eig[i, j])**2
            intensities[(i, j)] = {'Sx': ix, 'Sy': iy, 'Sz': iz}
    return intensities
