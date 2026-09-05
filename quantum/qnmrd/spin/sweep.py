import numpy as np
from .hamiltonian import SpinHamiltonian
from .transitions import transition_frequencies, transition_intensities

def sweep_field(S, D_MHz, E_MHz, B0_values_T, gamma_MHz_T):
    ham = SpinHamiltonian(S)
    results = []
    for B0 in B0_values_T:
        H0 = ham.get_H0(B0, gamma_MHz_T, D_MHz, E_MHz)
        evals, evecs = ham.exact_diagonalization(H0)
        freqs = transition_frequencies(evals)
        intensities = transition_intensities(evecs, ham.Sx, ham.Sy, ham.Sz)
        
        trans_list = []
        for t in freqs:
            i, j = t['i'], t['j']
            ints = intensities[(i, j)]
            trans_list.append({
                'i': i, 'j': j,
                'freq_MHz': t['freq_MHz'],
                'int_Sx': ints['Sx'],
                'int_Sy': ints['Sy'],
                'int_Sz': ints['Sz']
            })
            
        results.append({
            'B0_T': B0,
            'evals': evals,
            'evecs': evecs,
            'transitions': trans_list
        })
    return results
