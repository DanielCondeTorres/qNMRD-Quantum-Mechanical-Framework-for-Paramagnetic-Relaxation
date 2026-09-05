import numpy as np

def get_spin_matrices(S):
    """
    Returns Sx, Sy, Sz matrices for a given spin S.
    Basis is |m> in descending order: |S>, |S-1>, ..., |-S>.
    """
    dim = int(2 * S + 1)
    Sz = np.zeros((dim, dim), dtype=complex)
    Sp = np.zeros((dim, dim), dtype=complex)
    Sm = np.zeros((dim, dim), dtype=complex)
    
    for j in range(dim):
        m = S - j
        Sz[j, j] = m
        if j > 0:
            # i = j - 1 => m_i = m_j + 1
            Sp[j-1, j] = np.sqrt(S*(S+1) - m*(m+1))
        if j < dim - 1:
            # i = j + 1 => m_i = m_j - 1
            Sm[j+1, j] = np.sqrt(S*(S+1) - m*(m-1))
            
    Sx = (Sp + Sm) / 2.0
    Sy = (Sp - Sm) / (2.0j)
    
    return Sx, Sy, Sz
