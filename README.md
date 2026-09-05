# qNMRD: Quantum-Mechanical Framework for Paramagnetic Relaxation Dispersion

**A generalized Redfield approach to predict Nuclear Magnetic Relaxation Dispersion (NMRD) profiles for high-spin transition metal and lanthanide complexes (e.g., Mn(II), Gd(III)) incorporating Zero-Field Splitting (ZFS).**

This repository is designed to bridge the gap between microscopic quantum chemistry (and future quantum computing architectures) and macroscopic NMR relaxation observables. It provides a physically rigorous alternative to the phenomenological Solomon–Bloembergen–Morgan (SBM) theory, particularly in the low magnetic field regime where the Zeeman limit breaks down.

---

## 🔬 Scientific Context & Motivation

In paramagnetic NMR and MRI contrast agent design, the relaxivity (efficiency of a paramagnetic ion to enhance the relaxation of solvent nuclear spins) is typically analyzed using the **Solomon–Bloembergen–Morgan (SBM) theory**. 

However, SBM makes a critical assumption: it treats the electronic spin structure strictly in the **Zeeman limit**, assuming the 2S+1 electronic energy levels are uniformly spaced ($E_m = \gamma_e B_0 m_s$). This assumption fails dramatically at low magnetic fields (e.g., Fast Field-Cycling NMR) for ions like **Gd(III) (S=7/2)** and **Mn(II) (S=5/2)**, where the **Zero-Field Splitting (ZFS)** energy is comparable to or larger than the Zeeman interaction.

When ZFS is significant ($D, E \sim \text{GHz}$), the electronic states are no longer pure $|m_s\rangle$ states but complex superpositions. The transition probabilities between these states (which drive the relaxation of the nucleus) are drastically altered, leading to a quenching of the low-field relaxivity.

### The Solution: ZFS-Resolved Redfield Theory
This framework directly constructs the total electronic Hamiltonian ($H_0 = H_\text{Zeeman} + H_\text{ZFS}$) and performs exact diagonalization. The nuclear longitudinal relaxation rate ($R_1$) is then computed directly from the secular Redfield master equation, evaluated over the exact electronic eigenbasis.

This allows us to predict the low-field plateau and the transition into the high-field SBM limit seamlessly, without arbitrarily modifying the spectral density functions.

---

## 📊 Understanding the Output (NMRD Profiles)

When you run the framework, it generates NMRD profiles (Nuclear $R_1$ vs Proton Larmor Frequency). 

Here is how to interpret the graphs:

* **X-Axis (Proton Larmor Frequency):** Represents the external magnetic field strength ($B_0$), sweeping from ultra-low fields (0.01 MHz) to standard high-resolution MRI fields (100 MHz).
* **Y-Axis (Relaxation Rate $R_1$):** The enhancement of the nuclear relaxation rate (in $\text{mM}^{-1}\text{s}^{-1}$ or $\text{s}^{-1}$). Higher is better for contrast agents.
* **Redfield Line (ZFS-Resolved):** This is the exact quantum-mechanical calculation. You will notice that at low frequencies (left side of the graph), this curve is significantly lower than the SBM curve. This reflects the physical *quenching* of electronic transitions due to the ZFS barrier.
* **SBM Baseline (Dashed):** This is the classical Solomon equation. It dramatically overestimates the low-field relaxation because it ignores ZFS.
* **High-Field Convergence:** As the frequency increases (moving right), the Zeeman energy overpowers the ZFS. The electronic states become pure Zeeman states, and the Redfield curve beautifully converges with the SBM baseline, validating the model.

---

## 🖥️ Quantum Computing Pipeline (VQE)

*Where does Quantum Computing fit into this?*

Currently, ZFS parameters ($D$ and $E$) are often treated as empirical fitting parameters. In a fully *ab initio* predictive pipeline, these parameters must be derived from the electronic structure of the molecule.

Calculating the electronic structure of heavy, highly-correlated systems like Gd-DOTA requires multireference methods (e.g., CASSCF/NEVPT2), which scale exponentially on classical computers.

We propose a quantum-computational bridge using the **Variational Quantum Eigensolver (VQE)**:
1. The active-space molecular Hamiltonian is mapped to qubits.
2. VQE computes the ground and low-lying excited states of the spin multiplet.
3. These quantum states are projected onto an effective spin Hamiltonian to extract the tensors $D$ and $E$.
4. $D$ and $E$ are fed into this **qNMRD** Redfield module to predict the experimental NMRD profile.

> A conceptual Qiskit implementation for mapping the effective spin Hamiltonian to a quantum circuit is provided in `scripts/vqe_demo.py`.

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.9+
* `numpy`, `scipy`, `matplotlib`, `pytest`
* (Optional) `qiskit` for the VQE demonstration.

### Running the NMRD Calculations
Execute the main script to generate the NMRD profiles for Mn(II) and Gd(III):
```bash
python scripts/run_nmrd.py
```
This will output the profiles to the `results/figures/` directory and the raw numerical data to `results/data/`.

### Running the VQE Demonstration
```bash
python scripts/vqe_demo.py
```

### Running the Test Suite
The project utilizes `pytest` to validate the physics rigorously (e.g., verifying that the Redfield implementation algebraically reduces exactly to the Solomon limits when $D=E=0$).
```bash
pytest tests/ -v
```

---

## 📖 Key References

1. **Platas-Iglesias, C.** et al., *J. Phys. Chem. A* 2016, 120, 6467. (Standard reference for the breakdown of SBM and integration of ZFS in paramagnetic NMR).
2. **Kowalewski, J.; Mäler, L.** *Nuclear Spin Relaxation in Liquids*, CRC Press, 2006. (Redfield theory formulation).
3. **Cerezo, M.** et al., *Nat. Rev. Phys.* 2021, 3, 625. (Review of VQE and its application to electronic structure).
4. **Qiskit Documentation** (IBM Quantum) - For quantum circuit implementations.

---
*This framework is being prepared for submission to the Journal of Chemical Theory and Computation (JCTC).*
