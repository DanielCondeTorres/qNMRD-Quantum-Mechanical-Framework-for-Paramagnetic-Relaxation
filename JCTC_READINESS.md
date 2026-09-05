# JCTC submission-readiness audit

## Current verdict

**Not ready to submit as a JCTC research article yet.** The repository has a
promising, well-scoped methodological kernel: diagonalization of a static
second-order ZFS Hamiltonian followed by a Redfield-rate calculation. Its
value must be demonstrated as a method, not inferred from a schematic plot or
from a qubit encoding. The manuscript was revised to remove two claims that
the checked source cannot support: embedded mock NMRD points are not
experimental data, and the released ``quantum`` solver is an exact NumPy
eigensolver rather than VQE hardware data.

This assessment is aligned with JCTC's stated remit for new theory, methods,
and important applications, its expectation that software papers include a
novel method and be available to the community, and its recent editorial
emphasis on reproducibility and open data. See the journal [scope](https://pubs.acs.org/journal/jctcce/),
[software editorial](https://doi.org/10.1021/acs.jctc.2c00666), and
[reproducibility editorial](https://doi.org/10.1021/acs.jctc.6c00733).

## Work completed in this audit

- Added a no-fit, machine-readable Zeeman-limit benchmark:
  `python scripts/benchmark_zeeman_limit.py`. It compares the ZFS-Redfield
  implementation with an independently implemented SBM formula at `D=E=0`.
- Added an explicit Cartesian ZFS tensor and deterministic SO(3) static
  orientation average. This prevents the previous implicit alignment of the
  ZFS principal axes with the field from being mistaken for an isotropic
  solution calculation. It is deliberately labelled as a static reference,
  not as SLE dynamics.
- Changed the plotting script so it can only label a result experimental after
  receiving a supplied CSV and an explicit primary reference. It no longer
  contains invented ``literature`` points.
- Made the LaTeX draft accurately distinguish the exact reference solver from
  a prospective VQE/VQD computation, and distinguish a Lindblad illustration
  from a parameter-free transient-ZFS model.

## Decisive scientific package

1. **State the method precisely.** Derive the Redfield tensor from the full
   electron--nucleus Hamiltonian, including the orientational average and all
   prefactors. Define the validity domain (weak coupling, Markov/secular
   approximations, static versus transient ZFS), units, and sign conventions.
   The manuscript should show an equation-to-code traceability table in the
   Supporting Information.
2. **Use real, traceable NMRD data.** Choose at least two chemically distinct
   complexes with public primary measurements across the low-field region.
   Archive the raw/digitised data, digitisation procedure if applicable,
   temperature/concentration/pH, uncertainties, and a DOI for every source.
   One system should be withheld from parameter selection.
3. **Make a fair competing-method comparison.** Compare (i) SBM, (ii) a
   conventional SLE/transient-ZFS treatment, and (iii) qNMRD with identical
   structural and dynamical inputs. Report RMSE or likelihood with confidence
   intervals, residual plots, and sensitivity/identifiability for $D$, $E$,
   $\tau_R$, $\tau_M$, $T_{1e}$, hydration number, and outer-sphere parameters.
4. **Close the physical gap.** The present kernel applies a single BPP
   rotational correlation time to a static electronic spectrum. It does not
   derive electronic relaxation from transient ZFS or feed the Lindblad rates
   into the NMRD rate. Derive or calibrate that bath from molecular motion and
   demonstrate convergence versus its correlation time and spectral model.
5. **Make the quantum angle falsifiable.** A qubit encoding alone is not a
   quantum-computing result. Either frame it as an interface/future route (the
   honest present choice) or report VQE/VQD state energies, transition matrix
   elements, shot/noise/error-mitigation details, and errors against exact
   diagonalization for every state used in $R_1$.

## Submission artifacts still required

- A tagged release, permissive license selected by the authors, repository
  URL/archival DOI, `CITATION.cff`, and pinned environments/lockfile. A GitHub
  Actions workflow now runs the full test suite and no-fit benchmark on Python
  3.10 and 3.12; it must be observed passing in the public repository.
- Supporting Information containing derivations, all inputs/outputs, an
  executable reproduction script, and data provenance.
- A complete ACS bibliography (DOIs, correct primary sources), genuine author
  affiliations/email, a Table-of-Contents graphic, and a code/data
  availability statement. The ACS author page states that a graphical summary
  is required; check the current [JCTC author instructions](https://pubs.acs.org/jctcce/pages/info-for-authors)
  immediately before submission.

## Positioning

There is now direct JCTC competition in magnetic-relaxation software:
Atkinson and Chilton's *Tau2* (JCTC 2026, DOI
[10.1021/acs.jctc.6c00890](https://doi.org/10.1021/acs.jctc.6c00890)) reports
open, optimized simulation of phonon-driven magnetic relaxation. qNMRD should
therefore claim a sharply different contribution: quantitative paramagnetic
NMRD from an exact ZFS-resolved electron--nucleus Redfield treatment, validated
against real NMRD data and conventional SLE models. That comparison, rather
than the word ``quantum``, is the publication opportunity.
