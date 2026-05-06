
# Hybrid Capacity–Expansion Framework (v3.3)
A unified geometric model for galactic dynamics based on ODIM, Quiet Scalar Time, and the Foundry projection–capacity identity.

Author: David E. Blackwell  
ORCID: https://orcid.org/0009-0001-8447-9113  
### Computational Benchmarks
See [COMPUTATIONAL_BENCHMARKS.md](COMPUTATIONAL_BENCHMARKS.md) for full runtime tests,
hardware history, and reproducibility notes.


---

## Overview

This repository contains the code and data for the Hybrid Capacity–Expansion Framework (v3.3).  
The framework models dark–matter phenomenology as an emergent geometric effect arising from the observer‑dependent realization of spacetime.

The core relation is:

\[
g_{00}^{(O)} = -\frac{\Pi_O}{E},
\]

where  
- **Π_O** = projection capacity (informational coupling)  
- **E** = expansion (environmental decoherence)

Dark‑matter–like behavior appears when Π_O/E enters a low‑capacity regime.  
No new particles are required.

---

## Features

- Unified geometric model for spirals and dwarf spheroidals  
- 14‑parameter hybrid capacity–expansion law  
- Milky Way + Draco joint fit using a single parameter vector  
- Full 14‑dimensional stress test  
- Reproducible rotation‑curve pipeline  

---

## Repository Structure

```
/src
    capacity.py
    expansion.py
    hybrid_model.py
    rotation_curve.py
    stress_test.py

/data
    milky_way/
    draco/

/notebooks
    MW_fit.ipynb
    Draco_fit.ipynb
    Joint_fit.ipynb
    StressTest.ipynb

/results
    figures/
    parameters/
    diagnostics/

paper/
    manuscript.tex
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR-USERNAME>/<REPO-NAME>.git
cd <REPO-NAME>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Fits

Milky Way:

```bash
python src/rotation_curve.py --galaxy MW
```

Draco:

```bash
python src/rotation_curve.py --galaxy Draco
```

Joint fit:

```bash
python src/hybrid_model.py --joint
```

Stress test:

```bash
python src/stress_test.py
```

---

## Results Summary

Milky Way:
\[
\chi^2_{\rm MW} = 56.32
\]

Draco:
\[
\chi^2_{\rm Draco} = 298.52
\]

Joint fit:
\[
\chi^2_{\rm joint} = 362.54
\]

The best‑fit lies deep in the left tail of the 14‑dimensional parameter distribution.

---

## Citation

If you use this work, please cite:

Blackwell, D. E. (2026).  
*The Hybrid Capacity–Expansion Framework for Galactic Dynamics.*  
Zenodo. DOI: (https://doi.org/10.5281/zenodo.19503559)

---

## License

MIT License.
```

---

