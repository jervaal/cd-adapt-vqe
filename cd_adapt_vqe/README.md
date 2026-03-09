# CD-ADAPT-VQE

**Simulation code for the Counterdiabatic ADAPT-VQE algorithm.**

This repository contains the implementation used in:

> **Counterdiabatic ADAPT-VQE**  
> [arXiv:2601.05973](https://arxiv.org/abs/2601.05973)

---

## Overview

CD-ADAPT-VQE is a hybrid quantum-classical algorithm for computing molecular ground state energies on near-term quantum hardware. It combines two key ideas:

1. **Counterdiabatic Driving**: The variational operator pool is derived from the *Approximate Gauge Potential* (AGP) — the counterdiabatic correction that suppresses non-adiabatic transitions along the adiabatic path from a simple initial Hamiltonian H_i to the target molecular Hamiltonian H_f.

2. **ADAPT-VQE**: Operators are selected *adaptively* from the pool by measuring energy gradients, building a compact, problem-tailored ansatz one operator at a time.

The result is an ansatz that is physically motivated by the adiabatic path of the molecule, leading to improved convergence compared to standard ADAPT-VQE for strongly-correlated systems.

---

## Algorithm

```
Given: H_f (molecular Hamiltonian), H_i (initial Ising Hamiltonian)
       |ψ₀⟩ = Hartree-Fock state

1. POOL GENERATION (AGP-based):
   For each time point t ∈ {t₁, t₂, ...}:
     Compute H_ad(t) = (1−λ(t))·H_i + λ(t)·H_f
     For k = 1, ..., l:
       O_k = [H_ad, [H_ad, ...[H_ad, dH]...]]   (2k−1 commutators)
       Extract Pauli strings from O_k → add to pool

2. ADAPT LOOP:
   While ||∇E|| > threshold:
     a. For each pool operator G_k:
           gradient_k = |⟨ψ| [H_f, G_k] |ψ⟩|
     b. Select G* = argmax_k(gradient_k)
     c. Append G* to ansatz: |ψ(θ)⟩ = exp(−iθ·G*)|ψ_prev⟩
     d. Re-optimize all θ via VQE (L-BFGS-B)
```

---

## Repository Structure

```
cd-adapt-vqe/
├── cd_adapt_vqe/               # Main Python package
│   ├── __init__.py             # Public API
│   ├── solver.py               # CD_ADAPT_Solver class (main entry point)
│   ├── agp_pool.py             # AGP operator pool generation
│   ├── schedule.py             # Adiabatic scheduling functions λ(t)
│   └── utils.py                # Nested commutators, Frobenius norms
│
├── examples/
│   └── dissociation_curve.py   # Parallel dissociation curve calculation
│
├── data/                       # Pre-computed results from the paper
│   ├── LiH/                    # LiH results (l=1,2; t=0.25, 0.75)
│   ├── BeH2/                   # BeH₂ results
│   └── HF/                     # HF molecule results
│
├── comparisons/                # Reference algorithm implementations
│   ├── adapt_vqe/              # Standard ADAPT-VQE (Qiskit) for comparison
│   └── counterdiabatic/        # Pure counterdiabatic driving (QuTiP)
│
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

A clean virtual environment using [Anaconda](https://www.anaconda.com/) is recommended. Python >= 3.10 is required.

```bash
# 1. Create and activate environment
conda create -n cd_adapt python=3.10
conda activate cd_adapt

# 2. Clone the repository
git clone https://github.com/jervaal/cd-adapt-vqe.git
cd cd-adapt-vqe

# 3. Install the package and all dependencies
pip install .

# 4. (Optional) Install comparison algorithm dependencies
pip install ".[comparisons]"
```

> **Note for Windows users**: PySCF does not support Windows natively.  
> Use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu, then install Anaconda inside WSL.

---

## Quick Start

### Single-point energy calculation (LiH at 1.5 Å)

```python
from cd_adapt_vqe import CD_ADAPT_Solver

# 1. Set up the molecular system
solver = CD_ADAPT_Solver(
    molecule_geometry="Li 0.0 0.0 0.0; H 0.0 0.0 1.5",
    basis="sto-3g",
    active_space={"num_particles": 4, "num_spatial_orbitals": 5},
)

# 2. Generate the AGP operator pool (l=1, t=0.25)
pool = solver.compute_agp_pool(l_order=1, time_points=[0.25])
print(f"Pool size: {len(pool)} operators")

# 3. Run CD-ADAPT-VQE
result = solver.run_cd_adapt(pool, max_iterations=30, gradient_threshold=1e-2)

# 4. Compare with exact energy
print(f"CD-ADAPT energy : {result['energies'][-1]:.6f} Ha")
print(f"Exact energy    : {solver.exact_energy():.6f} Ha")
print(f"Ansatz size     : {result['num_ops']} operators")
```

### Dissociation curve (parallelized, as used in the paper)

```bash
# LiH dissociation curve, first-order AGP, t=0.25
python examples/dissociation_curve.py --molecule LiH --l_order 1 --time_point 0.25

# HF molecule, second-order AGP, t=0.75
python examples/dissociation_curve.py --molecule HF --l_order 2 --time_point 0.75

# BeH2, all defaults
python examples/dissociation_curve.py --molecule BeH2
```

---

## Test Systems

All results from the paper are provided in the `data/` directory as Excel files.

| Molecule | Active Space | Qubits | AGP Orders | Time Points |
|----------|-------------|--------|------------|-------------|
| LiH      | (4e, 5o)    | 10     | l=1, l=2   | t=0.25, 0.75 |
| BeH₂     | (4e, 5o)    | 10     | l=1, l=2   | t=0.25, 0.75 |
| HF       | (4e, 5o)    | 10     | l=1, l=2   | t=0.25, 0.75 |

---

## Comparison Algorithms

The `comparisons/` directory contains implementations of the reference methods:

- **`comparisons/adapt_vqe/`** — Standard ADAPT-VQE using Qiskit's built-in `AdaptVQE` and UCCSD pool, for LiH, BeH₂, and HF.
- **`comparisons/counterdiabatic/`** — Pure counterdiabatic driving simulation using QuTiP's `sesolve`, solving the time-dependent Schrödinger equation with the full STA Hamiltonian H_ad(t) + H_cd(t).

---

## Module Reference

### `CD_ADAPT_Solver`

Main class. Handles molecule setup and runs the algorithm.

| Method | Description |
|--------|-------------|
| `__init__(geometry, basis, active_space, mapper_type)` | Set up molecule, build H_f, H_i, HF state |
| `compute_agp_pool(l_order, T_duration, time_points)` | Generate AGP operator pool |
| `run_cd_adapt(pool, max_iterations, gradient_threshold)` | Run ADAPT loop, return results |
| `exact_energy()` | Exact ground state via diagonalization (benchmark) |

### `schedule.py`

| Function | Description |
|----------|-------------|
| `schedule_function(t, T)` | λ(t) — smooth schedule from 0 to 1 |
| `dt_schedule_function(t, T)` | dλ/dt — time derivative of schedule |
| `adiabatic_hamiltonian(Hi, Hf, t, T)` | H_ad(t) = (1−λ)H_i + λH_f |

### `agp_pool.py`

| Function | Description |
|----------|-------------|
| `compute_agp_pool(Hi, Hf, l_order, T_duration, time_points)` | Build the full AGP Pauli pool |
| `compute_agp_coefficients(Had, dH, l_order)` | Solve for alpha_k coefficients |

### `utils.py`

| Function | Description |
|----------|-------------|
| `nested_commutator(A, B, n)` | Compute [A,[A,...[A,B]...]] (n times) |
| `frobenius_norm_pauli(op)` | Efficient Frobenius norm for Pauli operators |

---

## Simulation Time

For larger systems (many bond distances, higher l), simulations may take several hours.
The parallelized `dissociation_curve.py` script uses up to 80% of available CPU cores
to distribute calculations across bond distances.

For typical workstation use (16–32 cores), a 10-point dissociation curve for LiH
completes in approximately 1–3 hours depending on the AGP order.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `qiskit` | ≥0.45 | Quantum circuit simulation |
| `qiskit-nature` | ≥0.7 | Molecular Hamiltonians, HF state |
| `pyscf` | ≥2.3 | Electronic structure (SCF, integrals) |
| `numpy` / `scipy` | ≥1.24 / ≥1.10 | Numerics and optimization |
| `pandas` / `openpyxl` | ≥2.0 | Results I/O |
| `qutip` | ≥4.7 | *(Optional)* CD driving comparison |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use this code, please cite:

```bibtex
@article{cd-adapt-vqe,
  title   = {Counterdiabatic ADAPT-VQE},
  author  = {<authors>},
  journal = {arXiv preprint arXiv:2601.05973},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.05973}
}
```

---

## References

1. [Counterdiabatic ADAPT-VQE — arXiv:2601.05973](https://arxiv.org/abs/2601.05973) *(this work)*
2. [Grimsley et al., ADAPT-VQE — Nat. Commun. 10, 3007 (2019)](https://www.nature.com/articles/s41467-019-10988-2)
3. [Sels & Polkovnikov, AGP approximation — PNAS 114, E3909 (2017)](https://www.pnas.org/doi/10.1073/pnas.1619826114)
4. [Perez-Salinas et al., TETRIS-ADAPT-VQE — Phys. Rev. Research 6 (2024)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013254)
