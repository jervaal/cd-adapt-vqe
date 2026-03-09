"""
CD-ADAPT-VQE: Counterdiabatic ADAPT-VQE Algorithm
===================================================

A Qiskit-based implementation of the Counterdiabatic Adaptive Derivative-Assembled
Problem-Tailored Variational Quantum Eigensolver (CD-ADAPT-VQE).

This package implements the algorithm described in:
    "Counterdiabatic ADAPT-VQE" - arXiv:2601.05973

Modules
-------
- hamiltonian   : Molecular Hamiltonian construction via PySCF + Qiskit Nature
- schedule      : Adiabatic scheduling functions lambda(t) and its derivatives
- agp_pool      : Approximate Gauge Potential (AGP) operator pool generation
- solver        : Main CD_ADAPT_Solver class (core algorithm loop)
- utils         : Mathematical utilities (norms, nested commutators)

Quick Start
-----------
>>> from cd_adapt_vqe import CD_ADAPT_Solver
>>> solver = CD_ADAPT_Solver("Li 0 0 0; H 0 0 1.5", basis="sto-3g",
...                          active_space={"num_particles": 4, "num_spatial_orbitals": 5})
>>> pool = solver.compute_agp_pool(l_order=1, time_points=[0.25])
>>> result = solver.run_cd_adapt(pool, max_iterations=30)
>>> print(result["energies"][-1])
"""

from .solver import CD_ADAPT_Solver
from .schedule import schedule_function, dt_schedule_function
from .agp_pool import compute_agp_pool
from .utils import nested_commutator, frobenius_norm_pauli

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    "CD_ADAPT_Solver",
    "schedule_function",
    "dt_schedule_function",
    "compute_agp_pool",
    "nested_commutator",
    "frobenius_norm_pauli",
]
