"""
agp_pool.py - Approximate Gauge Potential (AGP) Operator Pool
==============================================================

This module generates the variational operator pool used in CD-ADAPT-VQE.
The pool is derived from the Approximate Gauge Potential (AGP), which is
the counterdiabatic correction operator that suppresses non-adiabatic
transitions during the adiabatic evolution.

Physical Background
-------------------
The exact Gauge Potential A_lambda satisfies:

    i * [H_ad, A_lambda] = dH_ad/dlambda

For many-body systems, solving for A_lambda exactly is exponentially hard.
The approximation used here (Sels & Polkovnikov, PNAS 2017) expands A_lambda
in a basis of nested commutators:

    A_lambda^(l) = sum_{k=1}^{l}  alpha_k(lambda) * O_k(lambda)

where the basis operators O_k are:

    O_k = [H_ad, [H_ad, ...[H_ad, dH/dlambda]...]]   (2k-1 commutators)

The coefficients alpha_k are found by minimizing the action functional,
which leads to the linear system:

    Gamma * alpha = -Gamma_b

    Gamma_{jk}   = Tr( O_{j+k+1}† O_{j+k+1} )  = ||O_{j+k+1}||_F^2
    Gamma_b_{j}  = Tr( O_j† O_j )               = ||O_j||_F^2

For CD-ADAPT-VQE, we extract the *Pauli strings* from these O_k operators
and use them as the pool from which the ADAPT loop selects ansatz operators.
Each unique Pauli string is an independent variational gate.

References
----------
- Sels & Polkovnikov, PNAS 114, E3909 (2017) - Original AGP approximation
- arXiv:2601.05973 - This implementation (CD-ADAPT-VQE)
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, commutator

from .utils import nested_commutator, frobenius_norm_pauli
from .schedule import schedule_function, adiabatic_hamiltonian


def compute_agp_pool(
    Hi: SparsePauliOp,
    Hf: SparsePauliOp,
    l_order: int = 1,
    T_duration: float = 1.0,
    time_points: list = None,
) -> list:
    """
    Generate the CD-ADAPT-VQE operator pool from the Approximate Gauge Potential.

    This function computes nested commutators of the adiabatic Hamiltonian
    with the Hamiltonian derivative, extracts the unique Pauli strings from
    the resulting operators, and returns them as the variational pool.

    The pool construction proceeds as follows for each time point t:
        1. Compute H_ad(t) = (1-lambda)*Hi + lambda*Hf
        2. Compute dH/dlambda = Hf - Hi
        3. For order k = 1, ..., l_order:
               O_k = [H_ad, [H_ad, ...[H_ad, dH]...]]  (2k-1 times)
        4. Decompose each O_k into individual Pauli strings
        5. Collect all unique Pauli strings across all time points and orders

    The use of multiple time points ensures the pool captures operators that
    are important at different stages of the adiabatic path, not just at a
    single snap-shot lambda value.

    Parameters
    ----------
    Hi : SparsePauliOp
        Initial Hamiltonian (Ising-type, whose ground state is the HF state).
    Hf : SparsePauliOp
        Final (target) molecular Hamiltonian.
    l_order : int, optional
        Maximum order of the AGP approximation. Default is 1.
        - l=1: pool from O_1 = [H_ad, dH]  (first-order AGP)
        - l=2: additionally adds O_3 = [H_ad,[H_ad,[H_ad, dH]]]
        Higher orders capture more non-adiabatic physics but grow the pool.
    T_duration : float, optional
        Total adiabatic evolution time T. Used to evaluate lambda(t).
        Default is 1.0.
    time_points : list of float, optional
        Time values at which to evaluate H_ad for pool construction.
        Typical choices used in the paper: [0.25], [0.75], or [0.25, 0.75].
        Default is [0.25].

    Returns
    -------
    list of SparsePauliOp
        A list of single-term Pauli operators (one per unique Pauli string).
        These are the candidate gates for the ADAPT selection step.

    Notes
    -----
    - The identity operator (all-I string) is always excluded from the pool,
      as it contributes only a global phase.
    - Coefficients are discarded; only the Pauli string labels are kept.
      The actual weight of each gate is learned by the VQE optimizer.
    - For large molecules (many qubits), pool generation can be slow because
      SparsePauliOp commutators scale as O(N^2) in the number of terms.

    Examples
    --------
    >>> pool = compute_agp_pool(Hi, Hf, l_order=1, time_points=[0.25])
    >>> len(pool)
    42
    >>> pool[0]
    SparsePauliOp(['XYII'], coeffs=[1.+0.j])
    """
    if time_points is None:
        time_points = [0.25]

    pool_set = set()  # Use a set to collect unique Pauli string labels
    dH = (Hf - Hi).simplify()  # dH/dlambda = H_f - H_i (constant along path)

    print(f"Generating AGP pool (l={l_order}, time_points={time_points})...")

    for t in time_points:
        Had = adiabatic_hamiltonian(Hi, Hf, t, T_duration)

        for k in range(1, l_order + 1):
            # O_k uses 2k-1 nested commutators (odd orders only)
            n_comm = 2 * k - 1
            op_k = nested_commutator(Had, dH, n_comm).simplify()
            _add_pauli_strings_to_pool(pool_set, op_k)

    # Convert set of Pauli label strings back to SparsePauliOp objects
    num_qubits = Hf.num_qubits
    final_pool = [SparsePauliOp(label) for label in pool_set]

    print(f"Pool generated: {len(final_pool)} unique Pauli operators.")
    return final_pool


def _add_pauli_strings_to_pool(pool_set: set, operator: SparsePauliOp):
    """
    Decompose a multi-term Pauli operator and add each Pauli string to the pool set.

    Individual Pauli strings (the basis elements like 'XXYY', 'ZIIZ', etc.)
    are added as string labels into the set. Duplicate labels are automatically
    ignored by the set data structure. The identity string (all 'I') is excluded.

    Parameters
    ----------
    pool_set : set of str
        Mutable set to add Pauli string labels into.
    operator : SparsePauliOp
        Multi-term Pauli operator whose terms will be extracted.
    """
    n = operator.num_qubits
    identity = 'I' * n
    for pauli_str in operator.paulis.to_labels():
        if pauli_str != identity:
            pool_set.add(pauli_str)


def compute_agp_coefficients(
    Had: SparsePauliOp,
    dH: SparsePauliOp,
    l_order: int,
) -> np.ndarray:
    """
    Solve for the AGP expansion coefficients alpha_k at a given lambda.

    Given the nested commutator operators O_1, O_2, ..., O_l, this function
    solves the linear system that minimizes the AGP action functional:

        Gamma * alpha = -Gamma_b

    where:
        Gamma_{jk}  = ||O_{j+k+1}||_F^2
        Gamma_b_j   = ||O_j||_F^2

    If the Gamma matrix is near-singular (det < 1e-12), falls back to
    least-squares via numpy.linalg.lstsq for numerical stability.

    Parameters
    ----------
    Had : SparsePauliOp
        The adiabatic Hamiltonian at the current time step.
    dH : SparsePauliOp
        The Hamiltonian derivative dH/dlambda = H_f - H_i.
    l_order : int
        Order of the AGP approximation (number of alpha coefficients).

    Returns
    -------
    np.ndarray of shape (l_order,)
        The coefficients alpha_1, alpha_2, ..., alpha_l.

    References
    ----------
    Sels & Polkovnikov, PNAS 114, E3909 (2017), Eq. (9-10).
    """
    # Pre-compute nested commutators up to order 2*l
    commutators = [
        nested_commutator(Had, dH, k + 1).simplify()
        for k in range(2 * l_order)
    ]
    norms_sq = [frobenius_norm_pauli(c) ** 2 for c in commutators]

    Gamma = np.zeros((l_order, l_order))
    Gamma_b = np.zeros(l_order)

    for j in range(l_order):
        Gamma_b[j] = norms_sq[j]
        for k in range(l_order):
            Gamma[j, k] = norms_sq[j + k + 1]

    det = np.linalg.det(Gamma)
    if abs(det) < 1e-12:
        alphas, _, _, _ = np.linalg.lstsq(Gamma, -Gamma_b, rcond=None)
    else:
        alphas = np.linalg.solve(Gamma, -Gamma_b)

    return alphas
