"""
utils.py - Mathematical Utilities for CD-ADAPT-VQE
====================================================

This module provides core mathematical tools used throughout the algorithm,
including nested commutator computation and operator norm calculations.

Background
----------
In the CD-ADAPT-VQE algorithm, the Approximate Gauge Potential (AGP) is
expressed as a series of nested commutators of the adiabatic Hamiltonian
H_ad(t) with the Hamiltonian derivative dH/dlambda. These operators form
the basis for the variational ansatz.

The Frobenius norm of these operators is used to solve for the AGP
coefficients alpha_k via a linear system (see agp_pool.py).
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, commutator


def nested_commutator(A: SparsePauliOp, B: SparsePauliOp, n: int) -> SparsePauliOp:
    """
    Compute the n-th order nested commutator [A, [A, ...[A, B]...]] .

    This function builds the operator sequence that defines the terms
    O_k in the AGP expansion (Eq. 11 of the paper):

        O_1 = [H_ad, dH/dlambda]
        O_2 = [H_ad, [H_ad, dH/dlambda]]
        ...
        O_n = [H_ad, O_{n-1}]

    These operators are anti-Hermitian by construction (since H_ad is Hermitian),
    and form the operator pool from which CD-ADAPT-VQE selects ansatz terms.

    Parameters
    ----------
    A : SparsePauliOp
        The outer operator in the nested commutator (typically H_ad).
    B : SparsePauliOp
        The inner-most operator (typically dH/dlambda = H_f - H_i).
    n : int
        Order of the nesting. n=1 gives [A, B], n=2 gives [A,[A,B]], etc.

    Returns
    -------
    SparsePauliOp
        The resulting nested commutator, simplified (zero terms removed).

    Examples
    --------
    >>> O1 = nested_commutator(H_ad, dH, n=1)  # First-order AGP term
    >>> O3 = nested_commutator(H_ad, dH, n=3)  # Third-order AGP term
    """
    comm = B
    for _ in range(n):
        comm = commutator(A, comm).simplify()
    return comm.simplify()


def frobenius_norm_pauli(op: SparsePauliOp) -> float:
    """
    Compute the exact Frobenius norm of a Pauli operator without matrix construction.

    For a Pauli operator written as a sum of Pauli strings:
        Op = sum_i c_i * P_i

    the Frobenius norm is:
        ||Op||_F = sqrt( Tr(Op† Op) ) = sqrt( sum_i |c_i|^2 * 2^N )

    This identity holds because Pauli matrices are orthonormal under the
    Hilbert-Schmidt inner product: Tr(P_i† P_j) = 2^N * delta_{ij}.

    This is significantly more efficient than the naive approach of converting
    to a dense matrix (which scales as O(4^N)), and is exact for all system sizes.

    Parameters
    ----------
    op : SparsePauliOp
        A Pauli operator (sum of weighted Pauli strings) on N qubits.

    Returns
    -------
    float
        The Frobenius norm of the operator.

    Notes
    -----
    The Frobenius norm is used in cd_adapt_vqe/agp_pool.py to build the
    Gamma matrix that determines the alpha_k coefficients of the AGP expansion.
    """
    dim = 2 ** op.num_qubits
    return np.sqrt(np.sum(np.abs(op.coeffs) ** 2) * dim)
