"""
schedule.py - Adiabatic Scheduling Functions
=============================================

This module defines the time-dependent scheduling function lambda(t) and
its time derivative, which control the interpolation between the initial
Hamiltonian H_i and the target Hamiltonian H_f during the adiabatic evolution.

Physical Background
-------------------
In adiabatic quantum computing, the system evolves under a time-dependent
Hamiltonian:

    H_ad(t) = (1 - lambda(t)) * H_i  +  lambda(t) * H_f

where lambda(t) is a smooth schedule satisfying:
    - lambda(0) = 0   →  start at H_i (easy initial state)
    - lambda(T) = 1   →  end at H_f  (target ground state)

A smooth schedule reduces non-adiabatic excitations. The schedule used
here follows a doubly-sinusoidal form chosen for having zero first-order
derivative at both endpoints (Eq. (schedule) of arXiv:2601.05973).

The time derivative d(lambda)/dt appears explicitly in the counterdiabatic
Hamiltonian H_cd(t), which is the correction term that suppresses transitions
between adiabatic eigenstates.
"""

import numpy as np


def schedule_function(t: float, T: float) -> float:
    """
    Adiabatic scheduling function lambda(t).

    Computes the smooth interpolation parameter that drives the system
    from the initial Hamiltonian (lambda=0) to the final one (lambda=1).

    The schedule is defined as:

        lambda(t) = sin^2( (pi/2) * sin^2( pi*t / (2T) ) )

    This form guarantees:
        - lambda(0)  = 0  (starts at H_i)
        - lambda(T)  = 1  (ends at H_f)
        - dlambda/dt = 0  at both endpoints (smooth ramp-up and ramp-down)

    Parameters
    ----------
    t : float
        Current time in the evolution [same units as T].
    T : float
        Total evolution duration. If T=0, returns 0 to avoid division by zero.

    Returns
    -------
    float
        lambda(t) in [0, 1].

    Examples
    --------
    >>> schedule_function(0.0, 1.0)
    0.0
    >>> schedule_function(1.0, 1.0)
    1.0
    >>> schedule_function(0.5, 1.0)  # midpoint
    0.5
    """
    if T == 0:
        return 0.0
    inside_term = np.sin(np.pi * t / (2 * T))
    return float(np.sin((np.pi / 2) * (inside_term ** 2)) ** 2)


def dt_schedule_function(t: float, T: float) -> float:
    """
    Time derivative of the adiabatic scheduling function: d(lambda)/dt.

    This derivative appears explicitly in the counterdiabatic Hamiltonian:

        H_cd(t) = i * (dlambda/dt) * A_lambda(t)

    where A_lambda is the Approximate Gauge Potential (AGP). A large
    derivative at some time t means stronger counterdiabatic correction
    is needed at that moment.

    The analytic derivative is:

        d(lambda)/dt = (pi^2 / (4T)) * sin(pi*t/T) * sin(pi * sin^2(pi*t/(2T)))

    Parameters
    ----------
    t : float
        Current time in the evolution [same units as T].
    T : float
        Total evolution duration. If T=0, returns 0 to avoid division by zero.

    Returns
    -------
    float
        d(lambda)/dt at time t.

    Notes
    -----
    This derivative is zero at t=0 and t=T, consistent with the smooth
    boundary conditions of lambda(t). This property is important for
    convergence: the counterdiabatic correction vanishes at start and end,
    so the preparation protocol is compatible with the HF initial state.

    Examples
    --------
    >>> dt_schedule_function(0.0, 1.0)
    0.0
    >>> dt_schedule_function(1.0, 1.0)
    0.0
    >>> dt_schedule_function(0.5, 1.0)  # maximum derivative near midpoint
    """
    if T == 0:
        return 0.0
    sin_t_T = np.sin(np.pi * t / T)
    sin_inner_term = np.sin(np.pi * np.sin(np.pi * t / (2 * T)) ** 2)
    return float((np.pi ** 2 / (4 * T)) * sin_t_T * sin_inner_term)


def adiabatic_hamiltonian(Hi, Hf, t: float, T: float):
    """
    Construct the instantaneous adiabatic Hamiltonian H_ad(t).

    Computes:
        H_ad(t) = (1 - lambda(t)) * H_i  +  lambda(t) * H_f

    Parameters
    ----------
    Hi : SparsePauliOp
        Initial Hamiltonian (defines the easy ground state, e.g. a Ising-type
        model whose ground state is the Hartree-Fock state).
    Hf : SparsePauliOp
        Final (target) Hamiltonian (the molecular electronic Hamiltonian).
    t : float
        Current time.
    T : float
        Total evolution duration.

    Returns
    -------
    SparsePauliOp
        The adiabatic Hamiltonian at time t, simplified.
    """
    lam = schedule_function(t, T)
    return ((1.0 - lam) * Hi + lam * Hf).simplify()
