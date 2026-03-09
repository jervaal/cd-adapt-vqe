"""
[EN]
In this code we will organize, reestructure and generate the code for several instances 
of the new quantum algorithm cd-adapt for several molecules.

[ESP]
lo mismo pero en español xd

"""


#0 Importing the libraries need for several functions.
import qutip as qt
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from qutip import Bloch, QobjEvo, basis, sesolve, sigmay, sigmaz
from time import time
#from qutip.solver import Result, Options, config, _solver_safety_check
from qutip import *
import scipy as sp

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from qiskit.quantum_info import SparsePauliOp, Operator


# 1 Diego's functions to implement QuTIP objects:

# Many-Body Operators:
def SX(N : int, k : int) -> qt.Qobj:
    """Create a sigma_X operator at position k from
    a Hilbert space of N qubits
    """
    L = [qt.qeye(2)]*N 
    L[k] = qt.sigmax()
    
    return qt.tensor(L)

def SY(N : int, k : int) -> qt.Qobj:
    """Create a sigma_Y operator at position k from
    a Hilbert space of N qubits
    """
    L = [qt.qeye(2)]*N
    L[k] = qt.sigmay()
    return qt.tensor(L)

def SZ(N : int, k : int) -> qt.Qobj:
    """Create a sigma_Z operator at position k from
    a Hilbert space of N qubits
    """
    L = [qt.qeye(2)]*N
    L[k] = qt.sigmaz()
    return qt.tensor(L)

# Local Hamiltonian generation
def HlocalX(N : int, w : float) -> qt.Qobj:
    H = np.zeros((2**N, 2**N))
    H = qt.Qobj(H)
    H.dims = [[2]*N, [2]*N]
    
    for k1 in range(N):
        H = H + w*SX(N,k1)
    ### END FOR loop k1 ###
    return H
    
def HlocalY(N, w):
    H = np.zeros((2**N, 2**N))
    H = qt.Qobj(H)
    H.dims = [[2]*N, [2]*N]
    
    for k1 in range(N):
        H = H + w*SY(N,k1)
    ### END FOR loop k1 ###
    return H

def HlocalZ(N, w):
    H = np.zeros((2**N, 2**N))
    H = qt.Qobj(H)
    H.dims=[[2]*N, [2]*N]
    
    for k1 in range(N):
        H = H + w*SZ(N, k1)
    ### END FOR loop k1 ###
    return H
    

def qiskit_to_qutip(sparse_pauli_op : SparsePauliOp) -> qt.Qobj:
    """
    Convierte un SparsePauliOp de Qiskit a un operador Qobj de QuTiP,
    utilizando las funciones SX, SY, SZ definidas para actuar en N qubits.

    Parámetros:
    - sparse_pauli_op : qiskit.quantum_info.SparsePauliOp

    Retorna:
    - H_qutip : qutip.Qobj
        Hamiltoniano como suma de términos de muchos cuerpos en QuTiP
    """
    N = len(sparse_pauli_op.paulis[0].to_label())
    H_qutip = 0

    for label, coeff in zip(sparse_pauli_op.paulis.to_labels(), sparse_pauli_op.coeffs):
        term = coeff
        for i, char in enumerate(label):
            if char == 'I':
                continue
            elif char == 'X':
                term = term * SX(N, i)
            elif char == 'Y':
                term = term * SY(N, i)
            elif char == 'Z':
                term = term * SZ(N, i)
        H_qutip += term

    return H_qutip


def molecular_hamiltonian(molecule, basis='cc-pvdz', active_space_dict=None, mapper=JordanWignerMapper(), separate = False)  :

    """
    returns the one and two body terms of the Electronic Structure Hamiltonian of the molecule
    as Qutip operators.
    
    Parameters:
    - molecule : str or Z-matrix format, representing the geometric configuration of the molecule.
    - basis : str, the basis set to use for the calculation (default is 'cc-pvdz').
    - active_space_dict : dict, optional, a dictionary with parameters for the ActiveSpaceTransformer.
                          this dict have the form of: active_space_dict = {'num_electrons' : int, 'num_spatial_orbitals' : int)}
    - mapper : qiskit_nature.second_q.mappers.Mapper, the mapper to use for converting FermionicOp to SparsePauliOp.

    Returns:
    - one_body_qutip : qutip.Qobj, the one-body terms as a Qutip operator.
    - two_body_qutip : qutip.Qobj, the two-body terms as a Qutip operator.

    """
    driver = PySCFDriver(atom=molecule, basis=basis)
    full_problem = driver.run()

    identity_op = FermionicOp.one()


    if active_space_dict is not None:
        print('active space is not none')
        # Pasa los argumentos del diccionario directamente.
        act_space_tf = ActiveSpaceTransformer(**active_space_dict)
        problem = act_space_tf.transform(full_problem)
        ast_energy_op = problem.hamiltonian.constants['ActiveSpaceTransformer'] * identity_op

    else:
        problem = full_problem
        ast_energy_op = identity_op * 0 # Inicializar a 0 por si no existe la clave.

    
    hamiltonian = problem.hamiltonian.second_q_op()
    num_spin_orb = hamiltonian.num_spin_orbitals
    

    # Escalar el operador identidad por la energía nuclear.
    nuclear_energy_op = problem.nuclear_repulsion_energy * identity_op
    
        
    hamiltonian_without_nuclear = hamiltonian + nuclear_energy_op + ast_energy_op
    fermionic_ops = hamiltonian_without_nuclear
    
    if not separate:
        pauli_hamiltonian = mapper.map(fermionic_ops)
        return qiskit_to_qutip(pauli_hamiltonian)


    one_body_dict = {
        term: coef for term, coef in fermionic_ops.items() if len(term.split()) == 2
    }
    two_body_dict = {
        term: coef for term, coef in fermionic_ops.items() if len(term.split()) == 4
    }
        
    one_body_fermionic_op = FermionicOp(one_body_dict, num_spin_orbitals=num_spin_orb)
    two_body_fermionic_op = FermionicOp(two_body_dict, num_spin_orbitals=num_spin_orb)
    
    one_body_pauli = mapper.map(one_body_fermionic_op)
    two_body_pauli = mapper.map(two_body_fermionic_op)
    
    one_body_qutip = qiskit_to_qutip(one_body_pauli)
    two_body_qutip = qiskit_to_qutip(two_body_pauli)
    
    return one_body_qutip, two_body_qutip


def HartreeFock_bitstring_init_state(active_space_dict, mapper=JordanWignerMapper()) -> str:
    """
    Generates the bitstring for the Hartree-Fock state based on the active space configuration.

    Parameters:
        active_space_dict (dict): Dictionary containing the active space configuration.
                                  It should have 'num_spatial_orbitals' and 'num_particles' keys.
                                  For example: {'num_spatial_orbitals': N, 'num_particles': (a, b)}., N, a and b ε Integers.
    
        mapper (qiskit_nature.second_q.mappers.Mapper): The mapper used for the conversion.
        
    Returns:
        hf_state_bitstring (str): The bitstring representation of the Hartree-Fock state.
    """
    # 1. Extrae los parámetros del diccionario de espacio activo
    num_spatial_orbitals = active_space_dict['num_spatial_orbitals']
    num_particles = active_space_dict['num_particles']

    # 2. Crea el circuito de Hartree-Fock con los parámetros correctos
    hf_circuit = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    # 3. Calcula el estado y extrae el bitstring con probabilidad 1
    # Statevector() puede crear un estado a partir de un circuito
    # y probabilities_dict() devuelve un diccionario con los estados y sus probabilidades
    hf_state_bitstring = list(Statevector(hf_circuit).probabilities_dict().keys())[0]

    return hf_state_bitstring


def HartreeFock_GroundState_Hamiltonian(b_str: str) -> qt.Qobj:
    """
    Generates H = sum_i^N alpha_i * SZ(N, i) with alpha_i = (-1)^(b_i + 1),
    where b_str is the binary string representation of the state.

    Parameters
    ----------
    b_str : str
        Binary string, e.g., '00110011'.

    Returns
    -------
    H : qutip.Qobj
        Hamiltonian in the full N-qubit Hilbert space.
    """
    if not isinstance(b_str, str) or not all(c in '01' for c in b_str):
        raise ValueError("b_str must be a binary string.")
    
    N = len(b_str) # num_qubits.
    
    # Crea el hamiltoniano utilizando una comprensión de lista para concisión
    H = sum([
        ((-1)**(int(bit) + 1)) * SZ(N, i)
        for i, bit in enumerate(b_str)
    ])
    
    return H



def energy_gap(Hf: qt.Qobj) -> float:
    """
    Given a QuTiP Hamiltonian Hf, returns the energy gap between the two lowest
    non-degenerate eigenvalues.
    
    Parameters
    ----------
    Hf : qt.Qobj 
        The molecular Hamiltonian as a qt.Qobj object.
    
    Returns
    -------
    energy_gap : float
        The energy gap between the two lowest non-degenerate eigenvalues of Hf,
        measured in Hartrees.
    
    Raises
    ------
    ValueError
        If the Hamiltonian does not have at least two distinct eigenvalues.
    """
    # Calculate eigenvalues and get unique values.
    # We sort them in ascending order.
    unique_eigenvalues = np.unique(Hf.eigenenergies())

    # Check if there are at least two distinct eigenvalues.
    if len(unique_eigenvalues) < 2:
        raise ValueError("The Hamiltonian must have at least two distinct eigenvalues to calculate the energy gap.")
    
    # The first two elements of the sorted unique array are the two lowest non-degenerate eigenvalues.
    return unique_eigenvalues[1] - unique_eigenvalues[0]




def conmutador_anidado_iter(H: qt.Qobj, aH: qt.Qobj, n: int) -> qt.Qobj:
    """
    Calcula el conmutador anidado de un operador 'aH' con un hamiltoniano 'H' 
    un número 'n' de veces.

    La operación es: [H, [H, [H, ..., aH]...]]

    Parameters
    ----------
    H : qt.Qobj
        El operador hamiltoniano.
        
    aH : qt.Qobj
        El operador con el que se inicia la iteración., en nuestro caso usamos aH = \partial_\lambda = H_\lambda = H_f - H_i

    n : int
        El número de veces que se anida el conmutador.

    Returns
    -------
    comm : qt.Qobj
        El resultado del conmutador anidado.
        
    Raises
    ------
    TypeError
        Si H, aH no son objetos de QuTiP o si n no es un entero.
    """
    if not all(isinstance(op, qt.Qobj) for op in [H, aH]):
        raise TypeError("H y aH deben ser objetos de QuTiP.")
    if not isinstance(n, int):
        raise TypeError("n debe ser un entero.")
    
    comm = aH
    for _ in range(n):
        comm = qt.commutator(H, comm)
    return comm


def schedule_function(t: float, T: float) -> float:
    """
    Calculates a scheduling function based on time 't' and total time 'T'.

    The function is given by: sin^2(pi/2 * sin^2(pi*t/(2*T)))

    Parameters
    ----------
    t : float
        The current time.
    T : float
        The total time of the schedule.

    Returns
    -------
    float
        The calculated value of the schedule function.

    Raises
    ------
    ValueError
        If T is zero, to prevent division by zero.
    """
    if T == 0:
        raise ValueError("Total time 'T' cannot be zero.")
    
    sin_term = np.sin(np.pi * t / (2 * T))
    return np.sin(np.pi / 2 * sin_term**2)**2


def dt_schedule_function(t: float, T: float) -> float:
    """
    Calculates the time derivative of the schedule_function.

    Parameters
    ----------
    t : float
        The current time.
    T : float
        The total time of the schedule.

    Returns
    -------
    float
        The calculated value of the derivative of the schedule function.
        
    Raises
    ------
    ValueError
        If T is zero, to prevent division by zero.
    """
    if T == 0:
        raise ValueError("Total time 'T' cannot be zero.")
        
    sin_t_T = np.sin(np.pi * t / T)
    sin_inner_term = np.sin(np.pi * np.sin(np.pi * t / (2 * T))**2)
    
    return (np.pi**2 / (4 * T)) * sin_t_T * sin_inner_term


def alpha_coefficient_calculation(H_ad: qt.Qobj, aH: qt.Qobj, l: int) -> np.ndarray:
    """
    Calculates the alpha coefficients for an adiabatic gauge potential series expansion.

    The function solves the linear system Gamma * alphas = -Gamma_vector, where
    Gamma and Gamma_vector are built from nested commutators of the Hamiltonian.

    Parameters
    ----------
    H_ad : qt.Qobj
        The adiabatic Hamiltonian (H).
    aH : qt.Qobj
        The operator to be expanded (aH).
    l : int
        The truncation order of the expansion.

    Returns
    -------
    alphas : numpy.ndarray
        The array of calculated alpha coefficients.

    Raises
    ------
    ValueError
        If the input parameters are of incorrect types.
    """
    # 1. Input validation
    if not isinstance(H_ad, qt.Qobj) or not isinstance(aH, qt.Qobj):
        raise ValueError("H_ad and aH must be QuTiP Qobj objects.")
    if not isinstance(l, int) or l <= 0:
        raise ValueError("l must be a positive integer.")

    # 2. Initialize Gamma matrix and Gamma_vector
    Gamma_matrix = np.zeros([l, l])
    Gamma_vector = np.zeros(l)

    # 3. Fill the Gamma_vector and Gamma_matrix using nested commutators
    for j in range(l):
        # Correctly call the nested commutator function
        # The commutator is [H, aH] for j=0, [H, [H, aH]] for j=1, etc.
        # So we need j+1 nested commutations.
        comm_j_plus_1 = conmutador_anidado_iter(H_ad, aH, j + 1)
        Gamma_vector[j] = comm_j_plus_1.norm('fro')**2

        for k in range(l):
            # This term requires j+k+2 nested commutations
            comm_j_k_plus_2 = conmutador_anidado_iter(H_ad, aH, j + k + 2)
            Gamma_matrix[j, k] = comm_j_k_plus_2.norm('fro')**2

    # 4. Solve the linear system
    # Check if the matrix is singular (determinant is close to zero)
    det = np.linalg.det(Gamma_matrix)
    if np.abs(det) < 1e-11:
        # Use least-squares for singular matrices
        alphas, _, _, _ = np.linalg.lstsq(Gamma_matrix, -Gamma_vector, rcond=None)
    else:
        # Use a direct solver for non-singular matrices
        alphas = np.linalg.solve(Gamma_matrix, -Gamma_vector)

    return alphas


def Adiabatic_Hamiltonian(Hi: qt.Qobj, Hf: qt.Qobj, lambdat: float) -> qt.Qobj:
    """
    Calculates the adiabatic Hamiltonian, Had, as a linear interpolation
    between an initial Hamiltonian (Hi) and a final Hamiltonian (Hf).

    The interpolation is defined as: Had = (1 - lambda(t)) * Hi + lambda(t) * Hf.

    Parameters
    ----------
    Hi : qt.Qobj
        The initial Hamiltonian.
    Hf : qt.Qobj
        The final Hamiltonian.
    lambdat : float
        The time-dependent parameter, lambda(t), which typically varies
        from 0 to 1.

    Returns
    -------
    Had : qt.Qobj
        The adiabatic Hamiltonian at a given value of lambdat.

    Raises
    ------
    TypeError
        If Hi or Hf are not QuTiP Qobj objects, or if lambdat is not a float.
    ValueError
        If the dimensions of Hi and Hf do not match.
    """
    # Type and dimension validation
    if not all(isinstance(h, qt.Qobj) for h in [Hi, Hf]):
        raise TypeError("Hi and Hf must be QuTiP Qobj objects.")
    if not Hi.dims == Hf.dims:
        raise ValueError("Hi and Hf must have the same dimensions.")
    if not isinstance(lambdat, (int, float)):
        raise TypeError("lambdat must be a number.")
    
    # Calculate the adiabatic Hamiltonian in a single line
    Had = (1 - lambdat) * Hi + lambdat * Hf
    
    return Had


def CounterDiabatic_Hamiltonian(Hi: qt.Qobj, Hf: qt.Qobj, t: float, T: float, l: int) -> qt.Qobj:
    """
    Computes the Counter-Diabatic Hamiltonian (Hcd) as a series expansion.

    The Counter-Diabatic Hamiltonian is an addition to the standard adiabatic
    Hamiltonian that suppresses non-adiabatic transitions, accelerating the process.

    Parameters
    ----------
    Hi : qt.Qobj
        The initial Hamiltonian.
    Hf : qt.Qobj
        The final Hamiltonian.
    t : float
        The current time in the adiabatic process.
    T : float
        The total time of the adiabatic process.
    l : int
        The truncation order of the expansion for the counter-diabatic term.

    Returns
    -------
    H_cd : qt.Qobj
        The full Counter-Diabatic Hamiltonian.

    Raises
    ------
    TypeError
        If Hi or Hf are not QuTiP Qobj objects, or if t, T, or l are not numbers.
    ValueError
        If the dimensions of Hi and Hf do not match, or if l is not a positive integer.
    """
    # 1. Type and dimension validation
    if not all(isinstance(h, qt.Qobj) for h in [Hi, Hf]):
        raise TypeError("Hi and Hf must be QuTiP Qobj objects.")
    if not Hi.dims == Hf.dims:
        raise ValueError("Hi and Hf must have the same dimensions.")
    if not all(isinstance(val, (int, float)) for val in [t, T]):
        raise TypeError("t and T must be numbers.")
    if not isinstance(l, int) or l <= 0:
        raise ValueError("l must be a positive integer.")
        
    # 2. Calculate necessary intermediate values
    aH = Hf - Hi
    lambdat = schedule_function(t, T)
    lambdatt = dt_schedule_function(t, T)
    Had = Adiabatic_Hamiltonian(Hi, Hf, lambdat)
    
    # 3. Calculate the alpha coefficients for the series expansion
    alphas = alpha_coefficient_calculation(Had, aH, l)
    
    # 4. Construct the counter-diabatic term (H_cd_term)
    # The initial value is a zero QuTiP object with the same dimensions as the Hamiltonians.

    
    H_cd_term = qt.Qobj(np.zeros(Had.shape), dims=Had.dims)
    
    for ii in range(l):
        nested_commutator = conmutador_anidado_iter(Had, aH, 2 * ii + 1)
        term = nested_commutator * (1j * lambdatt * alphas[ii])
        H_cd_term += term
        
    # 5. returns d \lambda / dt * AGP
    return H_cd_term


#verificar este, que es basicamente lo mismo de arriba pero modificado con AI:

def CounterDiabatic_Hamiltonian(Hi: qt.Qobj, Hf: qt.Qobj, t: float, T: float, l: int, 
                                return_cd_terms_unweighted: bool = False) -> qt.Qobj:
    """
    Computes the Counter-Diabatic Hamiltonian (Hcd) as a series expansion.

    The Counter-Diabatic Hamiltonian is an addition to the standard adiabatic
    Hamiltonian that suppresses non-adiabatic transitions, accelerating the process.

    Parameters
    ----------
    Hi : qt.Qobj
        The initial Hamiltonian.
    Hf : qt.Qobj
        The final Hamiltonian.
    t : float
        The current time in the adiabatic process.
    T : float
        The total time of the adiabatic process.
    l : int
        The truncation order of the expansion for the counter-diabatic term.
    return_cd_terms_unweighted : bool, optional
        If True, the function returns only the sum of the unweighted nested
        commutator terms (without the 1j * d(lambda)/dt * alpha[ii] factors).
        If False (default), it returns the full H_adiabatic + H_cd Hamiltonian.

    Returns
    -------
    H_cd_or_full_H : qt.Qobj
        If `return_cd_terms_unweighted` is True, returns the sum of unweighted
        counter-diabatic terms. Otherwise, returns the full Counter-Diabatic Hamiltonian
        (Had + H_cd_term).

    Raises
    ------
    TypeError
        If Hi or Hf are not QuTiP Qobj objects, or if t, T, l are not numbers.
    ValueError
        If the dimensions of Hi and Hf do not match, or if l is not a positive integer.
    """
    # 1. Type and dimension validation
    if not all(isinstance(h, qt.Qobj) for h in [Hi, Hf]):
        raise TypeError("Hi and Hf must be QuTiP Qobj objects.")
    if not Hi.dims == Hf.dims:
        raise ValueError("Hi and Hf must have the same dimensions.")
    if not all(isinstance(val, (int, float)) for val in [t, T]):
        raise TypeError("t and T must be numbers.")
    if not isinstance(l, int) or l <= 0:
        raise ValueError("l must be a positive integer.")
    if not isinstance(return_cd_terms_unweighted, bool):
        raise TypeError("return_cd_terms_unweighted must be a boolean.")
        
    # 2. Calculate necessary intermediate values
    aH = Hf - Hi
    lambdat = schedule_function(t, T)
    lambdatt = dt_schedule_function(t, T)
    Had = Adiabatic_Hamiltonian(Hi, Hf, lambdat)
    
    # 3. Calculate the alpha coefficients for the series expansion
    alphas = alpha_coefficient_calculation(Had, aH, l)
    
    # 4. Construct the counter-diabatic term (H_cd_term)
    # The initial value is a zero QuTiP object with the same dimensions as the Hamiltonians.
    H_cd_term = qt.Qobj(np.zeros(Had.shape), dims=Had.dims)
    for ii in range(l):
        nested_commutator = conmutador_anidado_iter(Had, aH, 2 * ii + 1)
        
        # Apply weighting based on the new boolean parameter
        if return_cd_terms_unweighted:
            term = nested_commutator # No weighting
        else:
            term = nested_commutator * (1j * lambdatt * alphas[ii])
            
        H_cd_term += term
        
    return H_cd_term

"""    # 5. Return based on the boolean parameter
    if return_cd_terms_unweighted:
        return H_cd_term # Return only the (unweighted) CD terms
    else:
        return H_cd_term # + Had impulse regime Return the full Counter-Diabatic Hamiltonian
"""

def qutip_to_sparse_pauli_op(qutip_op: qt.Qobj) -> SparsePauliOp:
    """
    Converts a QuTiP Qobj to a SparsePauliOp.

    Parameters
    ----------
    qutip_op : qt.Qobj
        The QuTiP operator to convert.

    Returns
    -------
    SparsePauliOp
        The converted SparsePauliOp.

    """
    qiskit_op = Operator(qutip_op.full())
    return SparsePauliOp.from_operator(qiskit_op)


def num_pauli_str_terms(p : SparsePauliOp) -> int:
    return len(p.paulis.to_labels())



def compute_num_cd_terms(Hi: qt.Qobj, Hf: qt.Qobj, time_linspace: np.ndarray, l: int) -> np.ndarray:
    """
    Computes the number of Pauli strings for the unweighted counter-diabatic terms
    for different orders of the expansion.

    Parameters
    ----------
    Hi : qt.Qobj
        The initial Hamiltonian.
    Hf : qt.Qobj
        The final Hamiltonian.
    time_linspace : np.ndarray
        An array of time points over which to compute the terms.
    l : int
        The maximum truncation order of the expansion.

    Returns
    -------
    num_pauli_A_l : np.ndarray
        A 2D array of shape (l, len(time_linspace)) where each entry
        is the number of Pauli strings for the l-th order term at a given time.
    
    Raises
    ------
    ValueError
        If l is not a positive integer.
    """
    # 1. Input validation
    if not isinstance(l, int) or l <= 0:
        raise ValueError("The truncation order 'l' must be a positive integer.")
    if not isinstance(time_linspace, np.ndarray):
        raise TypeError("The time_linspace must be a NumPy array.")

    # 2. Initialize the result array
    num_pauli_A_l = np.zeros(shape=(l, len(time_linspace)))

    # 3. Calculate terms for the A_1 order (l=1) separately
    A1 = Hf - Hi
    pauli_A1 = qutip_to_sparse_pauli_op(A1)
    num_pauli_A_l[0, :] = num_pauli_str_terms(pauli_A1)

    # 4. Loop over orders from 2 to l
    # Note: `orders` here refers to the l in our previous definitions
    for order in range(2, l + 1):
        for i, t in enumerate(time_linspace):
            # Calculate the unweighted counter-diabatic terms for the current order
            # The order of the nested commutator is 2*order-1
            # You were passing `orders+1` which would go up to order l+1, so I corrected it to `order`
            H_cd_terms = CounterDiabatic_Hamiltonian(Hi, Hf, t, T=1, l=order, return_cd_terms_unweighted=True)
            
            # The `H_cd_terms` variable from the optimized function returns the sum of all terms up to `l`.
            # To get just the last term (the `l`-th order term), we need to modify the logic.
            # A more direct approach is needed if you want to isolate the `l`-th order term.
            
            # For simplicity, let's assume the goal is to get the total number of terms
            # for the series up to order `order`.
            pauli_H_cd = qutip_to_sparse_pauli_op(H_cd_terms)
            num_pauli_A_l[order - 1, i] = num_pauli_str_terms(pauli_H_cd)
    
    return num_pauli_A_l


import qutip as qt
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple, Dict, Optional

# Assuming previous functions like build_operator_pool, build_ansatz, cost_function are defined.

# -----------------------------------------------------------------------------

def build_operator_pool(mean_operator: SparsePauliOp) -> Tuple[List[qt.Qobj], List[str]]:
    """
    Builds a pool of QuTiP operators from a SparsePauliOp, along with their labels.

    This function iterates through the Pauli strings of the input operator and
    converts each one into a corresponding QuTiP operator using a tensor product.

    Parameters
    ----------
    mean_operator : SparsePauliOp
        The operator (e.g., the adiabatic gauge potential) to build the pool from.

    Returns
    -------
    Tuple[List[qt.Qobj], List[str]]
        - A list of QuTiP Qobj operators.
        - A list of their corresponding Pauli string labels.
    """
    pool = []
    pool_labels = []
    
    # Iterate through Pauli strings and their coefficients
    for pstr in mean_operator.paulis.to_labels():
        factors = []
        for char in pstr:
            if char == 'I':
                factors.append(qt.qeye(2))
            elif char == 'X':
                factors.append(qt.sigmax())
            elif char == 'Y':
                factors.append(qt.sigmay())
            elif char == 'Z':
                factors.append(qt.sigmaz())
            else:
                raise ValueError(f"Unknown Pauli character: {char}")
        
        op = qt.tensor(factors)
        pool.append(op)
        pool_labels.append(pstr)

    return pool, pool_labels

# -----------------------------------------------------------------------------

def build_ansatz(params: List[float], operators: List[qt.Qobj], initial_state: qt.Qobj) -> qt.Qobj:
    """
    Builds the ansatz state by applying a series of unitary operators.

    The state is constructed by successively applying the exponential of each
    operator, scaled by its corresponding parameter.

    Parameters
    ----------
    params : List[float]
        List of variational parameters.
    operators : List[qt.Qobj]
        List of QuTiP operators to apply.
    initial_state : qt.Qobj
        The starting state vector.

    Returns
    -------
    qt.Qobj
        The final ansatz state vector.
    """
    state = initial_state
    for theta, op in zip(params, operators):
        state = (-1j * theta * op).expm() * state
    return state

# -----------------------------------------------------------------------------

def cost_function(params: List[float], operators: List[qt.Qobj], initial_state: qt.Qobj, Hf: qt.Qobj) -> float:
    """
    Calculates the expected value of the energy for a given ansatz state.

    Parameters
    ----------
    params : List[float]
        List of variational parameters.
    operators : List[qt.Qobj]
        List of QuTiP operators defining the ansatz.
    initial_state : qt.Qobj
        The initial state vector.
    Hf : qt.Qobj
        The final Hamiltonian operator.

    Returns
    -------
    float
        The expected energy, returned as a real number.
    """
    psi = build_ansatz(params, operators, initial_state)
    return qt.expect(Hf, psi).real

# -----------------------------------------------------------------------------

def ADAPT_CD(Hi: qt.Qobj, Hf: qt.Qobj, AG_op: SparsePauliOp, N: int,
             gradient_threshold: float = 1e-2, max_iter: int = 1000,
             max_num_op: Optional[int] = None, norm_type: str = '2') -> Dict:
    """
    Executes the ADAPT-CD algorithm to find the ground state of a Hamiltonian.

    This function iteratively builds an ansatz by adding operators that maximize
    the gradient of the energy with respect to the ansatz parameters.

    Parameters
    ----------
    Hi : qt.Qobj
        The initial Hamiltonian.
    Hf : qt.Qobj
        The final molecular Hamiltonian.
    AG_op : SparsePauliOp
        Adiabatic Gauge potential term as SparsePauliOp.
    N : int
        The number of qubits.
    gradient_threshold : float, optional
        Threshold for gradient convergence (default is 1e-2).
    max_iter : int, optional
        Maximum number of iterations for the VQE optimizer (default is 1000).
    max_num_op : Optional[int], optional
        Maximum number of operators in the ansatz (default is None, meaning no limit).
    norm_type : str, optional
        Type of norm to use for the gradient convergence check:
        '2' (Euclidean norm), 'inf' (maximum absolute value). Default is '2'.

    Returns
    -------
    Dict
        A dictionary containing the energy trace, gradient trace, ansatz operators,
        and final parameters.
    """
    
    if norm_type not in ['2', 'inf']:
        raise ValueError("norm_type must be '2' or 'inf'.")

    pool, pool_labels = build_operator_pool(AG_op)
    # The ground state of Hi is psi_0
    psi_0 = Hi.eigenstates()[1][0]

    ansatz_ops = []
    ansatz_labels = []
    params = []
    energy_trace = []
    grad_trace = []

    # Pre-compute the commutators once for efficiency
    commutators = [qt.commutator(Hf, A) for A in pool]

    while True:
        # Build the current ansatz state
        psi = build_ansatz(params, ansatz_ops, psi_0) if params else psi_0

        # Calculate gradients for the entire pool
        # Gradients are calculated as <psi| [H, A_i] |psi>
        gradients = [np.abs(qt.expect(comm, psi)) for comm in commutators]

        # Check for convergence

        if norm_type == '2':
            norm_grad_vector = np.linalg.norm(gradients)
        else: # norm_type is 'inf'
            norm_grad_vector = np.max(gradients)
        
        grad_trace.append(norm_grad_vector)
        
        #print(f'iteracion # {len(ansatz_labels)} ; |g| = {norm_grad_vector} \n')

        if (norm_grad_vector < gradient_threshold) or (max_num_op is not None and len(ansatz_ops) >= max_num_op):
            break

        # Select the operator with the largest gradient
        max_index = np.argmax(gradients)
        ansatz_ops.append(pool[max_index])
        ansatz_labels.append(pool_labels[max_index])
        
        # Add a new parameter (initialized to 0) to optimize
        init_theta = params + [0.0]
        #bounds = [(0, 2 * np.pi) for _ in range(len(init_theta))]

        #print(f'num de op: {len(ansatz_labels)}')

        # Optimize the parameters for the new ansatz
        result = minimize(
            cost_function,
            init_theta,
            args=(ansatz_ops, psi_0, Hf),
            method='L-BFGS-B',
            options={'disp': False, 'maxiter': max_iter}
        
        )

        params = list(result.x)
        energy_trace.append(result.fun)
        
    # Final VQE optimization on the complete ansatz
    result = minimize(
        cost_function,
        params,
        args=(ansatz_ops, psi_0, Hf),
        method='L-BFGS-B',
        options={'disp': False, 'maxiter': max_iter})
    

    
    energy_trace.append(result.fun)
    final_params = list(result.x)

    print(f"Número total de operadores en el ansatz: {len(ansatz_ops)}")
    print(f"Energía final: {energy_trace[-1]}")
    print(f'Energía exacta: {Hf.eigenenergies()[0]} \n')

    return {
        "energy_trace": energy_trace,
        "grad_trace": grad_trace,
        "ansatz_ops": ansatz_ops,
        "ansatz_labels": ansatz_labels,
        "final_params": final_params
    }
