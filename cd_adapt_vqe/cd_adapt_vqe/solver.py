"""
solver.py - Core CD-ADAPT-VQE Solver
======================================

This module contains the CD_ADAPT_Solver class, which is the main entry
point for running the Counterdiabatic ADAPT-VQE algorithm.

Algorithm Overview
------------------
CD-ADAPT-VQE is a hybrid quantum-classical algorithm that combines:

1. **Counterdiabatic Driving**: Uses the Approximate Gauge Potential (AGP)
   to construct a physically-motivated operator pool. Unlike standard
   ADAPT-VQE (which uses fermionic or qubit excitation pools), the CD pool
   is derived from the adiabatic path between H_i and H_f.

2. **ADAPT (Adaptive Derivative-Assembled Problem-Tailored) Loop**:
   Iteratively selects the most important operator from the pool (by
   measuring the energy gradient), appends it to the ansatz, and
   re-optimizes all parameters via VQE.

3. **VQE Subroutine**: At each ADAPT step, L-BFGS-B minimizes the energy
   expectation value <psi(theta)|H_f|psi(theta)> over all ansatz parameters.

The resulting ansatz has the form:

    |psi(theta)> = prod_{k=1}^{n} exp(-i * theta_k * G_k) |psi_HF>

where G_k are the selected Pauli operators and |psi_HF> is the Hartree-Fock
reference state.

Workflow
--------
1. Instantiate CD_ADAPT_Solver with molecule geometry
2. Call compute_agp_pool() to generate the operator pool
3. Call run_cd_adapt() to run the ADAPT loop
4. Access results: energies, gradients, final ansatz operators

References
----------
- Grimsley et al., Nat. Commun. 10, 3007 (2019) - Original ADAPT-VQE
- Sels & Polkovnikov, PNAS 114, E3909 (2017) - AGP approximation
- arXiv:2601.05973 - This work (CD-ADAPT-VQE)
"""

import numpy as np
from scipy.optimize import minimize
from time import time

from qiskit.quantum_info import SparsePauliOp, Statevector, commutator
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock

from .agp_pool import compute_agp_pool
from .schedule import schedule_function


class CD_ADAPT_Solver:
    """
    Counterdiabatic ADAPT-VQE Solver for molecular ground state energies.

    This class handles the full pipeline: molecule setup, Hamiltonian
    construction, initial state preparation, AGP pool generation, and
    execution of the ADAPT-VQE loop.

    Parameters
    ----------
    molecule_geometry : str
        Molecular geometry in PySCF/XYZ format.
        Example: "Li 0.0 0.0 0.0; H 0.0 0.0 1.5"
    basis : str, optional
        Basis set for the electronic structure calculation. Default: 'sto-3g'.
    active_space : dict or None, optional
        Active space specification with keys:
            - 'num_particles'         : total number of active electrons
            - 'num_spatial_orbitals'  : number of active spatial orbitals
        If None, uses the full orbital space.
    mapper_type : str, optional
        Fermion-to-qubit mapping. 'JW' for Jordan-Wigner (default),
        'Parity' for Parity mapping.

    Attributes
    ----------
    Hf : SparsePauliOp
        The target (final) molecular Hamiltonian in qubit representation.
    Hi : SparsePauliOp
        The initial Hamiltonian: a diagonal Ising-type operator whose
        ground state is the Hartree-Fock state.
    init_state : Statevector
        The Hartree-Fock initial state.
    num_qubits : int
        Number of qubits in the simulation.

    Examples
    --------
    >>> solver = CD_ADAPT_Solver(
    ...     "Li 0.0 0.0 0.0; H 0.0 0.0 1.5",
    ...     basis="sto-3g",
    ...     active_space={"num_particles": 4, "num_spatial_orbitals": 5}
    ... )
    >>> pool = solver.compute_agp_pool(l_order=1, time_points=[0.25])
    >>> result = solver.run_cd_adapt(pool, max_iterations=30)
    """

    def __init__(
        self,
        molecule_geometry: str,
        basis: str = "sto-3g",
        active_space: dict = None,
        mapper_type: str = "JW",
    ):
        self.geometry = molecule_geometry
        self.basis = basis
        self.mapper = (
            JordanWignerMapper() if mapper_type == "JW" else ParityMapper()
        )

        # --- Step 1: Run PySCF electronic structure ---
        driver = PySCFDriver(atom=self.geometry, basis=self.basis)
        self.problem = driver.run()

        # --- Step 2: Apply active space reduction (optional) ---
        if active_space:
            transformer = ActiveSpaceTransformer(
                num_electrons=active_space["num_particles"],
                num_spatial_orbitals=active_space["num_spatial_orbitals"],
            )
            self.problem = transformer.transform(self.problem)

        # --- Step 3: Build qubit Hamiltonian H_f ---
        # Include nuclear repulsion energy so energies are total energies
        identity_op = FermionicOp.one()
        nuclear_energy_op = self.problem.nuclear_repulsion_energy * identity_op
        self.hamiltonian_second_q = (
            self.problem.hamiltonian.second_q_op() + nuclear_energy_op
        )
        self.Hf = self.mapper.map(self.hamiltonian_second_q).simplify()
        self.num_qubits = self.Hf.num_qubits

        # Store energy shifts for reference
        self.nuclear_repulsion = self.problem.nuclear_repulsion_energy
        self.energy_shift = self.problem.hamiltonian.constants.get(
            "ActiveSpaceTransformer", 0
        )

        # --- Step 4: Build initial state and initial Hamiltonian H_i ---
        self._setup_initial_conditions()

    def _setup_initial_conditions(self):
        """
        Prepare the Hartree-Fock initial state and the initial Hamiltonian H_i.

        The Hartree-Fock state |psi_HF> is a computational basis state,
        uniquely identified by a bitstring (e.g. '10110100' for a 8-qubit system).

        The initial Hamiltonian H_i is constructed as a diagonal Ising model:

            H_i = sum_k  s_k * Z_k

        where s_k = +1 if qubit k is in state |0> (unoccupied orbital),
        and s_k = -1 if qubit k is in state |1> (occupied orbital).

        This ensures that |psi_HF> is the unique ground state of H_i with a
        large spectral gap, making it an ideal starting point for the
        adiabatic path from H_i to H_f.
        """
        num_spatial = self.problem.num_spatial_orbitals
        num_particles = self.problem.num_particles

        hf_circuit = HartreeFock(num_spatial, num_particles, self.mapper)
        hf_state = Statevector(hf_circuit)
        self.init_state = hf_state

        # Extract the HF bitstring from the statevector
        probs = hf_state.probabilities_dict()
        bitstring = list(probs.keys())[0]  # e.g. '10110100'

        # Build H_i = sum_k s_k * Z_k
        # Convention: Z|0>=+|0>, Z|1>=-|1>
        # To make |psi_HF> the ground state:
        #   bit=0 (|0>) -> coeff = -1  (so -Z has eigenvalue -1 for |0>)
        #   bit=1 (|1>) -> coeff = +1  (so +Z has eigenvalue -1 for |1>)
        pauli_list = []
        for k, bit in enumerate(reversed(bitstring)):  # Qiskit: little-endian
            coeff = -1.0 if bit == "0" else 1.0
            pauli_list.append(("Z", [k], coeff))

        self.Hi = SparsePauliOp.from_sparse_list(
            pauli_list, num_qubits=self.num_qubits
        ).simplify()

    def compute_agp_pool(
        self,
        l_order: int = 1,
        T_duration: float = 1.0,
        time_points: list = None,
    ) -> list:
        """
        Generate the CD-ADAPT-VQE operator pool using the AGP approximation.

        Delegates to cd_adapt_vqe.agp_pool.compute_agp_pool with the
        molecule's H_i and H_f already set up.

        Parameters
        ----------
        l_order : int, optional
            Order of the AGP approximation. Default: 1.
        T_duration : float, optional
            Total adiabatic evolution time (for schedule evaluation). Default: 1.0.
        time_points : list of float, optional
            Time points for pool construction. Default: [0.25].

        Returns
        -------
        list of SparsePauliOp
            The operator pool (unique Pauli strings from AGP basis operators).

        See Also
        --------
        cd_adapt_vqe.agp_pool.compute_agp_pool : Full documentation.
        """
        if time_points is None:
            time_points = [0.25]
        return compute_agp_pool(
            self.Hi,
            self.Hf,
            l_order=l_order,
            T_duration=T_duration,
            time_points=time_points,
        )

    def _evolve_state(
        self,
        ops: list,
        params: list,
        initial_state: Statevector,
    ) -> Statevector:
        """
        Apply the parameterized ansatz circuit to the initial state.

        Computes:
            |psi(theta)> = exp(-i*theta_n*G_n) ... exp(-i*theta_1*G_1) |psi_0>

        Each unitary is applied via the exact formula for Pauli operators:
            exp(-i*theta*G) |psi> = cos(theta)*|psi> - i*sin(theta)*G|psi>

        This is exact (no Trotterization) because each G_k is a single
        Pauli string and thus squares to the identity.

        Parameters
        ----------
        ops : list of SparsePauliOp
            Ansatz operators G_1, ..., G_n in order of application.
        params : list of float
            Variational angles theta_1, ..., theta_n.
        initial_state : Statevector
            The reference state (Hartree-Fock).

        Returns
        -------
        Statevector
            The evolved state |psi(theta)>.
        """
        state = initial_state
        for op, theta in zip(ops, params):
            state = np.cos(theta) * state - 1j * np.sin(theta) * state.evolve(op)
        return state

    def _cost_function(
        self,
        theta_params: np.ndarray,
        current_ansatz_ops: list,
        initial_state: Statevector,
    ) -> float:
        """
        VQE cost function: energy expectation value E(theta).

        Evaluates <psi(theta)|H_f|psi(theta)> for use by the classical optimizer.

        Parameters
        ----------
        theta_params : np.ndarray
            Current variational parameters (flattened 1D array).
        current_ansatz_ops : list of SparsePauliOp
            Current set of ansatz operators.
        initial_state : Statevector
            Reference Hartree-Fock state.

        Returns
        -------
        float
            Energy expectation value in Hartree.
        """
        state = self._evolve_state(current_ansatz_ops, theta_params, initial_state)
        return state.expectation_value(self.Hf).real

    def exact_energy(self) -> float:
        """
        Compute the exact ground state energy by full diagonalization.

        Constructs the full 2^N x 2^N Hamiltonian matrix and finds the
        minimum eigenvalue. Used as the reference for benchmarking.

        Returns
        -------
        float
            Ground state energy in Hartree.

        Warning
        -------
        This scales exponentially with system size and is only feasible
        for small molecules (N <= ~20 qubits).
        """
        eigvals = np.linalg.eigvalsh(self.Hf.to_matrix())
        return float(eigvals[0])

    def run_cd_adapt(
        self,
        agp_pool: list,
        max_iterations: int = 30,
        gradient_threshold: float = 1e-2,
        optimizer: str = "L-BFGS-B",
        verbose: bool = True,
    ) -> dict:
        """
        Run the main CD-ADAPT-VQE optimization loop.

        Algorithm Steps (per iteration)
        --------------------------------
        1. **Gradient measurement**: For each pool operator G_k, compute
               gradient_k = |<psi| [H_f, G_k] |psi>|
           The commutator [H_f, G_k] is pre-computed once outside the loop.

        2. **Convergence check**: If the L2 norm of all gradients falls
           below `gradient_threshold`, stop.

        3. **Operator selection**: Append the operator G_k* with the
           largest gradient to the ansatz.

        4. **VQE re-optimization**: Use L-BFGS-B to minimize E(theta) over
           ALL current ansatz parameters (not just the new one).

        5. **State update**: Re-evolve the state with optimized parameters
           for gradient computation in the next iteration.

        Parameters
        ----------
        agp_pool : list of SparsePauliOp
            Operator pool from compute_agp_pool().
        max_iterations : int, optional
            Maximum number of ADAPT steps. Default: 30.
        gradient_threshold : float, optional
            Convergence threshold on the L2 norm of pool gradients. Default: 1e-2.
        optimizer : str, optional
            Scipy optimizer for the VQE subroutine. Default: 'L-BFGS-B'.
        verbose : bool, optional
            Print iteration progress. Default: True.

        Returns
        -------
        dict with keys:
            - 'energies'     : list of float  - VQE energy after each ADAPT step
            - 'gradients'    : list of float  - Max pool gradient per iteration
            - 'final_params' : list of float  - Final optimized theta values
            - 'num_ops'      : int            - Number of operators in final ansatz
            - 'ansatz_ops'   : list of str    - Pauli labels of selected operators

        Examples
        --------
        >>> result = solver.run_cd_adapt(pool, max_iterations=30, gradient_threshold=1e-3)
        >>> print(f"Final energy: {result['energies'][-1]:.6f} Ha")
        >>> print(f"Ansatz size:  {result['num_ops']} operators")
        >>> print(f"Operators:    {result['ansatz_ops']}")
        """
        ansatz_ops = []
        ansatz_params = []
        energies_history = []
        gradients_history = []

        current_state = self.init_state

        if verbose:
            print(f"\n{'='*60}")
            print(f"CD-ADAPT-VQE: {self.geometry}")
            print(f"Qubits: {self.num_qubits}  |  Pool size: {len(agp_pool)}")
            print(f"{'='*60}")

        # Pre-compute [H_f, G_k] for all pool operators (done once)
        if verbose:
            print("Pre-computing pool commutators [H_f, G_k]...")
        pool_commutators = [
            commutator(self.Hf, op).simplify() for op in agp_pool
        ]

        for n_iter in range(max_iterations):

            # --- Step 1: Measure gradients ---
            pool_gradients = np.array([
                abs(current_state.expectation_value(comm))
                for comm in pool_commutators
            ])

            max_grad_idx = int(np.argmax(pool_gradients))
            max_grad_val = float(pool_gradients[max_grad_idx])
            gradients_history.append(max_grad_val)

            if verbose:
                selected_label = agp_pool[max_grad_idx].to_list()[0][0]
                print(
                    f"Iter {n_iter+1:3d} | "
                    f"Max grad = {max_grad_val:.6f} | "
                    f"Selected: {selected_label}"
                )

            # --- Step 2: Convergence check ---
            if np.linalg.norm(pool_gradients) < gradient_threshold:
                if verbose:
                    print(f"Converged: gradient norm < {gradient_threshold}")
                break

            # --- Step 3: Select and append operator ---
            ansatz_ops.append(agp_pool[max_grad_idx])
            ansatz_params.append(0.0)

            # --- Step 4: VQE re-optimization ---
            result = minimize(
                self._cost_function,
                x0=np.array(ansatz_params),
                args=(ansatz_ops, self.init_state),
                method=optimizer,
                options={"disp": False, "maxiter": 1000},
            )

            ansatz_params = list(result.x)
            current_energy = float(result.fun)
            energies_history.append(current_energy)

            if verbose:
                print(f"           | VQE energy   = {current_energy:.8f} Ha")

            # --- Step 5: Update current state for next gradient measurement ---
            current_state = self._evolve_state(
                ansatz_ops, ansatz_params, self.init_state
            )

        if verbose:
            print(f"\nFinal ansatz: {len(ansatz_ops)} operators")
            print(f"Final energy: {energies_history[-1]:.8f} Ha")

        return {
            "energies": energies_history,
            "gradients": gradients_history,
            "final_params": ansatz_params,
            "num_ops": len(ansatz_ops),
            "ansatz_ops": [op.to_list()[0][0] for op in ansatz_ops],
        }
