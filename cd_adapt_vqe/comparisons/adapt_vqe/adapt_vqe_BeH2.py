import numpy as np
import csv
import time

# --- Importaciones de Qiskit ---
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit.primitives import Estimator

# --- Constantes Globales ---
ESTIMATOR = Estimator()
IDENTITY_OP = FermionicOp.one()
MAPPER = JordanWignerMapper()

def calculate_energies(molecule_name, atoms, distance, active_space_dict):
    """
    Calcula las energías VQE, AdaptVQE y Exacta para una molécula dada
    a una distancia específica.
    """
    print(f"  Calculando {molecule_name} a {distance:.4f} Å...")
    
    # 1. Definir la geometría de la molécula
    atom_coords = [(0.0, 0.0, 0.0), (0.0, 0.0, distance), (0.0, 0.0, -distance)]
    info = MoleculeInfo(atoms, atom_coords)

    # 2. Ejecutar PySCF para obtener el problema completo
    try:
        driver = PySCFDriver.from_molecule(info, basis="sto-3g")
        full_problem = driver.run()
    except Exception as e:
        print(f"    ERROR en PySCFDriver para {molecule_name} a {distance} Å: {e}")
        return np.nan, np.nan, np.nan

    # 3. Aplicar el transformador de espacio activo
    try:
        act_space_tf = ActiveSpaceTransformer(**active_space_dict)
        problem = act_space_tf.transform(full_problem)
    except Exception as e:
        print(f"    ERROR en ActiveSpaceTransformer para {molecule_name} a {distance} Å: {e}")
        return np.nan, np.nan, np.nan

    # 4. Construir el Hamiltoniano de Qubit
    ast_energy_op = problem.hamiltonian.constants.get('ActiveSpaceTransformer', 0.0) * IDENTITY_OP
    hamiltonian_fermionic = problem.hamiltonian.second_q_op()
    nuclear_energy_op = problem.nuclear_repulsion_energy * IDENTITY_OP
    total_fermionic_op = hamiltonian_fermionic + nuclear_energy_op + ast_energy_op
    qubit_op = MAPPER.map(total_fermionic_op)

    # 5. Configurar el estado inicial (Hartree-Fock)
    try:
        init_state = HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            MAPPER,
        )
    except ValueError as e:
        print(f"    ERROR creando HartreeFock (prob. espacio activo incorrecto): {e}")
        return np.nan, np.nan, np.nan

    # --- 6. (NUEVO) Calcular Energía Exacta (Diagonalización) ---
    energy_exact = np.nan
    try:
        print("    Calculando energía exacta (numpy)...")
        # qubit_op.to_matrix() crea la matriz 2^N x 2^N
        matrix = qubit_op.to_matrix()
        # np.linalg.eigvals() calcula todos los valores propios
        eigval = np.linalg.eigvals(matrix)
        # El estado fundamental es el valor propio mínimo.
        # Usamos np.real() para descartar ruido numérico complejo.
        energy_exact = np.real(np.min(eigval))
        print(f"    Exacta: {energy_exact:.8f} Hartree")
    except Exception as e:
        # ¡IMPORTANTE! Esto fallará si la molécula es muy grande (p.ej. > 16 qubits)
        # debido a que la matriz no cabe en la memoria.
        print(f"    ADVERTENCIA: No se pudo calcular la energía exacta (matriz demasiado grande): {e}")


    # Inicializar energías en caso de que fallen
    energy_vqe = np.nan
    energy_adapt_vqe = np.nan

    # --- 7. Ejecutar VQE estándar ---
    try:
        optimizer_vqe = SLSQP(maxiter=10000, ftol=1e-9)
        vqe_ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            MAPPER,
            initial_state=init_state
        )
        
        vqe = VQE(ESTIMATOR, vqe_ansatz, optimizer_vqe)
        vqe.initial_point = [0] * vqe_ansatz.num_parameters
        
        algo_vqe = GroundStateEigensolver(MAPPER, vqe)
        result_vqe = algo_vqe.solve(problem)
        energy_vqe = result_vqe.eigenvalues[0]
        print(f"    VQE: {energy_vqe:.8f} Hartree")

    except Exception as e:
        print(f"    ERROR durante VQE: {e}")

    # --- 8. Ejecutar AdaptVQE ---
    try:
        optimizer_adapt = SLSQP(maxiter=10000, ftol=1e-9)
        adapt_vqe_ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            MAPPER,
            initial_state=init_state
        )
        
        vqe_instance_for_adapt = VQE(ESTIMATOR, adapt_vqe_ansatz, optimizer_adapt)
        adapt_vqe = AdaptVQE(vqe_instance_for_adapt, gradient_threshold=1e-02)
        result_adapt_vqe = adapt_vqe.compute_minimum_eigenvalue(qubit_op)
        energy_adapt_vqe = result_adapt_vqe.eigenvalue
        print(f"    AdaptVQE: {energy_adapt_vqe:.8f} Hartree")

    except Exception as e:
        print(f"    ERROR durante AdaptVQE: {e}")

    # Devolvemos los tres valores
    return energy_vqe, energy_adapt_vqe, energy_exact

# --- Bucle Principal de Simulación ---

def main():
    # 1. Definir las moléculas, sus espacios activos Y SUS DISTANCIAS
    molecule_configs = {
        "HF": {
            "atoms": ["Be", "H", "H"],
            "active_space": {'num_electrons': 4, 'num_spatial_orbitals': 5},
            "distances": np.linspace(0.6, 2, 15) 
        }
        
    }

    # 2. Preparar el archivo CSV
    output_filename = "vqe_adaptvqeBeH2.csv"
    results_data = []
    
    # Escribir la cabecera (¡ACTUALIZADA!)
    header = ["Molecule", "Distance_A", "Exact_Energy_Hartree", "VQE_Energy_Hartree", "AdaptVQE_Energy_Hartree"]
    results_data.append(header)

    print(f"Iniciando escaneo de energía. Resultados se guardarán en {output_filename}")
    start_time = time.time()

    # 3. Ejecutar el bucle
    for molecule_name, config in molecule_configs.items():
        print(f"\n--- Procesando Molécula: {molecule_name} ---")
        
        distances_for_this_molecule = config["distances"]
        
        for dist in distances_for_this_molecule:
            
            # Recibimos los tres valores (¡ACTUALIZADO!)
            e_vqe, e_adapt, e_exact = calculate_energies(
                molecule_name,
                config["atoms"],
                dist,
                config["active_space"]
            )
            
            # Añadir los resultados a la lista (¡ACTUALIZADO!)
            results_data.append([molecule_name, f"{dist:.4f}", e_exact, e_vqe, e_adapt])
            

    # 4. Escribir todos los resultados al archivo CSV
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(results_data)
        
        end_time = time.time()
        print("\n--- ¡Simulación Completada! ---")
        print(f"Resultados guardados en: {output_filename}")
        print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")

    except IOError as e:
        print(f"Error fatal al escribir el archivo CSV: {e}")

# Ejecutar el script
if __name__ == "__main__":
    main()
