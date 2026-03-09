import numpy as np
import time
import pandas as pd  # <--- NUEVO: Para manejar Excel

# --- Importaciones de Qiskit ---
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_algorithms.optimizers import SLSQP, L_BFGS_B
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
    atom_coords = [(0.0, 0.0, 0.0), (0.0, 0.0, distance)]
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

    # --- 6. Calcular Energía Exacta (Diagonalización) ---
    energy_exact = np.nan
    try:
        # qubit_op.to_matrix() crea la matriz 2^N x 2^N
        matrix = qubit_op.to_matrix()
        eigval = np.linalg.eigvals(matrix)
        energy_exact = np.real(np.min(eigval))
        print(f"    Exacta: {energy_exact:.8f} Hartree")
    except Exception as e:
        print(f"    ADVERTENCIA: No se pudo calcular la energía exacta: {e}")


    # Inicializar energías en caso de que fallen
    energy_vqe = np.nan
    energy_adapt_vqe = np.nan

    # --- 7. Ejecutar VQE estándar ---
    try:
        optimizer_vqe = L_BFGS_B(maxiter=500, ftol=1e-9)
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
        optimizer_adapt = L_BFGS_B(maxiter=500, ftol=1e-9)
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
    # 1. Definir las moléculas
    distancias = list(np.linspace(0.8, 1.8, 10))
    distancias.append(1.5)
    
    molecule_configs = {
        "LiH": {
            "atoms": ["Li", "H"],
            "active_space": {'num_electrons': 4, 'num_spatial_orbitals': 5},
            "distances": sorted(distancias)
        },
    }

    # 2. Preparar nombre del archivo
    output_filename = "vqe_adaptvqeLiH.xlsx" # <--- Extensión .xlsx
    
    # Usaremos una lista para guardar diccionarios de resultados
    results_list = []

    print(f"Iniciando escaneo de energía. Resultados se guardarán en {output_filename}")
    start_time = time.time()

    # 3. Ejecutar el bucle
    for molecule_name, config in molecule_configs.items():
        print(f"\n--- Procesando Molécula: {molecule_name} ---")
        
        distances_for_this_molecule = config["distances"]
        
        for dist in distances_for_this_molecule:
            
            e_vqe, e_adapt, e_exact = calculate_energies(
                molecule_name,
                config["atoms"],
                dist,
                config["active_space"]
            )
            
            # --- NUEVO: Guardamos como diccionario ---
            # Esto facilita mucho la creación del Excel después
            row_data = {
                "Molecule": molecule_name,
                "Distance_A": dist,  # Guardamos como número para poder graficar en Excel
                "Exact_Energy_Hartree": e_exact,
                "VQE_Energy_Hartree": e_vqe,
                "AdaptVQE_Energy_Hartree": e_adapt,
                # Opcional: Calcular error ahí mismo si quieres
                "Error_Adapt": abs(e_adapt - e_exact) if (e_adapt and e_exact) else np.nan
            }
            results_list.append(row_data)

    # 4. Guardar en Excel usando Pandas
    try:
        print("\n--- Guardando datos en Excel... ---")
        
        # Convertimos la lista de diccionarios a un DataFrame
        df = pd.DataFrame(results_list)
        
        # Guardamos en Excel. index=False evita que guarde el número de fila (0, 1, 2...)
        df.to_excel(output_filename, index=False)
        
        end_time = time.time()
        print("\n--- ¡Simulación Completada! ---")
        print(f"Resultados guardados exitosamente en: {output_filename}")
        print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")

        # Mostramos una vista previa de los primeros datos
        print("\nVista previa de los datos:")
        print(df.head())

    except Exception as e:
        print(f"Error fatal al escribir el archivo Excel: {e}")
        # Si falla Excel, intentamos guardar un respaldo en CSV
        try:
            csv_backup = output_filename.replace(".xlsx", "_backup.csv")
            df.to_csv(csv_backup, index=False)
            print(f"Se ha guardado un respaldo en CSV: {csv_backup}")
        except:
            pass

# Ejecutar el script
if __name__ == "__main__":
    main()
