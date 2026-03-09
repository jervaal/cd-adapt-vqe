import numpy as np
import pandas as pd
import time

# --- Importaciones de Qiskit ---
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from qiskit_algorithms.optimizers import SLSQP, L_BFGS_B
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit.primitives import Estimator
from qiskit import transpile  # <--- Importante para la métrica
from qiskit

# --- Configuración Global ---
ESTIMATOR = Estimator()
MAPPER = JordanWignerMapper()

# Definimos las puertas base para que el conteo sea realista y estandarizado
from qiskit.providers.fake_provider import GenericBackendV2

# --- CONFIGURACIÓN DE HARDWARE ---
backend = GenericBackendV2(10)
basis_gates_opt = backend._basis_gates

BASIS_GATES = ['u', 'cx', 'rz', 'h', 'sx', 'x', 'measure', 'barrier', 'reset']

def calculate_energies(molecule_name, atoms, distance, active_space_dict):
    """
    Calcula energías y métricas del circuito final transpilado.
    """
    # 1. Geometría
    atom_coords = [(0.0, 0.0, 0.0), (0.0, 0.0, distance)]
    info = MoleculeInfo(atoms, atom_coords)

    # 2. Driver y Problema
    driver = PySCFDriver.from_molecule(info, basis="6-31g")
    full_problem = driver.run()

    # 3. Espacio Activo
    act_space_tf = ActiveSpaceTransformer(**active_space_dict)
    problem = act_space_tf.transform(full_problem)

    # 4. Hamiltoniano y Constantes
    main_op = problem.hamiltonian.second_q_op()
    energy_shift = (problem.nuclear_repulsion_energy + 
                    problem.hamiltonian.constants.get('ActiveSpaceTransformer', 0.0))
    qubit_op = MAPPER.map(main_op)

    # 5. Estado Inicial
    init_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        MAPPER,
    )

    # 6. Energía Exacta (Referencia)
    matrix = qubit_op.to_matrix()
    eigvals = np.linalg.eigvals(matrix)
    energy_exact = np.real(np.min(eigvals)) + energy_shift

    # 7. VQE Estándar (Referencia)
    optimizer_vqe = L_BFGS_B(maxiter=250)
    vqe_ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        MAPPER,
        initial_state=init_state
    )
    vqe = VQE(ESTIMATOR, vqe_ansatz, optimizer_vqe)
    result_vqe = vqe.compute_minimum_eigenvalue(qubit_op)
    energy_vqe = result_vqe.eigenvalue.real + energy_shift

    # 8. AdaptVQE
    print(f"  Ejecutando AdaptVQE para r={distance}...")
    optimizer_adapt = L_BFGS_B(maxiter=250)
    
    # Nota: AdaptVQE construye su propio ansatz iterativamente, 
    # pero necesita un ansatz base para definir el pool de operadores (usualmente UCCSD)
    adapt_ansatz_base = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        MAPPER,
        initial_state=init_state
    )
    
    vqe_for_adapt = VQE(ESTIMATOR, adapt_ansatz_base, optimizer_adapt)
    
    # threshold reducido un poco para asegurar convergencia rápida en prueba
    adapt_vqe = AdaptVQE(vqe_for_adapt, gradient_threshold=0.01) 
    result_adapt = adapt_vqe.compute_minimum_eigenvalue(qubit_op)
    energy_adapt_vqe = result_adapt.eigenvalue.real + energy_shift

    # --- MÉTRICAS DEL CIRCUITO (TRANSPILACIÓN) ---
    # Obtenemos el circuito óptimo abstracto
    opt_circuit_abstract = result_adapt.optimal_circuit
    # Obtenemos los parámetros óptimos encontrados
    opt_params = result_adapt.optimal_point
    
    # Vinculamos los parámetros para tener un circuito concreto (con números, no variables)
    bound_circuit = opt_circuit_abstract.bind_parameters(opt_params)
    
    # Transpilamos a puertas físicas (igual que en tu código anterior)
    qc_transpiled = transpile(bound_circuit, basis_gates=BASIS_GATES, optimization_level=3)
    
    # Extraemos métricas
    transpiled_depth = qc_transpiled.depth()
    transpiled_cx = qc_transpiled.count_ops().get('cx', 0)
    total_ops = qc_transpiled.count_ops() # Diccionario completo por si acaso

    return {
        "Molecule": molecule_name,
        "Distance_A": distance,
        "Exact_Energy": energy_exact,
        "VQE_Energy": energy_vqe,
        "AdaptVQE_Energy": energy_adapt_vqe,
        "Num_Parameters": result_adapt.optimal_circuit.num_parameters, # Params libres originales
        "Transpiled_CX": transpiled_cx,          # Conteo CNOT real
        "Transpiled_Depth": transpiled_depth,    # Profundidad real
        "Error_Adapt": abs(energy_adapt_vqe - energy_exact)
    }

def main():
    # Configuración de simulación
    molecule_configs = {
        "HF": {
            "atoms": ["F", "H"],
            "active_space": {'num_electrons': 4, 'num_spatial_orbitals': 5},
            "distances": list(np.sort(np.append(np.linspace(0.6, 2, 5), 0.917))) # Reduje a 5 ptos para prueba rápida
        },
    }

    results_list = []

    print("--- Iniciando Simulación Adapt-VQE con Métricas de Transpilación ---")

    for name, config in molecule_configs.items():
        for d in config["distances"]:
            try:
                data = calculate_energies(name, config["atoms"], d, config["active_space"])
                results_list.append(data)
                print(f"  -> Terminado r={d:.3f} | CX: {data['Transpiled_CX']} | Profundidad: {data['Transpiled_Depth']}")
            except Exception as e:
                print(f"  ERROR en r={d}: {e}")
                results_list.append({
                    "Molecule": name,
                    "Distance_A": d,
                    "Exact_Energy": "ERROR",
                    "VQE_Energy": str(e),
                    "AdaptVQE_Energy": "",
                    "Num_Parameters": 0,
                    "Transpiled_CX": 0,
                    "Transpiled_Depth": 0,
                    "Error_Adapt": 0
                })

    # Crear DataFrame y exportar a Excel
    if results_list:
        df = pd.DataFrame(results_list)
        output_filename = "vqe_results_workstation_HF_transpiled.xlsx"
        
        # Ordenamos las columnas para mejor lectura
        cols = ["Molecule", "Distance_A", "Exact_Energy", "AdaptVQE_Energy", "Error_Adapt", "Transpiled_CX", "Transpiled_Depth", "Num_Parameters"]
        # Aseguramos que existan todas las columnas antes de filtrar
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]

        df.to_excel(output_filename, index=False) # engine openpyxl es default en pandas nuevos
        print(f"\nResultados guardados en: {output_filename}")
    else:
        print("No se generaron resultados.")

if __name__ == "__main__":
    main()
