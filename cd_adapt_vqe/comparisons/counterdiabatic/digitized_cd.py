import cluster_code_adapt_cd as ccacd
import numpy as np
import time
import pandas as pd
from tqdm import tqdm

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HamiltonianGate
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit import transpile

# --- CONFIGURACIONES GLOBALES ---
basis_gates = ['u', 'cx', 'rz', 'h', 'sx', 'x', 'measure', 'barrier', 'reset']
mapper = JordanWignerMapper()

# --- DEFINICIÓN DE MOLÉCULAS ---
molecule_configs = {
    "LiH": {
        "atoms": ["Li", "H"],
        # En STO-3G, LiH tiene exactamente 6 orbitales espaciales.
        # 4 electrones en 6 orbitales = 12 Qubits.
        "active_space": {'num_electrons': 4, 'num_spatial_orbitals': 5},
        "distances": list(np.linspace(0.8, 1.8, 10)) 
    }
}

# Parámetros del algoritmo
l_values = [1, 2]
T = 1.0
N = 10
dt = T / N

# --- BUCLE PRINCIPAL SERIAL ---
results_list = []

print(f"--- Iniciando Simulación LiH (Base STO-3G) ---")
global_start = time.time()

for mol_name, config in molecule_configs.items():
    atoms = config["atoms"]
    active_space = config["active_space"]
    distances = config["distances"]
    
    print(f"\nConfiguración: {mol_name} | Electrones: {active_space['num_electrons']} | Orbitales: {active_space['num_spatial_orbitals']}")
    
    # 1. Configuración Estática
    num_particles = (active_space['num_electrons'] // 2, active_space['num_electrons'] // 2)
    
    # Circuito Base HF
    hf_circuit_base = HartreeFock(
        num_spatial_orbitals=active_space['num_spatial_orbitals'], 
        num_particles=num_particles, 
        qubit_mapper=mapper
    )
    
    # Hamiltoniano Inicial (Hi)
    print("Calculando Hamiltoniano Inicial (Hi)...")
    hf_bitstring = ccacd.HartreeFock_bitstring_init_state(
        active_space_dict={'num_particles': num_particles, 
                           'num_spatial_orbitals': active_space['num_spatial_orbitals']}
    )
    Hi_qutip = ccacd.HartreeFock_GroundState_Hamiltonian(hf_bitstring)
    Hi_pauli = ccacd.qutip_to_sparse_pauli_op(Hi_qutip)
    
    # 2. Bucle sobre Distancias
    for r in tqdm(distances, desc="Procesando Distancias"):
        
        try:
            # Definición geométrica
            mol_string = f'{atoms[0]} 0 0 0 ; {atoms[1]} 0 0 {r}'
            
            # --- CAMBIO AQUÍ: basis='sto-3g' ---
            Hf_qutip = ccacd.molecular_hamiltonian(
                molecule=mol_string, 
                active_space_dict=active_space, 
                basis='sto-3g'  # <--- Base cambiada
            )
            Hf_pauli = ccacd.qutip_to_sparse_pauli_op(Hf_qutip)
            exact_energy = Hf_qutip.eigenenergies()[0]
            
            # 3. Bucle sobre orden l
            for l in l_values:
                step_start = time.time()
                
                # Copia circuito
                qc = hf_circuit_base.copy()
                
                # --- Evolución Temporal ---
                for j in range(N):
                    tj = j * dt
                    lamda_tj = ccacd.schedule_function(tj, T)
                    
                    Hcd_qutip = ccacd.CounterDiabatic_Hamiltonian(Hi_qutip, Hf_qutip, tj, T, l)
                    Hcd_pauli = ccacd.qutip_to_sparse_pauli_op(Hcd_qutip)
                    
                    Ht_pauli = (1 - lamda_tj) * Hi_pauli + lamda_tj * Hf_pauli + Hcd_pauli
                    Ht_pauli = Ht_pauli.simplify()
                    
                    evo_gate = HamiltonianGate(Ht_pauli, time=dt)
                    qc.append(evo_gate, qc.qubits)
                
                # --- TRANSPILACIÓN Y MÉTRICAS ---
                qc_transpiled = transpile(qc, basis_gates=basis_gates, optimization_level=3)
                
                depth_val = qc_transpiled.depth()
                cx_val = qc_transpiled.count_ops().get('cx', 0)
                
                # --- Energía Final ---
                final_sv = Statevector(qc)
                final_e = np.real(final_sv.expectation_value(Hf_pauli))
                
                elapsed = time.time() - step_start
                
                # Guardar datos
                res = {
                    "Molecule": mol_name,
                    "r": r,
                    "l": l,
                    "Exact_Energy": exact_energy,
                    "CD_Energy": final_e,
                    "Error": abs(final_e - exact_energy),
                    "Time_s": elapsed,
                    "Transpiled_Depth": depth_val,
                    "Transpiled_CX": cx_val
                }
                
                results_list.append(res)
                
        except Exception as e:
            print(f"    ERROR CRÍTICO en r={r}: {e}")

# --- GUARDAR RESULTADOS ---
if results_list:
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by=["Molecule", "r", "l"])
    
    print("\n" + "="*50)
    print("RESULTADOS FINALES (STO-3G)")
    print("="*50)
    print(df_results[["r", "l", "Error", "Transpiled_Depth", "Transpiled_CX"]].head(6))
    
    filename = "resultados_LiH_STO3G_series.xlsx"
    df_results.to_excel(filename, index=False)
    print(f"\nDatos guardados exitosamente en '{filename}'")
else:
    print("No se generaron resultados.")
