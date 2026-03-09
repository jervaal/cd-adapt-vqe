import cluster_code_adapt_cd as ccacd
import qutip as qt
from qutip import Qobj, sesolve
import numpy as np
import pandas as pd
import os
import time

# --- 1. Funciones de Ayuda (Definidas Globalmente) ---

def conmutador_anidado(H: qt.Qobj, aH: qt.Qobj, n: int) -> qt.Qobj:
    """
    Calcula el conmutador anidado [H, [H, ..., aH]...] n veces.
    """
    if not all(isinstance(op, qt.Qobj) for op in [H, aH]):
        raise TypeError("H y aH deben ser objetos de QuTiP.")
    if not isinstance(n, int):
        raise TypeError("n debe ser un entero.")
    
    comm = aH
    for _ in range(n):
        comm = qt.commutator(H, comm)
    return comm

def Lambda_t(t, tf):
    """
    Función de scheduling (rampa) de 0 a 1.
    """
    # Tu función de rampa personalizada
    lambda_t = np.sin(np.pi/2 * np.sin(np.pi*t / (2*tf))**2)**2
    # Asegurémonos de que sea numéricamente estable y esté en [0, 1]
    return np.clip(lambda_t, 0.0, 1.0)

def Lambda_tt(t, tf):
    """
    Derivada exacta de la función de scheduling.
    """
    # Definimos el argumento interno para legibilidad
    u = (np.pi * t) / (2 * tf)
    
    # Componentes de la regla de la cadena
    inner_sin_sq = np.sin(u)**2
    arg_outer = (np.pi / 2) * inner_sin_sq
    
    # La derivada simplificada es:
    # (pi^2 / (2 * tf)) * sin(pi*t/tf) * sin(arg_outer) * cos(arg_outer)
    # O usando de nuevo sin(2x) = 2sin(x)cos(x) para el término externo:
    
    term_a = (np.pi**2 / (4 * tf)) 
    term_b = np.sin(np.pi * t / tf) # Esto equivale a 2 * sin(u) * cos(u)
    term_c = np.sin(2 * arg_outer)  # Esto equivale a 2 * sin(arg_outer) * cos(arg_outer)
    
    lambda_dot = term_a * term_b * term_c
    
    return lambda_dot

def HAD(Hi, Hf, lambdat):
    """
    Hamiltoniano Adiabático H(t) = (1-lambda)Hi + lambda*Hf
    """
    return (1.0 - lambdat) * Hi + lambdat * Hf

def Alphas(Had: qt.Qobj, aH: qt.Qobj, l: int) -> np.ndarray:
    """
    Calcula los coeficientes alpha_k resolviendo el sistema lineal.
    
    Args:
        Had (qt.Qobj): Hamiltoniano adiabático H(lambda(t)).
        aH (qt.Qobj): Derivada dH/dlambda = Hf - Hi.
        l (int): Orden de la aproximación.
    """
    Gamma = np.zeros([l, l])
    Gammab = np.zeros(l)
    
    # Pre-calcular conmutadores para eficiencia
    # Calculamos hasta 2*l
    commutators = [conmutador_anidado(Had, aH, k + 1) for k in range(2 * l)]
    norms_sq = [c.norm('fro')**2 for c in commutators]
    
    for j in range(l):
        Gammab[j] = norms_sq[j] # Gammab[j] = ||O_{j+1}||^2
        for k in range(l):
            # Gamma[j,k] = ||O_{j+k+2}||^2
            Gamma[j, k] = norms_sq[j + k + 1] 
            
    det = np.linalg.det(Gamma)
    
    if abs(det) < 1e-12:
        # Si la matriz es singular, usa mínimos cuadrados
        alphas, _, _, _ = np.linalg.lstsq(Gamma, -Gammab, rcond=None) 
    else:
        alphas = np.linalg.solve(Gamma, -Gammab)  
    return alphas    

def HCD(t, args):
    """
    Hamiltoniano completo H_STA(t) = H_ad(t) + H_cd(t)
    Esta función es la que se pasa a sesolve.
    """
    # Extrae todos los parámetros de 'args'
    Hi = args["Hi"]
    Hf = args["Hf"]
    aH = args["aH"]
    l = args["l"]
    T = args["T"]
    
    # Calcula los valores dependientes del tiempo
    lambdat = Lambda_t(t, T)
    lambdatt = Lambda_tt(t, T) 
    
    # Si estamos exactamente al inicio o al final, la derivada es 0
    if t == 0.0 or t == T:
        lambdatt = 0.0

    Had = HAD(Hi, Hf, lambdat)      
    
    Hcd = 0
    
    # Solo calcula Hcd si la derivada no es cero
    if lambdatt != 0.0:
        # Calcula los alphas (dependientes del tiempo porque Had lo es)
        try:
            alphas = Alphas(Had, aH, l)
        except np.linalg.LinAlgError:
            # Si falla el cálculo (ej. matriz singular), no aplica corrección
            alphas = np.zeros(l)

        # Construye el Hamiltoniano contradiabático
        # Hcd = sum( alpha_k * O_{2k+1} )
        for ii in range(l):
            # Para ii=0 -> 2*0+1 = 1
            # Para ii=1 -> 2*1+1 = 3
            Hcd = Hcd + conmutador_anidado(Had, aH, 2*ii + 1) * alphas[ii]
        
        # Multiplica por la derivada (y el i) al final
        Hcd = Hcd * (1j * lambdatt)
    
    return Had + Hcd

# --- 2. Configuración de la Simulación ---

# Órdenes de aproximación a probar
l_values = [1, 2, 3]

# Tiempos de evolución (puedes añadir más)
T_values = [1.0] #, 0.5, 2.0] 

# Puntos de tiempo para el solver
Nt = 1000

# Definición de las moléculas
molecule_configs = {
    "HF": {
        "atoms": ["H", "F"], # Nota: El orden no importa para la molécula
        "active_space": {'num_electrons': 4, 'num_spatial_orbitals': 5},
        "distances": np.sort(np.append(np.linspace(0.6, 1.3, 10), 0.917))
    },

}

# Lista para guardar los resultados
results_data = []
output_filename = "resultados_cd_driving_HF.csv"

# --- 3. Bucle Principal de Simulación ---

print("--- Iniciando Simulación de Counterdiabatic Driving ---")
start_time_total = time.time()

try:
    # Bucle por cada molécula (HF, LiH)
    for molecule_name, config in molecule_configs.items():
        print(f"\n=========================================")
        print(f" MOLÉCULA: {molecule_name}")
        print(f"=========================================")
        
        active_space = config["active_space"]
        atoms = config["atoms"]
        
        # --- 3a. Generar Hi y psi0 (específico de la molécula) ---
        # Convierte el formato de active_space
        n_elec = active_space['num_electrons']
        n_orbs = active_space['num_spatial_orbitals']
        n_alpha = n_elec // 2
        n_beta = n_elec - n_alpha
        active_space_v2 = {'num_particles': (2, 2), 'num_spatial_orbitals': 5}

        print(f"  Generando Hi y psi0 para {active_space_v2}...")
        
        # --- CORRECCIÓN DE LÓGICA ---
        # 1. Obtener el bitstring de Hartree-Fock (esto es un string, ej: '11')
        hf_bitstring = ccacd.HartreeFock_bitstring_init_state(active_space_dict=active_space_v2)
        print(f"  Bitstring HF generado: {hf_bitstring}")

        # 2. Generar el Hamiltoniano Inicial (Hi) usando el bitstring
        Hi = ccacd.HartreeFock_GroundState_Hamiltonian(hf_bitstring)
        
        # 3. Encontrar el estado inicial (psi0) calculando el groundstate de Hi
        print(f"  Buscando groundstate de Hi...")
        try:
            _, psi0 = Hi.groundstate()
        except Exception as e:
            print(f"    ERROR: No se pudo calcular el groundstate de Hi. Saltando molécula. Error: {e}")
            continue # Salta al siguiente 'molecule_name'
            
        # 4. Verificación
        if not isinstance(psi0, qt.Qobj):
             print(f"    ERROR: El groundstate de Hi no es un Qobj. Saltando molécula.")
             continue # Salta al siguiente 'molecule_name'
        # --- FIN DE LA CORRECCIÓN ---
        
        # Bucle por cada distancia interatómica
        for r in config["distances"]:
            print(f"\n  --- Procesando r = {r} Å ---")
            
            # --- 3b. Generar Hf (específico de la distancia) ---
            mol_string = f'{atoms[0]} 0 0 0 ; {atoms[1]} 0 0 {r}'
            
            try:
                Hf = ccacd.molecular_hamiltonian(molecule=mol_string, active_space_dict=active_space, basis= '6-31g')
                exact_energy = Hf.eigenenergies()[0]
                aH = Hf - Hi # dH/dlambda
            except Exception as e:
                print(f"    ERROR: No se pudo generar Hf para r={r}. Saltando. Error: {e}")
                continue

            # Bucle por cada tiempo de evolución T
            for T in T_values:
            
                # Bucle por cada orden de aproximación l
                for l in l_values:
                    
                    start_time_sim = time.time()
                    print(f"    Iniciando simulación (l={l}, T={T})...", end="")
                    
                    tlist = np.linspace(0, T, Nt)        
                    
                    # --- 3c. Ejecutar la simulación ---
                    sesolve_args = {
                        "Hi": Hi, 
                        "Hf": Hf, 
                        "aH": aH, 
                        "l": l, 
                        "T": T
                    }
                    
                    # Opciones para el solver (más pasos para estabilidad)
                    options = qt.Options(nsteps=5000, atol=1e-8, rtol=1e-6)
                    
                    try:
                        result = sesolve(HCD, psi0, tlist, args=sesolve_args, options=options)
                        
                        final_state = result.states[-1]
                        Ef_CD = qt.expect(Hf, final_state)
                        fidelidad = qt.fidelity(Hf.groundstate()[1], final_state)**2
                        
                        sim_time = time.time() - start_time_sim
                        print(f" Completa ({sim_time:.2f}s). E_final = {Ef_CD:.6f}")

                        # --- 3d. Guardar resultados ---
                        results_data.append({
                            "Molecule": molecule_name,
                            "r": r,
                            "l": l,
                            "T": T,
                            "Exact_Energy": exact_energy,
                            "CD_Energy": Ef_CD,
                            "Abs_Error": abs(Ef_CD - exact_energy),
                            "Fidelity": fidelidad,
                            "Sim_Time_s": sim_time
                        })
                        
                    except Exception as e:
                        print(f" FALLÓ. Error en sesolve: {e}")
                        results_data.append({
                            "Molecule": molecule_name,
                            "r": r,
                            "l": l,
                            "T": T,
                            "Exact_Energy": exact_energy,
                            "CD_Energy": np.nan,
                            "Abs_Error": np.nan,
                            "Fidelity": np.nan,
                            "Sim_Time_s": np.nan
                        })

                    # --- 4. Guardado Incremental (NUEVO) ---
                    # Guardamos el CSV después de CADA simulación (l, T, r)
                    # Esto es más lento pero mucho más seguro en un cluster.
                    if results_data:
                        try:
                            # Creamos un DataFrame con TODOS los resultados hasta ahora
                            df_results_incremental = pd.DataFrame(results_data)
                            # Sobreescribimos el archivo
                            df_results_incremental.to_csv(output_filename, index=False)
                            # No imprimimos esto cada vez para no saturar el log
                            # print(f"  Resultados intermedios guardados.")
                        except Exception as e:
                            # Si falla el guardado, solo advertimos y continuamos
                            print(f"  ADVERTENCIA: No se pudo guardar el archivo intermedio: {e}")
                    # --- Fin del guardado incremental ---

finally:
    # --- 4. Guardado Final ---
    # EL GUARDADO YA NO SE HACE AQUÍ, AHORA ES INCREMENTAL.
    # Mantenemos el 'finally' para imprimir el resumen final.
    if results_data:
        print("\n--- Simulación terminada. Resumen de peores fidelidades: ---")
        
        # Para el resumen, leemos el archivo final que se guardó incrementalmente
        try:
            df_results_final = pd.read_csv(output_filename)
            print(df_results_final.sort_values(by="Fidelity").head(10))
            print(f"Resultados guardados en: {os.path.abspath(output_filename)}")
        except FileNotFoundError:
            print(f"ERROR: No se encontró el archivo de resultados '{output_filename}'.")
        except Exception as e:
            print(f"No se pudo leer el archivo final de resultados para el resumen: {e}")
    else:
        print("\n--- Simulación terminada sin resultados. ---")

total_time = time.time() - start_time_total
print(f"\n--- Tiempo Total de Ejecución: {total_time/60:.2f} minutos ---")

