"""
Digitized adiabatic quantum computation for random 3-SAT (simulated with Qiskit Aer).
Full version: every instance is also solved by brute force and compared with
the quantum result.

Algorithm
---------
H(s) = (1 - s) H_B + s H_P, with s going from 0 to 1.

  * H_P = sum over clauses of the projector onto the (single) assignment that
    violates the clause. Its ground states (energy 0) are the solutions.
  * H_B = sum_i d_i (1 - X_i) / 2, where d_i is the number of clauses in which
    variable i appears. Its ground state is |+>^N.

The evolution is digitized: s takes the N_D + 1 values k / N_D (k = 0..N_D,
both ends included) and for each value N_t Trotter steps of duration
T / (N_D * N_t) are applied. H_B steps are RX rotations; H_P steps are
multi-controlled phase gates (after X gates that turn each clause into the
all-ones case). At the end all qubits are measured.

Parameter names: T = total evolution time, N_t = Trotter steps per value of s,
N_D = number of discretization steps of s (called N_s in the thesis).

Instance classification: an instance is declared SAT if at least one measured
bit string satisfies the formula (checked classically), UNSAT otherwise.
With ``comprobar_soluciones`` the set of satisfying strings found is compared
with the exhaustive solution set; mismatches are stored in
``results/mismatches/``.

Usage (from the repository root):
    python src/adiabatic_3sat.py                       # quick test (about 1 minute)
    python src/adiabatic_3sat.py --N 8 --alphas 2 3 4 5 6 --samples 500 \\
        --shots 10000000 --T 100 --nt 15 --nd 200      # thesis-like run (slow)

Progress is saved periodically in ``results/`` and a rerun with the same
(N, T, Nt, ND) resumes from it. Use ``MPLBACKEND=Agg`` or ``--no-plots`` on
machines without a display.
"""

import argparse
import ast
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer

from sat_instances import (Num_clausulas_violadas, check_result,
                           generar_formula_logica, get_real_solutions,
                           precompute_formula, print_pretty_formula)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

####################################################
#                 Configuration                    #
####################################################

Config = {
    "devolver_circuito":    False,  # also return the circuit from the solver
    "comprobar_soluciones": True,   # compare with exhaustive search
    "calcular_energia":     True,
    "calcular_unsat":       True,
    "calcular_n_efectiva":  True,
    "calcular_n_max":       True,
    "forzar_N":             False   # regenerate formulas until all N variables appear
}

####################################################
#               Main functions                     #
####################################################


def resolver_formula_3_SAT(formula, q_bits, trys, T, N_t, N_D):
    """
    Solve a K-SAT formula with the digitized adiabatic algorithm.

    - formula : list of clauses, each a list of (var_index, negated).
    - q_bits  : number of qubits = number of variables N.
    - trys    : shots of the circuit.
    - T       : total evolution time.
    - N_t     : Trotter steps per value of s.
    - N_D     : discretization steps of s.

    Returns the counts dictionary (character i of each key = value of
    variable i) and, if ``Config["devolver_circuito"]`` is set, also the circuit.
    """

    # Values of s (N_D + 1 points, both ends included)
    s_values = np.arange(N_D + 1) / N_D

    d_i = np.zeros((q_bits))  # number of times each qubit appears in the formula
    for i in formula:
        for var, negada in i:
            d_i[var] += 1

    qreg_q = QuantumRegister(q_bits, 'q')
    creg_c = ClassicalRegister(q_bits, 'c')
    q_circ = QuantumCircuit(qreg_q, creg_c)

    # Initial state: ground state of H_B, i.e. |+> on every qubit
    for i in range(q_bits):
        q_circ.h(i)

    delta_t = T/N_D
    theta_B = [0]*q_bits
    theta_P = 0

    # Precompute lists used later to apply H_P
    listas_qbits_en_clausula = []
    listas_bits_negados = []

    for clausula in formula:
        qbits_en_clausula = []
        bits_negados = []

        for var, negada in clausula:
            qbits_en_clausula.append(var)
            if not negada:
                bits_negados.append(var)  # qubits that need an X so that the clause becomes "all ones"

        listas_qbits_en_clausula.append(qbits_en_clausula)
        listas_bits_negados.append(bits_negados)

    for s in s_values:

        theta_P = -1 * s * delta_t/N_t

        for i in range(q_bits):
            theta_B[i] = -1 * (1 - s) * d_i[i] * delta_t/N_t

        for step_trotter in range(N_t):
            # First H_B (RX rotations)
            for i in range(q_bits):
                q_circ.rx(theta_B[i], i)

            # Then H_P: negate the needed variables so that the clause is equivalent
            # to one without negations, then apply the phase with a multi-controlled gate
            for n_clausula in range(len(formula)):

                bits_negados_actual = listas_bits_negados[n_clausula]
                qbits_en_clausula_actual = listas_qbits_en_clausula[n_clausula]

                for bit in bits_negados_actual:
                    q_circ.x(bit)

                q_circ.mcp(theta_P, qbits_en_clausula_actual[:-1], qbits_en_clausula_actual[-1])

                for bit in bits_negados_actual:
                    q_circ.x(bit)

    for i in range(q_bits):
        q_circ.measure(qreg_q[i], creg_c[i])

    aer_sim = Aer.get_backend('aer_simulator')

    t_qc = transpile(q_circ, aer_sim)

    resultados = aer_sim.run(t_qc, shots=int(trys)).result().get_counts()
    resultados_reves = {k[::-1]: v for k, v in resultados.items()}

    if Config["devolver_circuito"] == True:
        return resultados_reves, q_circ

    return resultados_reves


def algoritmo_de_corte(resultados, formula, verbose=True):
    """
    Return the set of measured bit strings that satisfy the formula (checked
    classically). There is no probability threshold: any measured string that
    is a solution counts. If ``verbose``, print the percentage of shots that
    landed on solutions.
    """
    items = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
    solucion = set()
    for key, values in items:
        if (check_result(formula, key)):
            solucion.add(key)
    if verbose:
        total = sum(resultados.values())
        buenos = sum(v for k, v in items if check_result(formula, k))
        print(f"Percentage of good results: {100 * buenos / total:.2f}%")
    return solucion


def Energia(resultados, formula):
    """Mean number of violated clauses of the measured states, divided by M."""

    total = sum(resultados.values())  # total counts

    E = 0
    clausulas = precompute_formula(formula)

    for bits, value in resultados.items():
        E += value * Num_clausulas_violadas(bits, clausulas)

    return E / len(formula) / total  # normalized by the number of clauses


def save_progress(nomb_archivo, estadisticas, unsat_por_alpha, energia, n_eff_por_alpha, n_max_por_alpha):
    """Save the accumulated statistics as JSON (keys kept in Spanish: read by analysis/fit_alpha_c.py)."""
    datos = {
        "estadisticas":    {str(k): v for k, v in estadisticas.items()},
        "unsat_por_alpha": unsat_por_alpha,
        "energia":         energia,
        "n_eff_por_alpha": n_eff_por_alpha,
        "n_max_por_alpha": n_max_por_alpha
    }
    os.makedirs(os.path.dirname(nomb_archivo), exist_ok=True)
    with open(nomb_archivo, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=4, sort_keys=True)
    print(f"Progress saved in: {nomb_archivo}")


def load_progress(archivo_guardado):
    """Load a previous run (if any). Returns the statistics and the counters."""
    if not os.path.exists(archivo_guardado):
        print("No previous save found.")
        return {}, {}, {}, {}, {}, 0, 0

    print("Save file found. Loading progress...")

    with open(archivo_guardado, "r", encoding="utf-8") as f:
        datos = json.load(f)

    estadisticas    = {ast.literal_eval(k): v for k, v in datos.get("estadisticas", {}).items()}
    unsat_por_alpha = {float(k): v for k, v in datos.get("unsat_por_alpha", {}).items()}
    energia         = {float(k): v for k, v in datos.get("energia",         {}).items()}
    n_eff_por_alpha = {float(k): v for k, v in datos.get("n_eff_por_alpha", {}).items()}
    n_max_por_alpha = {float(k): v for k, v in datos.get("n_max_por_alpha", {}).items()}

    count             = sum(v["total"]     for v in estadisticas.values())
    cantidad_buenos   = sum(v["correctos"] for v in estadisticas.values())

    print("Data loaded correctly.")
    return estadisticas, unsat_por_alpha, energia, n_eff_por_alpha, n_max_por_alpha, count, cantidad_buenos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Digitized adiabatic algorithm on random 3-SAT: fraction of UNSAT instances vs alpha = M/N.")
    parser.add_argument("--N", type=int, default=4, help="number of variables (qubits). Default: 4.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[3, 5],
                        help="clause densities alpha = M/N to simulate (M = round(alpha*N)). Default: 3 5.")
    parser.add_argument("--samples", type=int, default=3,
                        help="random instances per (N, M) (thesis: 1000-6000). Default: 3.")
    parser.add_argument("--T", type=int, default=100, help="total evolution time. Default: 100.")
    parser.add_argument("--nt", type=int, default=15, help="Trotter steps per value of s (N_t). Default: 15.")
    parser.add_argument("--nd", type=int, default=200, help="discretization steps of s (N_D). Default: 200.")
    parser.add_argument("--shots", type=float, default=1e4,
                        help="shots per instance (script used for thesis data: 1e7). Default: 1e4.")
    parser.add_argument("--force-n", action="store_true",
                        help="regenerate formulas until all N variables appear in them.")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the random instance generator (measurement sampling is not seeded).")
    parser.add_argument("--no-plots", action="store_true", help="do not open figures.")
    return parser.parse_args()


####################################################
#                  Main program                    #
####################################################

if __name__ == "__main__":
    args = parse_args()

    if args.no_plots:
        plt.switch_backend("Agg")
    if args.seed is not None:
        np.random.seed(args.seed)
    Config["forzar_N"] = args.force_n

    estadisticas    = {}     # {(N,M): {"total":X, "correctos":Y}}
    unsat_por_alpha = {}     # {alpha: {"total":X, "unsat":Y}}
    energia         = {}     # {alpha: {"total":X, "energia_total":Y, "energia_cuadrado_total": Z}}
    n_eff_por_alpha = {}     # {alpha: {"total":X, "n_eff_total":Y, "n_eff_cuadrado_total":Z}}
    n_max_por_alpha = {}     # {alpha: {"total":X, "n_max_total":Y, "n_max_cuadrado_total":Z}}

    # Number of samples per (N, M)
    Muestra_test = args.samples

    # Number of variables
    N = args.N

    # Values of alpha to simulate
    alpha_list = list(args.alphas)

    # Main simulation parameters
    Tiempo_Evo              = args.T
    Numero_Trotter          = args.nt
    Numero_Discretizaciones = args.nd
    num_trys                = args.shots

    ###################################

    archivo_guardado = str(
        RESULTS_DIR / f"adiabatic_full_N{N}_T{Tiempo_Evo}_Nt{Numero_Trotter}_ND{Numero_Discretizaciones}.json"
    )
    print(archivo_guardado)

    count = 0
    Cantidad_de_buenos_resultados = 0

    # If a previous save exists, load it
    estadisticas, unsat_por_alpha, energia, n_eff_por_alpha, n_max_por_alpha, count, Cantidad_de_buenos_resultados = load_progress(archivo_guardado)

    K = 3
    todas_las_alfas = set()
    errores = 0
    output_dir = str(RESULTS_DIR / "mismatches")
    os.makedirs(output_dir, exist_ok=True)

    Lista_test = {(N, int(round(alfa * N))) for alfa in alpha_list}
    iteraciones_totales = len(Lista_test) * Muestra_test

    print(f"Expected total: {iteraciones_totales} iterations")
    alpha_list.sort()

    for alfa in alpha_list:

        M = int(round(alfa * N))
        clave = (N, M)

        if clave in estadisticas and estadisticas[clave]["total"] >= Muestra_test:
            print(f"(N={N}, M={M}) already completed. Skipping.")
            continue

        todas_las_alfas.add(round(M / N, 4))

        if clave not in estadisticas:
            estadisticas[clave] = {"total": 0, "correctos": 0}

        ya_hechas = estadisticas[clave]["total"]
        print(f"Variables used: N = {N}, M = {M}, K = {K}")

        for iter in range(ya_hechas, Muestra_test):

            if iter % 100 == 0:
                save_progress(archivo_guardado, estadisticas, unsat_por_alpha, energia, n_eff_por_alpha, n_max_por_alpha)

            count += 1

            print(f"Total progress: {(100 * count / iteraciones_totales):.2f}%  "
                  f"[alpha={alfa}, iter {iter+1}/{Muestra_test}]", end="\r")

            formula, variables_utilizadas = generar_formula_logica(N, M, K)

            if Config["forzar_N"]:
                while len(variables_utilizadas) < N:
                    formula, variables_utilizadas = generar_formula_logica(N, M, K)

            # n_eff: how many distinct variables actually appear in the formula
            if Config["calcular_n_efectiva"]:
                n_eff = len(variables_utilizadas)

            estadisticas[clave]["total"] += 1

            resultado = resolver_formula_3_SAT(
                formula, N,
                trys=num_trys, T=Tiempo_Evo, N_t=Numero_Trotter, N_D=Numero_Discretizaciones
            )

            if Config["calcular_energia"]:
                E_med = Energia(resultado, formula)

            soluciones_algoritmo = algoritmo_de_corte(resultado, formula)

            if Config["comprobar_soluciones"]:
                soluciones_reales = set(get_real_solutions(formula, N))

                if soluciones_algoritmo == soluciones_reales:
                    estadisticas[clave]["correctos"] += 1
                    Cantidad_de_buenos_resultados += 1
                else:
                    print("\nMismatch between the algorithm and exhaustive search. Details: ")
                    print("\tChosen formula:")
                    if N < 8:
                        print_pretty_formula(formula)
                    else:
                        print(formula)

                    print("algorithm solutions:", soluciones_algoritmo, "real solutions:", soluciones_reales)

                    filename = f"N{N}_M{M}_sample{errores + 1}.pkl"
                    errores += 1
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "wb") as f:
                        pickle.dump({
                            "formula": formula,
                            "solucion_algoritmo": soluciones_algoritmo,
                            "solucion real": soluciones_reales
                        }, f)

            alpha = round(M / N, 4)

            # Initialize the entries the first time this alpha appears
            if alpha not in unsat_por_alpha:
                unsat_por_alpha[alpha]   = {"total": 0, "unsat": 0}
                energia[alpha]           = {"total": 0, "energia_total": 0, "energia_cuadrado_total": 0}
                n_eff_por_alpha[alpha]   = {"total": 0, "n_eff_total": 0.0, "n_eff_cuadrado_total": 0.0}
                n_max_por_alpha[alpha]   = {"total": 0, "n_max_total": 0.0, "n_max_cuadrado_total": 0.0}

            # UNSAT
            if Config["calcular_unsat"]:
                unsat_por_alpha[alpha]["total"] += 1
                if len(soluciones_algoritmo) == 0:
                    unsat_por_alpha[alpha]["unsat"] += 1

            # Energy (running mean and sum of squared deviations, Welford)
            if Config["calcular_energia"]:
                energia[alpha]["total"] += 1
                delta = E_med - energia[alpha]["energia_total"]
                energia[alpha]["energia_total"] += delta / energia[alpha]["total"]
                delta2 = E_med - energia[alpha]["energia_total"]
                energia[alpha]["energia_cuadrado_total"] += delta * delta2

            # n_eff
            if Config["calcular_n_efectiva"]:
                n_eff_por_alpha[alpha]["total"] += 1
                delta = n_eff - n_eff_por_alpha[alpha]["n_eff_total"]
                n_eff_por_alpha[alpha]["n_eff_total"] += delta / n_eff_por_alpha[alpha]["total"]
                delta2 = n_eff - n_eff_por_alpha[alpha]["n_eff_total"]
                n_eff_por_alpha[alpha]["n_eff_cuadrado_total"] += delta * delta2

            if Config["calcular_n_max"]:
                # flatten the formula: keep ONLY the variables (ignore negation)
                flat_vars = [var for clause in formula for (var, _) in clause]
                counts_vars = Counter(flat_vars)
                n_max = max(counts_vars.values()) if counts_vars else 0

                n_max_por_alpha[alpha]["total"] += 1
                delta = n_max - n_max_por_alpha[alpha]["n_max_total"]
                n_max_por_alpha[alpha]["n_max_total"] += delta / n_max_por_alpha[alpha]["total"]
                delta2 = n_max - n_max_por_alpha[alpha]["n_max_total"]
                n_max_por_alpha[alpha]["n_max_cuadrado_total"] += delta * delta2

        save_progress(archivo_guardado, estadisticas, unsat_por_alpha, energia, n_eff_por_alpha, n_max_por_alpha)
        print(f"Data saved at: {time.strftime('%H:%M:%S %d/%m/%Y')}")

    print(f"Alpha values simulated in this run:\n\t{todas_las_alfas}")

    save_progress(archivo_guardado, estadisticas, unsat_por_alpha, energia, n_eff_por_alpha, n_max_por_alpha)

    ####################################
    #       Errors and summary         #
    ####################################

    # Probability of UNSAT as a function of alpha

    alphas = sorted(unsat_por_alpha.keys())

    fracciones = [
        unsat_por_alpha[a]["unsat"] / unsat_por_alpha[a]["total"]
        for a in alphas
    ]

    errores_p_sat = [
        (p * (1 - p) / unsat_por_alpha[a]["total"]) ** 0.5  # binomial standard error
        for p, a in zip(fracciones, alphas)
    ]

    plt.figure()
    plt.errorbar(alphas, fracciones, yerr=errores_p_sat, fmt='o-', capsize=4)
    plt.xlabel("α = M/N")
    plt.ylabel("Fraction of Unsatisfiable Formulae")
    plt.title("Fraction of unsatisfiable formulae vs α")
    plt.grid(True)
    plt.show()

    # unsat / total for each alpha

    print("\nUNSAT results per alpha:\n")
    print(f"{'alpha':>8}  {'total':>7}  {'unsat':>7}  {'p(UNSAT)':>10}")
    print("-" * 40)
    for a, p_unsat in zip(alphas, fracciones):
        total = unsat_por_alpha[a]["total"]
        unsat = unsat_por_alpha[a]["unsat"]
        print(f"{a:8.4f}  {total:7d}  {unsat:7d}  {p_unsat:10.4f}")

    # Mean energy

    if Config["calcular_energia"]:
        medias = []
        errores_ener = []

        for a in alphas:
            data = energia[a]
            N_tot = data["total"]

            mean = data["energia_total"]

            var = data["energia_cuadrado_total"] / (N_tot - 1) if N_tot > 1 else 0.0
            error = (var / N_tot) ** 0.5   # standard error

            medias.append(mean)
            errores_ener.append(error)

        plt.figure()
        plt.errorbar(alphas, medias, yerr=errores_ener, fmt='o-', capsize=4)
        plt.xlabel("α = M/N")
        plt.ylabel("Mean energy")
        plt.title("Energy vs α")
        plt.grid(True)
        plt.show()

    total_tests = sum(v["total"] for v in estadisticas.values())

    print(
        f"Good results: {Cantidad_de_buenos_resultados} out of {total_tests} tests "
        f"({100*Cantidad_de_buenos_resultados/total_tests:.2f}%)"
    )

    # Mean effective N for each alpha

    if Config["calcular_n_efectiva"]:

        medias_n_eff  = []
        errores_n_eff = []

        for a in alphas:
            data = n_eff_por_alpha[a]
            n    = data["total"]
            mean = data["n_eff_total"]

            var   = data["n_eff_cuadrado_total"]/(n - 1) if n > 1 else 0.0
            error = (var / n) ** 0.5   # standard error of the mean

            medias_n_eff.append(mean)
            errores_n_eff.append(error)

        print("\nMean effective N (per α):\n")
        for a, m, e in zip(alphas, medias_n_eff, errores_n_eff):
            print(f"α={a:.4f} → n_eff = {m:.4f} ± {e:.4f}")

        # p(UNSAT) against alpha and against alpha_eff = alpha * n_eff / N
        plt.figure()
        plt.errorbar(alphas, fracciones, yerr=errores_p_sat, fmt='o-', capsize=4, label='p(UNSAT) vs α')
        alphas_eff = [a * (m / N) for a, m in zip(alphas, medias_n_eff)]
        plt.errorbar(alphas_eff, fracciones, yerr=errores_p_sat, fmt='o-', capsize=4, label='p(UNSAT) vs α_eff')
        plt.xlabel("α = M/N")
        plt.ylabel("Fraction of Unsatisfiable Formulae")
        plt.title("Fraction of unsatisfiable formulae vs α")
        plt.legend()
        plt.grid(True)

        plt.show()

        # N_eff as a function of alpha
        plt.figure()
        plt.errorbar(alphas, medias_n_eff, yerr=errores_n_eff, fmt='o-', capsize=4)
        plt.axhline(N, color='gray', linestyle='--', label=f'Nominal N = {N}')
        plt.xlabel("α = M/N")
        plt.ylabel("Mean effective N")
        plt.title("Effective N vs α")
        plt.legend()
        plt.grid(True)

        plt.show()

    if Config["calcular_n_max"]:

        medias_n_max  = []
        errores_n_max = []

        for a in alphas:
            data = n_max_por_alpha[a]
            n    = data["total"]
            mean = data["n_max_total"]

            var   = data["n_max_cuadrado_total"]/(n - 1) if n > 1 else 0.0
            error = (var / n) ** 0.5   # standard error of the mean

            medias_n_max.append(mean)
            errores_n_max.append(error)

        print("\nMean N max (per α):\n")
        for a, m, e in zip(alphas, medias_n_max, errores_n_max):
            print(f"α={a:.4f} → n_max = {m:.4f} ± {e:.4f}")

        plt.figure()
        plt.errorbar(alphas, medias_n_max, yerr=errores_n_max, fmt='o-', capsize=4)
        plt.xlabel("α = M/N")
        plt.ylabel("Mean N max (most frequent variable)")
        plt.title("N max vs α")
        plt.grid(True)
        plt.show()
