"""
Digitized adiabatic quantum computation for random 3-SAT: "lite" version.

Same circuit and same physics as ``adiabatic_3sat.py`` (see the algorithm
description there), but built for long runs on many values of alpha:

  * It does NOT compare the quantum result with an exhaustive search, so it
    scales to larger N. An instance is declared SAT if at least one measured
    bit string satisfies the formula (checked classically) and UNSAT
    otherwise. There are no false SAT, but an instance with solutions that
    are never measured would be counted as UNSAT.
  * It stores only p(UNSAT) and the mean energy per alpha, in a JSON file in
    ``results/`` that ``analysis/fit_alpha_c.py`` can read directly.

Usage (from the repository root):
    python src/adiabatic_3sat_lite.py                  # quick test (about 1 minute)
    python src/adiabatic_3sat_lite.py --N 8 --samples 2000 --nt 30 --shots 100000 \\
        --alphas 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4 4.25 4.5 4.75 5 5.25 5.5 6 6.5 7
        # settings of the N = 8, Nt = 30 curve of the thesis (slow)

Progress is saved every 100 instances and a rerun with the same
(N, T, Nt, ND) resumes from the saved file. Use ``--no-plots`` (or
``MPLBACKEND=Agg``) on machines without a display.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer

from sat_instances import (Num_clausulas_violadas, generar_formula_logica,
                           precompute_formula)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def resolver_formula_3_SAT(formula, q_bits, trys, T, N_t, N_D):
    """
    Digitized adiabatic evolution for a K-SAT formula (see adiabatic_3sat.py).
    Returns the counts dictionary; character i of each key = value of variable i.
    """

    s_values = np.arange(N_D + 1) / N_D

    d_i = np.zeros(q_bits)
    for clausula in formula:
        for var, _ in clausula:
            d_i[var] += 1

    qreg_q = QuantumRegister(q_bits, 'q')
    creg_c = ClassicalRegister(q_bits, 'c')
    q_circ = QuantumCircuit(qreg_q, creg_c)

    for i in range(q_bits):
        q_circ.h(i)

    delta_t = T / N_D
    theta_B = [0.0] * q_bits

    # Precompute the qubits of each clause and the ones that need an X gate
    listas_qbits  = []
    listas_neg    = []
    for clausula in formula:
        qbits = [var for var, _ in clausula]
        neg   = [var for var, negada in clausula if not negada]
        listas_qbits.append(qbits)
        listas_neg.append(neg)

    for s in s_values:
        theta_P = -s * delta_t / N_t
        for i in range(q_bits):
            theta_B[i] = -(1 - s) * d_i[i] * delta_t / N_t

        for _ in range(N_t):
            for i in range(q_bits):
                q_circ.rx(theta_B[i], i)
            for nc in range(len(formula)):
                for bit in listas_neg[nc]:
                    q_circ.x(bit)
                q_circ.mcp(theta_P, listas_qbits[nc][:-1], listas_qbits[nc][-1])
                for bit in listas_neg[nc]:
                    q_circ.x(bit)

    for i in range(q_bits):
        q_circ.measure(qreg_q[i], creg_c[i])

    aer_sim = Aer.get_backend('aer_simulator')
    t_qc    = transpile(q_circ, aer_sim)
    raw     = aer_sim.run(t_qc, shots=int(trys)).result().get_counts()

    # Reverse the bit order to follow the convention q_0 = first character
    return {k[::-1]: v for k, v in raw.items()}


def check_result_precomp(formula_precomp, estado_str):
    """True if the bit string satisfies the formula (formula in mask form, see precompute_formula)."""
    val = int(estado_str[::-1], 2)   # reversed so that q_0 = least significant bit
    for mask_vars, mask_neg in formula_precomp:
        bits = val & mask_vars
        bits ^= mask_neg
        if bits == 0:
            return False   # violated clause
    return True


def is_sat(resultados, formula_precomp):
    """
    True if any measured state satisfies the formula.
    Fast approximation: it does not enumerate the 2^N states.
    """
    for estado in resultados:
        if check_result_precomp(formula_precomp, estado):
            return True
    return False


def Energia(resultados, formula_precomp, M):
    """Mean number of violated clauses of the measured states, divided by M."""
    total = sum(resultados.values())
    E = 0.0
    for estado, cnt in resultados.items():
        E += cnt * Num_clausulas_violadas(estado, formula_precomp)
    return E / M / total


def save_progress(nombre, unsat_por_alpha, energia):
    """Save the statistics as JSON (keys kept in Spanish: read by analysis/fit_alpha_c.py)."""
    datos = {
        "unsat_por_alpha": unsat_por_alpha,
        "energia":         energia,
    }
    os.makedirs(os.path.dirname(nombre), exist_ok=True)
    with open(nombre, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=4, sort_keys=True)
    print(f"\n[saved] {nombre}")


def load_progress(nombre, N):
    """Load a previous run (if any). Returns the statistics and the number of instances done per (N, M)."""
    if not os.path.exists(nombre):
        print("No previous save found.")
        return {}, {}, {}

    print("Loading previous progress...")
    with open(nombre, "r", encoding="utf-8") as f:
        datos = json.load(f)

    unsat_por_alpha = {float(k): v for k, v in datos.get("unsat_por_alpha", {}).items()}
    energia         = {float(k): v for k, v in datos.get("energia",         {}).items()}

    # Rebuild the number of iterations already done per (N, M)
    ya_hechos = {}
    for a, d in unsat_por_alpha.items():
        M_a = int(round(a * N))
        ya_hechos[(N, M_a)] = d["total"]

    print("Data loaded.")
    return unsat_por_alpha, energia, ya_hechos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lite digitized adiabatic algorithm on random 3-SAT: p(UNSAT) and energy vs alpha = M/N.")
    parser.add_argument("--N", type=int, default=4, help="number of variables (qubits). Default: 4.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[3, 5],
                        help="clause densities alpha = M/N (M = round(alpha*N)). Default: 3 5.")
    parser.add_argument("--samples", type=int, default=3,
                        help="random instances per value of alpha (thesis: 1000-3000). Default: 3.")
    parser.add_argument("--T", type=int, default=100, help="total evolution time. Default: 100.")
    parser.add_argument("--nt", type=int, default=15, help="Trotter steps per value of s (N_t). Default: 15.")
    parser.add_argument("--nd", type=int, default=200, help="discretization steps of s (N_D). Default: 200.")
    parser.add_argument("--shots", type=float, default=1e4, help="shots per instance (thesis: 1e5). Default: 1e4.")
    parser.add_argument("--no-energy", action="store_true", help="skip the mean-energy computation (faster).")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the random instance generator (measurement sampling is not seeded).")
    parser.add_argument("--no-plots", action="store_true", help="do not open figures.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.no_plots:
        plt.switch_backend("Agg")
    if args.seed is not None:
        np.random.seed(args.seed)

    Flag_Energia = not args.no_energy  # compute the mean energy (slower)

    N                       = args.N      # number of variables

    K                       = 3           # literals per clause

    Muestra_test            = args.samples  # samples per value of alpha

    cada_cuanto_guardar     = 100         # save progress every this many iterations

    alpha_list = list(args.alphas)

    Tiempo_Evo              = args.T

    Numero_Trotter          = args.nt

    Numero_Discretizaciones = args.nd

    Num_trys                = args.shots

    archivo_guardado = str(
        RESULTS_DIR / f"adiabatic_lite_N{N}_T{Tiempo_Evo}_Nt{Numero_Trotter}_ND{Numero_Discretizaciones}.json"
    )

    print(archivo_guardado)

    unsat_por_alpha, energia, ya_hechos = load_progress(archivo_guardado, N)

    alpha_list.sort()

    Lista_test          = [(N, int(round(a * N))) for a in alpha_list]

    iteraciones_totales = len(Lista_test) * Muestra_test

    count               = sum(ya_hechos.values())

    print(f"Expected total iterations: {iteraciones_totales}")

    print(f"Already done: {count}\n")

    for alfa in alpha_list:

        M     = int(round(alfa * N))
        alpha = round(M / N, 4)

        ya = ya_hechos.get((N, M), 0)

        if ya >= Muestra_test:
            print(f"α={alpha} (N={N}, M={M}) — already completed, skipping.")
            continue

        if alpha not in unsat_por_alpha:
            unsat_por_alpha[alpha] = {"total": 0, "unsat": 0}
        if alpha not in energia:
            energia[alpha]         = {"total": 0, "energia_total": 0.0, "energia_cuadrado_total": 0.0}

        print(f"\nα={alpha}  (N={N}, M={M})  — {Muestra_test - ya} samples missing")

        for iter_i in range(ya, Muestra_test):

            if iter_i % (cada_cuanto_guardar) == 0:
                save_progress(archivo_guardado, unsat_por_alpha, energia)

            count += 1
            pct    = 100.0 * count / iteraciones_totales
            print(f"  [{pct:5.1f}% ] iter {iter_i+1}/{Muestra_test}", end="\r")

            formula, _ = generar_formula_logica(N, M, K)
            formula_precomp = precompute_formula(formula)

            resultado = resolver_formula_3_SAT(
                formula, N,
                trys=Num_trys, T=Tiempo_Evo, N_t=Numero_Trotter, N_D=Numero_Discretizaciones
            )

            unsat_por_alpha[alpha]["total"] += 1
            if not is_sat(resultado, formula_precomp):
                unsat_por_alpha[alpha]["unsat"] += 1

            if Flag_Energia:
                # running mean and sum of squared deviations (Welford)
                E_med = Energia(resultado, formula_precomp, M)
                energia[alpha]["total"] += 1
                delta  = E_med - energia[alpha]["energia_total"]
                energia[alpha]["energia_total"]         += delta / energia[alpha]["total"]
                delta2  = E_med - energia[alpha]["energia_total"]
                energia[alpha]["energia_cuadrado_total"] += delta * delta2

        save_progress(archivo_guardado, unsat_por_alpha, energia)

    print(f"\n\nSimulation completed: {time.strftime('%H:%M:%S %d/%m/%Y')}")

    alphas = sorted(unsat_por_alpha.keys())

    print(f"\n{'alpha':>8}  {'total':>7}  {'unsat':>7}  {'p(UNSAT)':>10}  {'σ':>8}")

    print("-" * 50)

    fracciones    = []

    errores_p_sat = []

    for a in alphas:
        tot   = unsat_por_alpha[a]["total"]
        uns   = unsat_por_alpha[a]["unsat"]
        p_uns = uns / tot if tot > 0 else 0.0
        err   = (p_uns * (1 - p_uns) / tot) ** 0.5 if tot > 1 else 0.0  # binomial standard error
        fracciones.append(p_uns)
        errores_p_sat.append(err)
        print(f"{a:8.4f}  {tot:7d}  {uns:7d}  {p_uns:10.4f}  {err:8.4f}")

    plt.figure()

    plt.errorbar(alphas, fracciones, yerr=errores_p_sat, fmt='o-', capsize=4)

    plt.xlabel("α = M/N")

    plt.ylabel("p(UNSAT)")

    plt.title(f"Fraction of unsatisfiable formulae vs α  (N={N}, lite)")

    plt.grid(True)

    plt.show()

    if Flag_Energia:
        medias_e = []
        errores_e = []
        for a in alphas:
            d      = energia[a]
            n_tot  = d["total"]
            mean   = d["energia_total"]
            var    = d["energia_cuadrado_total"] / (n_tot - 1) if n_tot > 1 else 0.0
            err    = (var / n_tot) ** 0.5
            medias_e.append(mean)
            errores_e.append(err)

        plt.figure()
        plt.errorbar(alphas, medias_e, yerr=errores_e, fmt='o-', capsize=4, color='tab:orange')
        plt.xlabel("α = M/N")
        plt.ylabel("Normalized mean energy")
        plt.title(f"Energy vs α  (N={N}, lite)")
        plt.grid(True)
        plt.show()

        # Energy vs p(UNSAT)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.errorbar(alphas, fracciones, yerr=errores_p_sat,
                     fmt='o-', color='tab:blue', capsize=4, label='p(UNSAT)')
        ax2.errorbar(alphas, medias_e,   yerr=errores_e,
                     fmt='s--', color='tab:orange', capsize=4, label='Energy')
        ax1.set_xlabel("α = M/N")
        ax1.set_ylabel("p(UNSAT)",       color='tab:blue')
        ax2.set_ylabel("Mean energy",    color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        fig.suptitle(f"p(UNSAT) and energy vs α  (N={N}, lite)")
        fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
        plt.grid(True)
        plt.show()
