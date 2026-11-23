"""
Grover's algorithm for random 3-SAT instances (simulated with Qiskit Aer).

The circuit uses N variable qubits, one ancilla per clause and one "master"
ancilla prepared in |-> (phase kickback):

    1. For every clause, a multi-controlled X flips its ancilla when the clause
       is *violated*; a final X turns that into "ancilla = 1 <=> satisfied".
    2. A multi-controlled X from all clause ancillas onto the master ancilla
       marks (phase -1) the assignments that satisfy every clause.
    3. Step 1 is undone (uncomputation) and the diffusion operator
       (inversion about the mean) is applied.
    4. Steps 1-3 form one Grover iteration; ``reps`` iterations are applied.

Running this file as a script reproduces the experiment of the thesis:
success probability of Grover's search for N = 3 variables as a function of the
number of iterations, for instances with different numbers of solutions. The
results are saved as a pickle in ``results/`` (read by
``analysis/plot_grover.py``).

Usage (from the repository root):
    python src/grover_3sat.py                       # quick run (about 20 seconds)
    python src/grover_3sat.py --instances 200 --shots 100000000 --clauses 1 4 5
        # settings of the thesis figures. Aer needs roughly 113 bytes per shot
        # (about 11 GB at 1e8 shots, extrapolated from a measurement at 1e7), so this
        # can fail with std::bad_alloc on a laptop. Using --shots 1000000 gives the same
        # curves (the success probability only depends on the number of solutions)
        # with a statistical error below 1e-3.

This is a statevector-based simulation (no hardware, no noise): measurements
are sampled from the exact final state.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer

from sat_instances import generar_formula_logica, get_real_solutions

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def cmx_gate_def(n_controls):
    """
    Build a multi-controlled X (MCX) gate with ``n_controls`` control qubits
    (the target is the last qubit) and return it as a Qiskit ``Gate``.
    """
    q = QuantumRegister(n_controls + 1)
    qc = QuantumCircuit(q)
    qc.mcx(list(range(n_controls)), n_controls)

    return qc.to_gate()


def resolver_formula_3_SAT(formula, shots=1e4, reps=1):
    """
    Run Grover's algorithm on a K-SAT instance.

    formula (list): list of clauses; each clause is a list of ``(var, negated)``
                    tuples (0-based variable index, ``True`` if negated).
                    All variables 0..N-1 must appear in the formula.
    shots (int):    number of measurements. Default 10000.
    reps  (int):    number of Grover iterations. Default 1.

    Returns:
        resultados (dict): measured bit strings and their counts. Character i
            is the value of variable i (Qiskit's order is reversed here).
        q_circ (QuantumCircuit): the circuit that was built.
    """

    # Instance parameters
    N = len(set(var for clausula in formula for var, _ in clausula))
    M = len(formula)
    K = len(formula[0]) if M > 0 else 0

    # Qubits: N variables + M clause ancillas + 1 master ancilla
    nq = N + M + 1
    nc = N

    # Reusable multi-controlled gates
    cmx_gate   = cmx_gate_def(K)     # MCX with K controls (one per clause)
    cpmx_gate  = cmx_gate_def(M)     # MCX with M controls (marks the master ancilla)
    cppmx_gate = cmx_gate_def(N - 1) # MCX with N-1 controls (inversion about the mean)

    qreg_q = QuantumRegister(nq, 'q')
    creg_c = ClassicalRegister(nc, 'c')

    q_circ = QuantumCircuit(qreg_q, creg_c)

    # Master ancilla in |->, variable qubits in the uniform superposition
    q_circ.x(N + M)
    q_circ.h(N + M)

    for ind in range(N):
        q_circ.h(ind)

    q_circ.barrier()

    # Grover iterations
    for it in range(reps):
        # Oracle, part 1: set each clause ancilla to 1 if the clause is satisfied
        for j, clausula in enumerate(formula):
            for var, negada in clausula:
                if not negada:
                    q_circ.x(var)
            q_circ.append(cmx_gate, [var for var, _ in clausula] + [N + j])  # flips the ancilla if the clause is violated
            for var, negada in clausula:
                if not negada:
                    q_circ.x(var)
            q_circ.x(N + j)  # ancilla = 1 <=> clause satisfied

        # Oracle, part 2: mark (phase -1) the states that satisfy all clauses
        q_circ.append(cpmx_gate, [i for i in range(N, N + M + 1)])

        # Oracle, part 3: undo part 1 (uncomputation)
        for j, clausula in reversed(list(enumerate(formula))):
            q_circ.x(N + j)
            for var, negada in clausula:
                if not negada:
                    q_circ.x(var)
            q_circ.append(cmx_gate, [var for var, _ in clausula] + [N + j])
            for var, negada in clausula:
                if not negada:
                    q_circ.x(var)

        q_circ.barrier()

        # Diffusion operator (inversion about the mean)
        for ind in range(N):
            q_circ.h(ind)
            q_circ.x(ind)

        q_circ.h(N - 1)

        q_circ.append(cppmx_gate, [i for i in range(N)])

        q_circ.h(N - 1)

        for ind in range(N):
            q_circ.x(ind)
            q_circ.h(ind)

        q_circ.barrier()

    for i in range(N):
        q_circ.measure(qreg_q[i], creg_c[i])

    aer_sim = Aer.get_backend('aer_simulator')

    t_qc = transpile(q_circ, aer_sim)
    resultados = aer_sim.run(t_qc, shots=int(shots)).result().get_counts()
    resultados_reves = {k[::-1]: v for k, v in resultados.items()}
    return resultados_reves, q_circ


def save_data(datos, filename):
    """Save the experiment results as a pickle."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(datos, f)
    print(f"Data saved to {filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grover search on random 3-SAT instances with N = 3 variables: "
                    "success probability vs number of Grover iterations.")
    parser.add_argument("--clauses", type=int, nargs="+", default=[1, 4, 5],
                        help="numbers of clauses M to simulate (for N=3, K=3 distinct clauses "
                             "give 8-M solutions). Default: 1 4 5 (7, 4 and 3 solutions).")
    parser.add_argument("--instances", type=int, default=5,
                        help="random instances per value of M (thesis: 200). Default: 5.")
    parser.add_argument("--shots", type=float, default=1e5,
                        help="shots per circuit (thesis: 1e8). Default: 1e5.")
    parser.add_argument("--max-reps", type=int, default=19,
                        help="Grover iterations scanned: 1..max-reps. Default: 19.")
    parser.add_argument("--check-reps", type=int, default=2,
                        help="iterations for the initial sanity check on formula (2.1) of the "
                             "thesis (optimum: 2, ~97.6%% success). Default: 2.")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the random instance generator (the sampling of "
                             "measurements is not seeded).")
    parser.add_argument("--output", type=Path, default=None,
                        help="output pickle. Default: results/datos_grover_<instances>.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Sanity check on formula (2.1) of the thesis (N = 3, M = 3, 5 solutions out of 8):
    # probability that a measurement returns a true solution.
    # ------------------------------------------------------------------
    formula = [[(0, False), (1, False), (2, False)], [(0, True), (1, True), (2, True)], [(0, False), (1, True), (2, False)]]

    iteraciones = args.check_reps
    shots_check = int(args.shots)

    resultados_grover = resolver_formula_3_SAT(formula, shots=shots_check, reps=iteraciones)[0]
    resultados_reales = get_real_solutions(formula, 3)
    probabilidad_exito = sum(resultados_grover.get(sol, 0) for sol in resultados_reales) / shots_check

    print(f"Success probability with {iteraciones} iterations: {probabilidad_exito:.4f}")

    # ------------------------------------------------------------------
    # Experiment: success probability vs number of iterations, grouped by the
    # number of solutions of each random instance.
    # ------------------------------------------------------------------
    datos = {}  # key: number of solutions; value: list of (reps, p_values) curves
    N = 3
    n_iteraciones = args.instances

    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for M in args.clauses:  # M = number of clauses
        print("\n", f"M = {M} clauses", "\n")
        print("Step", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for iteraciones in range(n_iteraciones):
            print(f"{round(100*(iteraciones + 1)/n_iteraciones, 2)}%", end="\r")
            trys = int(args.shots)

            # make sure that all N variables appear in the formula
            variables = []
            while len(variables) < N:
                formula, variables = generar_formula_logica(N, M, K=3)

            soluciones = get_real_solutions(formula, N)
            M_sol = len(soluciones)

            # skip unsatisfiable instances
            if M_sol == 0:
                continue

            reps = []
            p_values = []

            for rep in range(1, args.max_reps + 1):
                res, circ = resolver_formula_3_SAT(formula, shots=trys, reps=rep)

                p_value = sum(res.get(sol, 0) for sol in soluciones) / trys

                reps.append(rep)
                p_values.append(p_value)

            # store grouped by number of solutions
            if M_sol not in datos:
                datos[M_sol] = []

            datos[M_sol].append((reps, p_values))

    print("End time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    file = args.output or RESULTS_DIR / f"datos_grover_{n_iteraciones}.pkl"
    save_data(datos, file)
    print("Plot it with: python analysis/plot_grover.py", file)
