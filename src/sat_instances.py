"""
Helpers shared by the Grover and adiabatic implementations: random 3-SAT
instance generation, classical checking and brute-force enumeration.

Formula representation
----------------------
A formula is a list of clauses. Each clause is a list of ``(var, negated)``
pairs, where ``var`` is the 0-based variable (= qubit) index and ``negated``
is ``True`` when the literal is negated.  Example: ``(x0 or not x1 or x2)``
is ``[(0, False), (1, True), (2, False)]``.

Bit-string convention
---------------------
A candidate assignment is a string of N bits where character ``i`` is the
value of variable ``i`` (so qubit 0 is the *leftmost* character). Qiskit
returns bit strings in the opposite order, which is why the solvers reverse
the keys of the counts dictionary before returning them.

The core functions keep the Spanish names used in Annex A of the thesis
(generar_formula_logica, resolver_formula_3_SAT, ...); a few helpers have English names.
"""

from math import comb

import numpy as np


def generar_formula_logica(N, M, K, p=0.5):
    """
    Generate a random K-SAT formula with the standard random model.

    Each clause picks K distinct variables out of N uniformly at random and
    negates each one with probability ``p``. Clauses are kept distinct.

    Parameters
    ----------
    N : number of variables.
    M : number of clauses.
    K : literals per clause (K-SAT).
    p : probability that a literal is negated (default 0.5).

    Returns
    -------
    formula : list of clauses, each a list of ``(var_index, negated)``.
    variables_usadas : set with the indices of the variables that actually
        appear in the formula (it can be smaller than N for small M).
    """
    if p > 1 or p < 0:
        raise ValueError(f"The negation probability must be in [0, 1], got p={p}")

    if K > N:  # a clause needs K distinct variables
        raise ValueError(f"Cannot pick K={K} distinct variables out of N={N}.")

    if N > K * M:
        # M clauses can contain at most K*M different variables
        print(f"Warning: N={N} is larger than K*M={K * M}. Only {K * M} variables will be used.")
        N = K * M

    M_max = comb(N, K) * (2 ** K)  # number of distinct clauses that exist

    if M > M_max:
        print(f"Warning: M={M} exceeds the number of distinct clauses available ({M_max}).")
        M = M_max

    formula = []
    variables_usadas = set()
    clausulas_generadas = set()

    while len(formula) < M:
        vars_in_clausula = np.random.choice(N, size=K, replace=False)
        clausula = tuple(sorted((var, np.random.rand() < p) for var in vars_in_clausula))
        # sorted so that (0, True), (1, False) and (1, False), (0, True) are the same clause

        if clausula not in clausulas_generadas:
            clausulas_generadas.add(clausula)
            formula.append(list(clausula))
            for var, _ in clausula:
                variables_usadas.add(var)
        # if the clause was already generated, simply try again
    return formula, variables_usadas


def print_pretty_formula(formula):
    """Print a formula in readable form (at most 10 variables: x, y, z, w, v, m, n, o, p, q)."""
    var_names = ['x', 'y', 'z', 'w', 'v', 'm', 'n', 'o', 'p', 'q']
    max_var = max(var for clausula in formula for var, _ in clausula)

    if max_var >= len(var_names):
        raise ValueError(f"More than {len(var_names)} variables: cannot print with only {var_names}")

    clauses_str = []

    for clausula in formula:
        lits = []
        for var, negada in clausula:
            name = var_names[var]
            lits.append(f"¬{name}" if negada else name)
        clauses_str.append("(" + " ∨ ".join(lits) + ")")

    print(" ∧ ".join(clauses_str))


def check_result(formula, resultado_propuesto):
    """
    Check classically whether a bit string satisfies the formula.

    ``resultado_propuesto`` is a string such as ``'0101'`` (character i = value
    of variable i). A literal is true when ``bit XOR negated == 1``.
    """
    bits = [int(b) for b in resultado_propuesto]
    return all(any(bits[var] ^ negada == 1 for var, negada in clausula) for clausula in formula)


def get_real_solutions(formula, N):
    """Brute-force enumeration of all satisfying assignments (cost O(2^N * M))."""
    soluciones = set()

    for i in range(2**N):
        estado = format(i, f"0{N}b")  # N-bit binary string

        if check_result(formula, estado):
            soluciones.add(estado)

    return soluciones


def precompute_formula(formula):
    """
    Encode every clause as a pair of bit masks ``(mask_vars, mask_neg)``:
    variables present in the clause and variables that appear negated.
    Used to count violated clauses with fast integer operations.
    """
    clausulas = []

    for clausula in formula:
        mask_vars = 0
        mask_neg = 0

        for var, negada in clausula:
            mask_vars |= (1 << var)
            if negada:
                mask_neg |= (1 << var)

        clausulas.append((mask_vars, mask_neg))

    return clausulas


def Num_clausulas_violadas(estado, clausulas):
    """
    Number of clauses violated by a bit string (convention above).
    ``clausulas`` is the output of :func:`precompute_formula`.
    """
    estado = int(estado[::-1], 2)  # variable i -> bit i of the integer

    violadas = 0

    for mask_vars, mask_neg in clausulas:
        bits = estado & mask_vars  # keep only the variables of this clause
        bits ^= mask_neg           # flip the negated ones
        if bits == 0:              # every literal is false -> clause violated
            violadas += 1

    return violadas
