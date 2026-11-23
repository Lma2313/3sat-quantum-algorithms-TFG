"""
Instantaneous spectrum of the adiabatic Hamiltonian H(s) = (1 - s) H_B + s H_P
for a small 3-SAT instance, by exact diagonalization.

    H_B = sum_i d_i (1 - X_i) / 2      (d_i = number of clauses containing variable i)
    H_P = sum over clauses of the projector onto the assignment that violates the clause

Adiabatic computation works if the gap between the ground state and the first
excited state never closes (or closes very slowly) along s in [0, 1].
This script reproduces the spectrum figure of the thesis for instance (2.1):

    (not x0 or not x1 or not x2) and (x0 or x1 or x2) and (x0 or not x1 or x2)

It was extracted from an exploratory notebook. Compared with the notebook
version, qubit i is now the same bit of the basis-state index in H_B and in
H_P (little-endian, as in Qiskit): the notebook used opposite orders in the
two operators, which gives the same spectrum for instance (2.1) (all d_i are
equal) but a wrong one for instances where the d_i differ.

Usage (from the repository root):
    python analysis/energy_spectrum.py
    python analysis/energy_spectrum.py --no-show

Cost grows as 4^N per value of s: meant for N <= 10 or so.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Instance (2.1) of the thesis, in the (var, negated) clause format of src/sat_instances.py
FORMULA_2_1 = [[(0, True), (1, True), (2, True)],
               [(0, False), (1, False), (2, False)],
               [(0, False), (1, True), (2, False)]]


def clausula_a_matriz(clausula, n_qubits):
    """
    Diagonal matrix of one clause: 1 on the basis state that violates it
    (non-negated variables = 0, negated variables = 1), 0 elsewhere.
    Variable ``var`` is bit ``var`` of the basis-state index (little-endian).
    """
    dim = 2**n_qubits
    M = np.zeros((dim, dim))

    for i in range(dim):
        bits = [(i >> k) & 1 for k in range(n_qubits)]  # bits[k] = bit k of i
        violada = True
        for var, negada in clausula:
            if not negada:
                violada = violada and (bits[var] == 0)
            else:
                violada = violada and (bits[var] == 1)
        if violada:
            M[i, i] = 1
    return M


def hamiltonianos(formula, n):
    """Return (H_B, H_P) as dense 2^n x 2^n matrices."""
    X = np.array([[0, 1],
                  [1, 0]])

    # d_i: number of occurrences of variable i
    d = np.zeros(n)
    for clausula in formula:
        for var, _ in clausula:
            d[var] += 1

    # X_i acts on bit i of the basis-state index (little-endian, like Qiskit)
    X_i = []
    for i in range(n):
        X_i.append(np.kron(np.eye(2**(n - i - 1)), np.kron(X, np.eye(2**i))))

    H_B = np.zeros((2**n, 2**n))
    for i in range(n):
        H_B += (1/2 * np.eye(2**n) - 1/2 * X_i[i]) * d[i]

    H_P = np.zeros((2**n, 2**n))
    for clausula in formula:
        H_P += clausula_a_matriz(clausula, n_qubits=n)

    return H_B, H_P


def espectro(formula, n, n_points=101):
    """Eigenvalues (sorted) of H(s) for s in [0, 1]. Returns (s_values, array [n_points, 2^n])."""
    H_B, H_P = hamiltonianos(formula, n)
    s_values = np.linspace(0, 1, n_points)
    eigenvalues = []
    for s in s_values:
        M_s = (1 - s) * H_B + s * H_P
        eigenvalues.append(np.linalg.eigvalsh(M_s))  # symmetric matrix -> real, sorted eigenvalues
    return s_values, np.array(eigenvalues)


def parse_args():
    parser = argparse.ArgumentParser(description="Instantaneous spectrum of H(s) for instance (2.1).")
    parser.add_argument("--points", type=int, default=101, help="number of values of s. Default: 101.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "figures",
                        help="where the figure is saved (default: results/figures)")
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening it")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n = 3
    s_values, eigenvalues = espectro(FORMULA_2_1, n, args.points)

    # At s = 1 the zero-energy eigenvalues are exactly the satisfying assignments
    n_soluciones = int(np.sum(np.abs(eigenvalues[-1]) < 1e-9))
    print(f"Ground-state degeneracy at s = 1 (= number of solutions): {n_soluciones}")

    plt.figure(figsize=(10, 6))
    for i in range(eigenvalues.shape[1]):
        plt.plot(s_values, eigenvalues[:, i], label=f'Eigenvalue {i+1}')

    plt.xlabel('s', fontsize=20)
    plt.ylabel('Energy', fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_dir / "energy_spectrum.png", dpi=200, bbox_inches="tight")
    print(f"Saved {args.out_dir / 'energy_spectrum.png'}")

    if args.no_show:
        plt.close()
    else:
        plt.show()
