"""
Plot the Grover success probability vs number of iterations from the pickle
written by ``src/grover_3sat.py``. One figure per number of solutions, showing
the mean over all simulated instances with that number of solutions.

For N = 3 the success probability depends only on the number of solutions
(not on which assignments they are), so averaging over random instances with
the same number of solutions is equivalent to simulating one of them.

Usage (from the repository root):
    python analysis/plot_grover.py                       # newest results/datos_grover_*.pkl
    python analysis/plot_grover.py results/datos_grover_200.pkl --no-show

Figures are saved as results/figures/grover_<n>_solutions.jpg.

Note: only load pickle files that you generated yourself (pickle can execute
arbitrary code when loading untrusted files).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Grover success probability vs iterations.")
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="pickle written by src/grover_3sat.py "
                             "(default: newest results/datos_grover_*.pkl)")
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening them.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    archivo = args.input
    if archivo is None:
        candidatos = sorted(RESULTS_DIR.glob("datos_grover_*.pkl"), key=lambda f: f.stat().st_mtime)
        if not candidatos:
            raise SystemExit("No results/datos_grover_*.pkl found. Run src/grover_3sat.py first.")
        archivo = candidatos[-1]

    # ---------- Style ----------
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 150
    })

    # ---------- Load data ----------
    with open(archivo, "rb") as f:
        datos = pickle.load(f)  # {number of solutions: [(reps, p_values), ...]}
    print(f"Data loaded from {archivo}")

    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Plots ----------
    for n_sol, curvas in sorted(datos.items()):

        reps = np.array(curvas[0][0])

        all_p = np.array([
            p_values
            for _, p_values in curvas
        ])

        media = np.mean(all_p, axis=0)

        plt.figure(figsize=(6, 4))

        plt.plot(
            reps,
            media,
            linewidth=2.5,
            label="Mean"
        )

        plt.xlabel("Grover iterations")
        plt.ylabel("Success probability")

        plt.grid(alpha=0.3)
        plt.legend()
        plt.xticks(reps.astype(int))
        plt.tight_layout()
        plt.savefig(
            out_dir / f"grover_{n_sol}_solutions.jpg",
            bbox_inches="tight"
        )
        print(f"Saved {out_dir / f'grover_{n_sol}_solutions.jpg'} ({len(curvas)} instances, {n_sol} solutions)")

        if args.no_show:
            plt.close()
        else:
            plt.show()
