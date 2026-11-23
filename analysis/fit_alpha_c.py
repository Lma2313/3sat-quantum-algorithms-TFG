"""
Estimate the satisfiability threshold alpha_c by fitting a 4-parameter
sigmoid to the fraction of UNSAT instances as a function of alpha = M/N.

    p_UNSAT(alpha) = A / (1 + exp(-k (alpha - alpha_c))) + B

The fit is weighted by the binomial standard error of each point (with a lower
bound of one instance so that points with p = 0 or 1 do not get a null error).
Parameter errors are the square root of the diagonal of the covariance matrix
returned by curve_fit (absolute_sigma=True); the +/- 1 sigma band of the fitted
curve is obtained by linear error propagation (delta method).

Input: JSON files with the key "unsat_por_alpha" ({alpha: {"total": ..., "unsat": ...}}),
such as the ones written by src/adiabatic_3sat.py and src/adiabatic_3sat_lite.py.
The file name must contain N, T, Nt and ND, e.g. N_8_T_100_Nt_15_ND_200[_M_3000].json
or adiabatic_lite_N8_T100_Nt30_ND200.json.

Output (in results/figures/ unless --out-dir is given):
    - one figure per file (data + fitted curve + sigma band),
    - one comparison figure with all the fitted curves,
    - one figure of alpha_c versus N (or versus Nt) with error bars and the
      reference value alpha_c = 4.267 (theoretical 3-SAT threshold).

Usage (from the repository root):
    python analysis/fit_alpha_c.py                          # all JSON files in results/
    python analysis/fit_alpha_c.py results/ --Nt 15 --variable N --no-show
    python analysis/fit_alpha_c.py results/ --N 8 --variable Nt    # vary Nt at fixed N
"""
 
import argparse
import glob
import json
import os
import re
from pathlib import Path
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
 
# ==============================================================
# Configuration
# ==============================================================

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Reference value of the threshold for random 3-SAT (large-N limit).
ALPHA_C_REFERENCIA = 4.267

# Output directory and name tag for the comparison figures (set in main()).
OUT_DIR = RESULTS_DIR / "figures"
TAG = ""
 
 
# ==============================================================
# Fit model
# ==============================================================
 
def sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))
 
 
def sigmoid4(x, k, x0, A, B):
    return A / (1 + np.exp(-k * (x - x0))) + B
 
 
def sigmoid4_jac(x, k, x0, A, B):
    """Partial derivatives of sigmoid4 with respect to (k, x0, A, B)."""
    s = sigmoid(x, k, x0)
    dk = A * (x - x0) * s * (1 - s)
    dx0 = -A * k * s * (1 - s)
    dA = s
    dB = np.ones_like(x)
    return np.stack([dk, dx0, dA, dB], axis=-1)  # shape (len(x), 4)
 
 
def banda_incertidumbre(x, popt, pcov, jac_func):
    """
    Linear error propagation (delta method):
        var(y(x)) = J(x) . pcov . J(x)^T
    where J(x) is the vector of partial derivatives of the model evaluated
    at x with respect to each parameter.

    Returns the standard deviation of y(x), sigma_y(x), with the same shape as x.
    """
    J = jac_func(x, *popt)  # (len(x), n_param)
    # var_y[i] = J[i] @ pcov @ J[i].T
    var_y = np.einsum("ij,jk,ik->i", J, pcov, J)
    var_y = np.clip(var_y, 0, None)  # avoid negative values from numerical noise
    return np.sqrt(var_y)
 
 
# ==============================================================
# Data reading
# ==============================================================
 
def leer_estadisticas(nombre_archivo):
    """
    Read a JSON file with SAT/UNSAT statistics and return the data lists.

    Returns
    -------
    alphas, total, unsat, sat, unsat_fraction, sat_fraction, unsat_fraction_error
    """
    with open(nombre_archivo, "r", encoding="utf-8") as f:
        datos = json.load(f)
 
    estadisticas = datos["unsat_por_alpha"]
 
    alphas, total, unsat, sat = [], [], [], []
    porcentaje_unsat, porcentaje_sat = [], []
    error_porcentaje_unsat = []
 
    for alpha_str, valores in estadisticas.items():
        t = valores["total"]
        u = valores["unsat"]
        s = t - u
        p_unsat = u / t
 
        alphas.append(float(alpha_str))
        total.append(t)
        unsat.append(u)
        sat.append(s)
        porcentaje_unsat.append(p_unsat)
        porcentaje_sat.append(s / t)
 
        # Standard error of a binomial proportion: sqrt(p(1-p)/n).
        # With p = 0 or p = 1 the binomial error vanishes; a lower bound of one
        # instance is used so that error bars are not null when the sample size
        # is in fact finite.
        u_clip = min(max(u, 1), t - 1) if t > 1 else u
        p_clip = u_clip / t if t > 0 else p_unsat
        err = np.sqrt(p_clip * (1 - p_clip) / t) if t > 0 else 0.0
        error_porcentaje_unsat.append(err)
 
    return (
        alphas, total, unsat, sat,
        porcentaje_unsat, porcentaje_sat,
        error_porcentaje_unsat,
    )
 
 
def obtener_parametros(nombre):
    """
    Extract N, T, Nt, ND (and the number of samples M, if present) from a file
    name. Accepted patterns: N_4_T_100_Nt_15_ND_200[_M_3000] and
    N4_T100_Nt15_ND200 (names written by src/adiabatic_3sat*.py).
    Returns None if the name does not match.
    """
    m = re.search(r"N_?(\d+)_T_?(\d+)_Nt_?(\d+)_ND_?(\d+)(?:_M_(\d+))?", os.path.basename(nombre))
    if m is None:
        return None
    return {
        "N": int(m.group(1)),
        "T": int(m.group(2)),
        "Nt": int(m.group(3)),
        "ND": int(m.group(4)),
        "M": int(m.group(5)) if m.group(5) else None,
    }
 
 
def filtrar_y_ordenar_datos(archivos, N_permitidos, Nt_permitidos):
    """
    Filter the files by N and Nt (None = keep all) and return them sorted by N,
    together with their already extracted parameters (avoids parsing the name twice).
    """
    seleccionados = []
    for archivo in archivos:
        parametros = obtener_parametros(archivo)
        if parametros is None:
            print(f"Skipping {archivo}: the name does not contain N, T, Nt and ND.")
            continue
        if ((N_permitidos is None or parametros["N"] in N_permitidos)
                and (Nt_permitidos is None or parametros["Nt"] in Nt_permitidos)):
            seleccionados.append((archivo, parametros))
 
    seleccionados.sort(key=lambda item: item[1]["N"])
    return seleccionados
 
 
# ==============================================================
# Fit and evaluation
# ==============================================================
 
def R2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot
 
 
def ajustar_sigmoide4(alphas, porcentaje_unsat, sigma=None):
    """
    Fit the 4-parameter sigmoid and return the optimal parameters, their
    standard errors (square root of the covariance diagonal), the full
    covariance matrix (used to propagate the uncertainty to the fitted curve)
    and R^2.

    If `sigma` (error of each experimental point) is given, curve_fit performs
    a weighted fit and, with absolute_sigma=True, the returned uncertainties
    are interpreted in the same units as `sigma` instead of being rescaled to
    force chi2_red = 1.
    """
    usar_sigma = sigma is not None and np.all(np.array(sigma) > 0)
 
    kwargs = {}
    if usar_sigma:
        kwargs["sigma"] = sigma
        kwargs["absolute_sigma"] = True
 
    popt4, pcov4 = curve_fit(
        sigmoid4, alphas, porcentaje_unsat, p0=[5, 4.2, 1, 0], **kwargs
    )
 
    perr4 = np.sqrt(np.diag(pcov4))
 
    r2_4 = R2(porcentaje_unsat, sigmoid4(alphas, *popt4))
 
    return popt4, pcov4, perr4, r2_4
 
 
def imprimir_resultados(archivo, popt4, perr4, r2_4):
    print("=" * 75)
    print(archivo)
 
    print("\n--- Sigmoid fit ---")
    print(f"k        = {popt4[0]:10.6f} +/- {perr4[0]:.6f}")
    print(f"alpha_c  = {popt4[1]:10.6f} +/- {perr4[1]:.6f}")
    print(f"A        = {popt4[2]:10.6f} +/- {perr4[2]:.6f}")
    print(f"B        = {popt4[3]:10.6f} +/- {perr4[3]:.6f}")
    print(f"R2       = {r2_4:10.6f}")
    print("=" * 75)
 
 
# ==============================================================
# Individual plots
# ==============================================================
 
def _texto_4param(popt4, perr4, r2_4):
    return (
        rf"$k={popt4[0]:.3f}\pm{perr4[0]:.3f}$" "\n"
        rf"$\alpha_c={popt4[1]:.3f}\pm{perr4[1]:.3f}$" "\n"
        rf"$A={popt4[2]:.3f}\pm{perr4[2]:.3f}$" "\n"
        rf"$B={popt4[3]:.3f}\pm{perr4[3]:.3f}$" "\n"
        rf"$R^2={r2_4:.5f}$"
    )
 
 
def graficar_ajuste(
    archivo, alphas, porcentaje_unsat, error_unsat,
    x, y_ajuste, sigma_y_ajuste, alpha_c, texto, sufijo,
):
    """
    Create and save one figure (data + fitted curve) for a file.

    Includes vertical error bars on the data (if `error_unsat` is given) and a
    shaded +/- 1 sigma band around the fitted curve, obtained by propagating
    the errors of the fit parameters.
    """
    fig = plt.figure()
    fig.canvas.manager.set_window_title(
        os.path.basename(archivo).replace(".json", "") + f"_{sufijo}"
    )
 
    if error_unsat is not None:
        plt.errorbar(
            alphas, porcentaje_unsat, yerr=error_unsat,
            fmt="o", ms=5, capsize=3, elinewidth=1, label="Data",
        )
    else:
        plt.scatter(alphas, porcentaje_unsat, s=40, label="Data")
 
    plt.plot(x, y_ajuste, linewidth=2, label="Fit")
 
    if sigma_y_ajuste is not None:
        plt.fill_between(
            x, y_ajuste - sigma_y_ajuste, y_ajuste + sigma_y_ajuste,
            alpha=0.25, label=r"$\pm 1\sigma$ (fit parameters)",
        )
 
    plt.axvline(alpha_c, color="0.7", linestyle="--", linewidth=1.5, alpha=0.6)
 
    plt.text(
        0.03, 0.97, texto,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )
 
    plt.xlabel(r"$\alpha$")
    plt.ylabel("UNSAT probability")
    plt.xlim(alphas.min(), alphas.max())
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
 
    plt.savefig(OUT_DIR / (Path(archivo).stem + f"_{sufijo}.png"), dpi=300, bbox_inches="tight")
 
 
# ==============================================================
# Comparison plots (all curves together)
# ==============================================================
 
FIG_COMPARATIVA_4P = 101
FIG_ALPHA_C = 200
 
 
def preparar_figuras_comparativas():
    plt.figure(FIG_COMPARATIVA_4P)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("UNSAT fraction")
    plt.grid(alpha=0.3)
 
 
def anadir_a_figuras_comparativas(x, popt4, pcov4, etiqueta, mostrar_banda=True):
    plt.figure(FIG_COMPARATIVA_4P)
    y4 = sigmoid4(x, *popt4)
    linea4, = plt.plot(x, y4, label=etiqueta)
    if mostrar_banda:
        sigma4 = banda_incertidumbre(x, popt4, pcov4, sigmoid4_jac)
        plt.fill_between(
            x, y4 - sigma4, y4 + sigma4,
            color=linea4.get_color(), alpha=0.15,
        )
 
 
def guardar_figuras_comparativas():
    plt.figure(FIG_COMPARATIVA_4P)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"comparison_4_params{TAG}.png", dpi=300, bbox_inches="tight")
 
 
def graficar_variable_vs_alpha_c(resultados, variable):
    """
    Final plot: variable ("N" or "Nt") versus alpha_c, with the 4-parameter
    fit, its error bars (1 sigma, from the covariance matrix of the fit) and a
    reference line at ALPHA_C_REFERENCIA.
    """
    resultados = sorted(resultados, key=lambda r: r[variable])
    x_vals = [r[variable] for r in resultados]
    alpha_c_4 = [r["alpha_c_4"] for r in resultados]
    err_alpha_c_4 = [r["err_alpha_c_4"] for r in resultados]
 
    etiqueta_x = r"$N$" if variable == "N" else r"$N_t$"
 
    plt.figure(FIG_ALPHA_C)
    plt.errorbar(
        x_vals, alpha_c_4, yerr=err_alpha_c_4,
        fmt="s-", capsize=4, elinewidth=1.2,
        label=r"$\alpha_c$",
    )

    plt.axhline(
        ALPHA_C_REFERENCIA,
        color="0.4",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\alpha_c={ALPHA_C_REFERENCIA}$ (reference)",
    )
 
    plt.xlabel(etiqueta_x)
    plt.ylabel(r"$\alpha_c$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.ylim(4, 5)
    plt.savefig(OUT_DIR / f"{variable}_vs_alpha_c{TAG}.png", dpi=300, bbox_inches="tight")
 
 
# ==============================================================
# Main program
# ==============================================================
 
def main(archivos, N_permitidos, Nt_permitidos, variable_analisis, out_dir, mostrar):
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })
 
    datos_filtrados = filtrar_y_ordenar_datos(archivos, N_permitidos, Nt_permitidos)
    if not datos_filtrados:
        raise SystemExit("No data files match the filters (check the file names and --N / --Nt).")

    # name tag for the comparison figures: fixed values of Nt and/or N
    global OUT_DIR, TAG
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nts = {p["Nt"] for _, p in datos_filtrados}
    ns = {p["N"] for _, p in datos_filtrados}
    TAG = (f"_Nt{nts.pop()}" if len(nts) == 1 else "") + (f"_N{ns.pop()}" if len(ns) == 1 else "")
    preparar_figuras_comparativas()
 
    resultados = []
 
    for archivo, parametros in datos_filtrados:
        etiqueta = (
            rf"$N={parametros['N']}$" if variable_analisis == "N"
            else rf"$N_t={parametros['Nt']}$"
        )
 
        (
            alphas, _, _, _,
            porcentaje_unsat, _, error_unsat,
        ) = leer_estadisticas(archivo)
        alphas = np.array(alphas)
        porcentaje_unsat = np.array(porcentaje_unsat)
        error_unsat = np.array(error_unsat)

        if len(alphas) < 5:  # 4 parameters: at least 5 points are needed for a meaningful fit
            print(f"Skipping {archivo}: only {len(alphas)} values of alpha (at least 5 are needed).")
            continue
 
        popt4, pcov4, perr4, r2_4 = ajustar_sigmoide4(
            alphas, porcentaje_unsat, sigma=error_unsat
        )
        imprimir_resultados(archivo, popt4, perr4, r2_4)
 
        x = np.linspace(alphas.min(), alphas.max(), 500)
 
        y4 = sigmoid4(x, *popt4)
        sigma_y4 = banda_incertidumbre(x, popt4, pcov4, sigmoid4_jac)
 
        graficar_ajuste(
            archivo, alphas, porcentaje_unsat, error_unsat,
            x, y4, sigma_y4, popt4[1],
            _texto_4param(popt4, perr4, r2_4), "4param",
        )
 
        anadir_a_figuras_comparativas(x, popt4, pcov4, etiqueta)
 
        resultados.append({
            "N": parametros["N"],
            "Nt": parametros["Nt"],
            "alpha_c_4": popt4[1],
            "err_alpha_c_4": perr4[1],
            "r2_4": r2_4,
        })
 
    if not resultados:
        raise SystemExit("No file had enough values of alpha to fit.")

    guardar_figuras_comparativas()
    graficar_variable_vs_alpha_c(resultados, variable_analisis)
 

    if mostrar:
        plt.show()
    else:
        plt.close("all")


def reunir_archivos(rutas):
    """Expand directories into their *.json files (default: results/); fail with a clear message on bad paths."""
    rutas = rutas or [RESULTS_DIR]
    archivos = []
    for ruta in rutas:
        ruta = Path(ruta)
        if ruta.is_dir():
            encontrados = sorted(glob.glob(str(ruta / "*.json")))
            if not encontrados:
                raise SystemExit(f"No .json files found in {ruta}")
            archivos += encontrados
        elif ruta.is_file():
            archivos.append(str(ruta))
        else:
            raise SystemExit(f"Path not found: {ruta} (run the command from the repository root)")
    return archivos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit alpha_c from p(UNSAT) vs alpha curves with a 4-parameter sigmoid.")
    parser.add_argument("data", nargs="*", type=Path,
                        help="JSON files or directories (default: results/)")
    parser.add_argument("--N", type=int, nargs="+", default=None, help="keep only these values of N")
    parser.add_argument("--Nt", type=int, nargs="+", default=None, help="keep only these values of Nt")
    parser.add_argument("--variable", choices=["N", "Nt"], default="N",
                        help="variable of the final alpha_c plot (default: N)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "figures",
                        help="where figures are saved (default: results/figures)")
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening them")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(reunir_archivos(args.data), args.N, args.Nt, args.variable, args.out_dir, not args.no_show)
