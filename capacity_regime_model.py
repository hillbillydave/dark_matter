# ============================================================
# Capacity-Regime Dark Matter Model — Hybrid v3.0
#   - Sofue-style MW baryons
#   - Two-term Pi_O(r)
#   - Extended E(r)
#   - Free alpha
#   - Physical priors
#   - Nelder–Mead χ² fit
#   - χ² heatmaps
#   - Auto-run MW + dwarf + stress sweeps
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
import os

G = 4.30091e-6  # kpc (km/s)^2 / Msun

# ------------------------------------------------------------
# Global plotting style (paper-ready)
# ------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (6.0, 4.5),
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ============================================================
# Radial grid
# ============================================================

r_min, r_max, n_r = 0.1, 50.0, 2000
r = np.linspace(r_min, r_max, n_r)

# ============================================================
# Sofue-style Milky Way baryonic mass model (approximate)
# ============================================================

def M_bulge_sofue(r):
    M_b = 1.0e10  # Msun
    a_b = 0.7     # kpc
    return M_b * (r**2 / (r + a_b)**2)

def M_disk_sofue(r):
    M_d = 6.0e10  # Msun
    R_d = 3.0     # kpc
    x = r / R_d
    return M_d * (1.0 - (1.0 + x) * np.exp(-x))

def M_gas_sofue(r):
    M_g = 1.0e10  # Msun
    R_g = 7.0     # kpc
    x = r / R_g
    return M_g * (1.0 - (1.0 + x) * np.exp(-x))

def M_baryon_mw(r):
    return M_bulge_sofue(r) + M_disk_sofue(r) + M_gas_sofue(r)

# ============================================================
# Dwarf baryonic mass model (toy)
# ============================================================

def M_baryon_dwarf(r):
    M0 = 5e8
    r_s = 1.5
    x = r / (r + r_s)
    return M0 * x**2

# ============================================================
# Hybrid capacity-regime fields
# ============================================================

def E_field(r, E_bg=1.0, epsilon_E=0.3, r_E=10.0,
            eta=0.0, r_H=20.0, p=1.0):
    return E_bg * (
        1.0
        + epsilon_E * np.exp(-r / r_E)
        + eta * (r / r_H)**p
    )

def Pi_O(r,
         Pi0=1.0, r_P=8.0, n=1.0,
         Pi1=0.5, r_Q=20.0, m=1.0):
    term1 = Pi0 / (1.0 + (r / r_P)**n)
    term2 = Pi1 / (1.0 + (r / r_Q)**m)
    return term1 + term2

# ============================================================
# Effective dark matter density
# ============================================================

def rho_dm_eff(r, rho_star=1.0, alpha=2.0,
               E_kwargs=None, Pi_kwargs=None):
    if E_kwargs is None:
        E_kwargs = {}
    if Pi_kwargs is None:
        Pi_kwargs = {}

    E = E_field(r, **E_kwargs)
    Pi = Pi_O(r, **Pi_kwargs)
    return rho_star * (E / Pi)**alpha

# ============================================================
# Mass integration
# ============================================================

def M_from_density(r, rho):
    integrand = 4.0 * np.pi * rho * r**2
    return cumulative_trapezoid(integrand, r, initial=0.0)

# ============================================================
# Rotation curve
# ============================================================

def rotation_curve(r, M_baryon_func,
                   rho_star=1.0, alpha=2.0,
                   E_kwargs=None, Pi_kwargs=None):

    rho_dm = rho_dm_eff(r, rho_star=rho_star, alpha=alpha,
                        E_kwargs=E_kwargs, Pi_kwargs=Pi_kwargs)

    M_dm = M_from_density(r, rho_dm)
    M_b = M_baryon_func(r)
    M_tot = M_b + M_dm

    v_tot = np.sqrt(G * M_tot / r)
    v_b = np.sqrt(G * M_b / r)
    return v_tot, v_b, M_dm, rho_dm

# ============================================================
# Data loader
# ============================================================

def load_rotation_data(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["r_kpc"], data["v_obs_kms"], data["v_err_kms"]

# ============================================================
# χ² and physical priors
# ============================================================

def chi2_model_vs_data(r_model, v_model, r_data, v_data, v_err):
    v_interp = np.interp(r_data, r_model, v_model)
    return np.sum(((v_interp - v_data) / v_err)**2)

def apply_physical_priors(params):
    # params = (log_rho_star, log_Pi0, log_r_P,
    #           log_Pi1, log_r_Q, log_alpha,
    #           log_eta, log_r_H, log_p)
    (log_rho_star, log_Pi0, log_r_P,
     log_Pi1, log_r_Q, log_alpha,
     log_eta, log_r_H, log_p) = params

    rho_star = np.exp(log_rho_star)
    Pi0 = np.exp(log_Pi0)
    r_P = np.exp(log_r_P)
    Pi1 = np.exp(log_Pi1)
    r_Q = np.exp(log_r_Q)
    alpha = np.exp(log_alpha)
    eta = np.exp(log_eta)
    r_H = np.exp(log_r_H)
    p = np.exp(log_p)

    penalty = 0.0

    # Physical priors (soft penalties)
    # rho_star in [0.1, 10]
    if not (0.1 <= rho_star <= 10.0):
        penalty += 1e6

    # Pi0, Pi1 in [0.1, 10]
    if not (0.1 <= Pi0 <= 10.0):
        penalty += 1e6
    if not (0.1 <= Pi1 <= 10.0):
        penalty += 1e6

    # r_P, r_Q, r_H in [0.5, 50] kpc
    if not (0.5 <= r_P <= 50.0):
        penalty += 1e6
    if not (0.5 <= r_Q <= 50.0):
        penalty += 1e6
    if not (0.5 <= r_H <= 50.0):
        penalty += 1e6

    # alpha in [0.5, 4]
    if not (0.5 <= alpha <= 4.0):
        penalty += 1e6

    # eta in [0, 5]
    if not (0.0 <= eta <= 5.0):
        penalty += 1e6

    # p in [0.5, 3]
    if not (0.5 <= p <= 3.0):
        penalty += 1e6

    return penalty

def mw_chi2_params_hybrid(params, r, r_data, v_data, v_err):
    penalty = apply_physical_priors(params)
    if penalty > 0:
        return penalty

    (log_rho_star, log_Pi0, log_r_P,
     log_Pi1, log_r_Q, log_alpha,
     log_eta, log_r_H, log_p) = params

    rho_star = np.exp(log_rho_star)
    Pi0 = np.exp(log_Pi0)
    r_P = np.exp(log_r_P)
    Pi1 = np.exp(log_Pi1)
    r_Q = np.exp(log_r_Q)
    alpha = np.exp(log_alpha)
    eta = np.exp(log_eta)
    r_H = np.exp(log_r_H)
    p = np.exp(log_p)

    E_kwargs = {
        "E_bg": 1.0,
        "epsilon_E": 0.3,
        "r_E": 10.0,
        "eta": eta,
        "r_H": r_H,
        "p": p,
    }
    Pi_kwargs = {
        "Pi0": Pi0,
        "r_P": r_P,
        "n": 1.0,
        "Pi1": Pi1,
        "r_Q": r_Q,
        "m": 1.0,
    }

    v_tot, v_b, M_dm, rho_dm = rotation_curve(
        r,
        M_baryon_func=M_baryon_mw,
        rho_star=rho_star,
        alpha=alpha,
        E_kwargs=E_kwargs,
        Pi_kwargs=Pi_kwargs,
    )

    return chi2_model_vs_data(r, v_tot, r_data, v_data, v_err)

def fit_mw_parameters_hybrid(r, r_data, v_data, v_err):
    # Initial guesses (log-space)
    x0 = np.log([
        1.0,   # rho_star
        1.0,   # Pi0
        8.0,   # r_P
        0.5,   # Pi1
        20.0,  # r_Q
        2.0,   # alpha
        0.5,   # eta
        20.0,  # r_H
        1.0,   # p
    ])

    res = minimize(
        mw_chi2_params_hybrid,
        x0,
        args=(r, r_data, v_data, v_err),
        method="Nelder-Mead",
        options={"maxiter": 1500, "disp": False}
    )

    (log_rho_star, log_Pi0, log_r_P,
     log_Pi1, log_r_Q, log_alpha,
     log_eta, log_r_H, log_p) = res.x

    rho_star = np.exp(log_rho_star)
    Pi0 = np.exp(log_Pi0)
    r_P = np.exp(log_r_P)
    Pi1 = np.exp(log_Pi1)
    r_Q = np.exp(log_r_Q)
    alpha = np.exp(log_alpha)
    eta = np.exp(log_eta)
    r_H = np.exp(log_r_H)
    p = np.exp(log_p)

    best_params = {
        "rho_star": rho_star,
        "Pi0": Pi0,
        "r_P": r_P,
        "Pi1": Pi1,
        "r_Q": r_Q,
        "alpha": alpha,
        "eta": eta,
        "r_H": r_H,
        "p": p,
    }

    return best_params, res.fun

# ============================================================
# Plotting helpers
# ============================================================

def plot_mw_with_data_hybrid(r, r_data, v_data, v_err,
                             params,
                             save_name="mw_capacity_hybrid_fit.png",
                             title_suffix=""):

    E_kwargs = {
        "E_bg": 1.0,
        "epsilon_E": 0.3,
        "r_E": 10.0,
        "eta": params["eta"],
        "r_H": params["r_H"],
        "p": params["p"],
    }
    Pi_kwargs = {
        "Pi0": params["Pi0"],
        "r_P": params["r_P"],
        "n": 1.0,
        "Pi1": params["Pi1"],
        "r_Q": params["r_Q"],
        "m": 1.0,
    }

    v_tot, v_b, M_dm, rho_dm = rotation_curve(
        r,
        M_baryon_func=M_baryon_mw,
        rho_star=params["rho_star"],
        alpha=params["alpha"],
        E_kwargs=E_kwargs,
        Pi_kwargs=Pi_kwargs,
    )

    plt.figure()
    plt.plot(r, v_tot, label="v_c total (model)")
    plt.plot(r, v_b, "--", label="baryons only")
    plt.errorbar(r_data, v_data, yerr=v_err,
                 fmt="o", ms=4, alpha=0.7,
                 label="Milky Way data (Eilers+2019)")

    plt.xlabel("r [kpc]")
    plt.ylabel("v_c [km/s]")
    plt.legend()
    plt.title(f"Milky Way — Hybrid Capacity-Regime {title_suffix}")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

def plot_dwarf_with_data_hybrid(r, r_data, v_data, v_err,
                                params,
                                save_name="draco_capacity_hybrid.png",
                                dwarf_name="Draco"):

    E_kwargs = {
        "E_bg": 1.1,
        "epsilon_E": 0.2,
        "r_E": 10.0,
        "eta": params["eta"],
        "r_H": params["r_H"],
        "p": params["p"],
    }
    Pi_kwargs = {
        "Pi0": 0.3 * params["Pi0"],
        "r_P": 0.5 * params["r_P"],
        "n": 1.0,
        "Pi1": 0.3 * params["Pi1"],
        "r_Q": 0.5 * params["r_Q"],
        "m": 1.0,
    }

    v_tot, v_b, M_dm, rho_dm = rotation_curve(
        r,
        M_baryon_func=M_baryon_dwarf,
        rho_star=params["rho_star"],
        alpha=params["alpha"],
        E_kwargs=E_kwargs,
        Pi_kwargs=Pi_kwargs,
    )

    plt.figure()
    plt.plot(r, v_tot, label="v_c total (model)")
    plt.plot(r, v_b, "--", label="baryons only")
    plt.errorbar(r_data, v_data, yerr=v_err,
                 fmt="o", ms=4, alpha=0.7,
                 label=f"{dwarf_name} data")

    plt.xlabel("r [kpc]")
    plt.ylabel("v_c [km/s]")
    plt.legend()
    plt.title(f"{dwarf_name} — Hybrid Capacity-Regime")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

# ============================================================
# χ² heatmaps (Pi0–r_P and eta–r_H slices)
# ============================================================

def mw_chi2_heatmap_Pi_rP(r, r_data, v_data, v_err,
                           base_params,
                           Pi0_grid=np.linspace(0.3, 3.0, 30),
                           rP_grid=np.linspace(2.0, 20.0, 30),
                           save_name="mw_chi2_Pi_rP.png"):

    chi2_map = np.zeros((len(Pi0_grid), len(rP_grid)))

    for i, Pi0 in enumerate(Pi0_grid):
        for j, r_P in enumerate(rP_grid):
            E_kwargs = {
                "E_bg": 1.0,
                "epsilon_E": 0.3,
                "r_E": 10.0,
                "eta": base_params["eta"],
                "r_H": base_params["r_H"],
                "p": base_params["p"],
            }
            Pi_kwargs = {
                "Pi0": Pi0,
                "r_P": r_P,
                "n": 1.0,
                "Pi1": base_params["Pi1"],
                "r_Q": base_params["r_Q"],
                "m": 1.0,
            }

            v_tot, v_b, M_dm, rho_dm = rotation_curve(
                r,
                M_baryon_func=M_baryon_mw,
                rho_star=base_params["rho_star"],
                alpha=base_params["alpha"],
                E_kwargs=E_kwargs,
                Pi_kwargs=Pi_kwargs,
            )

            chi2_map[i, j] = chi2_model_vs_data(r, v_tot, r_data, v_data, v_err)

    plt.figure(figsize=(6,5))
    X, Y = np.meshgrid(rP_grid, Pi0_grid)
    im = plt.pcolormesh(X, Y, chi2_map, shading="auto", cmap="viridis")
    plt.colorbar(im, label=r"$\chi^2$")
    plt.xlabel(r"$r_P$ [kpc]")
    plt.ylabel(r"$\Pi_0$")
    plt.title("MW χ² landscape — (Pi0, r_P)")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

def mw_chi2_heatmap_eta_rH(r, r_data, v_data, v_err,
                           base_params,
                           eta_grid=np.linspace(0.0, 3.0, 30),
                           rH_grid=np.linspace(5.0, 40.0, 30),
                           save_name="mw_chi2_eta_rH.png"):

    chi2_map = np.zeros((len(eta_grid), len(rH_grid)))

    for i, eta in enumerate(eta_grid):
        for j, r_H in enumerate(rH_grid):
            E_kwargs = {
                "E_bg": 1.0,
                "epsilon_E": 0.3,
                "r_E": 10.0,
                "eta": eta,
                "r_H": r_H,
                "p": base_params["p"],
            }
            Pi_kwargs = {
                "Pi0": base_params["Pi0"],
                "r_P": base_params["r_P"],
                "n": 1.0,
                "Pi1": base_params["Pi1"],
                "r_Q": base_params["r_Q"],
                "m": 1.0,
            }

            v_tot, v_b, M_dm, rho_dm = rotation_curve(
                r,
                M_baryon_func=M_baryon_mw,
                rho_star=base_params["rho_star"],
                alpha=base_params["alpha"],
                E_kwargs=E_kwargs,
                Pi_kwargs=Pi_kwargs,
            )

            chi2_map[i, j] = chi2_model_vs_data(r, v_tot, r_data, v_data, v_err)

    plt.figure(figsize=(6,5))
    X, Y = np.meshgrid(rH_grid, eta_grid)
    im = plt.pcolormesh(X, Y, chi2_map, shading="auto", cmap="viridis")
    plt.colorbar(im, label=r"$\chi^2$")
    plt.xlabel(r"$r_H$ [kpc]")
    plt.ylabel(r"$\eta$")
    plt.title("MW χ² landscape — (eta, r_H)")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

# ============================================================
# Stress-test sweep (auto-save)
# ============================================================

def auto_stress_test(r, out_dir="stress_plots_v3"):
    os.makedirs(out_dir, exist_ok=True)

    alpha_values = (1.0, 2.0, 3.0)
    eta_values = (0.0, 0.5, 1.0)
    Pi0_values = (0.5, 1.0, 2.0)
    Pi1_values = (0.2, 0.5, 1.0)
    r_P_values = (4.0, 8.0)
    r_Q_values = (15.0, 25.0)

    idx = 0
    for alpha in alpha_values:
        for eta in eta_values:
            for Pi0 in Pi0_values:
                for Pi1 in Pi1_values:
                    for r_P in r_P_values:
                        for r_Q in r_Q_values:
                            E_kwargs = {
                                "E_bg": 1.0,
                                "epsilon_E": 0.3,
                                "r_E": 10.0,
                                "eta": eta,
                                "r_H": 20.0,
                                "p": 1.0,
                            }
                            Pi_kwargs = {
                                "Pi0": Pi0,
                                "r_P": r_P,
                                "n": 1.0,
                                "Pi1": Pi1,
                                "r_Q": r_Q,
                                "m": 1.0,
                            }

                            v_tot, v_b, M_dm, rho_dm = rotation_curve(
                                r,
                                M_baryon_func=M_baryon_mw,
                                rho_star=1.0,
                                alpha=alpha,
                                E_kwargs=E_kwargs,
                                Pi_kwargs=Pi_kwargs,
                            )

                            plt.figure()
                            plt.plot(r, v_tot, label="v_c total")
                            plt.plot(r, v_b, "--", label="baryons only")
                            plt.xlabel("r [kpc]")
                            plt.ylabel("v_c [km/s]")
                            plt.legend()
                            plt.title(
                                f"alpha={alpha}, eta={eta}, "
                                f"Pi0={Pi0}, Pi1={Pi1}, "
                                f"r_P={r_P}, r_Q={r_Q}"
                            )
                            plt.tight_layout()

                            fname = (
                                f"stress_a{alpha}_eta{eta}_Pi0{Pi0}_Pi1{Pi1}_"
                                f"rP{r_P}_rQ{r_Q}_{idx}.png"
                            )
                            plt.savefig(os.path.join(out_dir, fname), dpi=200)
                            plt.close()
                            idx += 1

# ============================================================
# Auto-run blocks
# ============================================================

def auto_run_mw():
    print("🔍 Checking for mw_rotation_data.csv...")
    if not os.path.exists("mw_rotation_data.csv"):
        print("❌ mw_rotation_data.csv not found.")
        return

    print("📄 Loading Milky Way data...")
    r_data, v_data, v_err = load_rotation_data("mw_rotation_data.csv")

    print("🔧 Fitting hybrid capacity-regime parameters (Nelder–Mead χ²)...")
    best_params, chi2_best = fit_mw_parameters_hybrid(
        r, r_data, v_data, v_err
    )

    print("✅ Best-fit parameters (hybrid):")
    for k, v in best_params.items():
        print(f"   {k} = {v:.4g}")
    print(f"   χ² = {chi2_best:.2f}")

    print("📊 Generating paper-ready MW figure...")
    plot_mw_with_data_hybrid(
        r,
        r_data,
        v_data,
        v_err,
        params=best_params,
        save_name="mw_capacity_hybrid_fit.png",
        title_suffix="(hybrid capacity + Sofue baryons)",
    )

    print("🗺 Generating MW χ² heatmap (Pi0, r_P)...")
    mw_chi2_heatmap_Pi_rP(
        r,
        r_data,
        v_data,
        v_err,
        base_params=best_params,
        save_name="mw_chi2_Pi_rP.png",
    )

    print("🗺 Generating MW χ² heatmap (eta, r_H)...")
    mw_chi2_heatmap_eta_rH(
        r,
        r_data,
        v_data,
        v_err,
        base_params=best_params,
        save_name="mw_chi2_eta_rH.png",
    )

    return best_params

def auto_run_dwarf(best_params):
    if not os.path.exists("draco_rotation_data.csv"):
        print("ℹ️ No draco_rotation_data.csv found — skipping dwarf auto-run.")
        return

    print("📄 Loading Draco data...")
    r_data, v_data, v_err = load_rotation_data("draco_rotation_data.csv")

    print("📊 Generating Draco hybrid capacity figure...")
    plot_dwarf_with_data_hybrid(
        r,
        r_data,
        v_data,
        v_err,
        params=best_params,
        save_name="draco_capacity_hybrid.png",
        dwarf_name="Draco",
    )

def auto_run_stress():
    print("🧪 Running hybrid stress-test sweep (this may take a while)...")
    auto_stress_test(r, out_dir="stress_plots_v3")
    print("✅ Stress-test plots saved in ./stress_plots_v3")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    best_params = auto_run_mw()
    auto_run_dwarf(best_params)
    auto_run_stress()
    print("🎯 Hybrid v3.0 auto-run complete.")
