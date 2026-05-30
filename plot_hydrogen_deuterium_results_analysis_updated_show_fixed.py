# -*- coding: utf-8 -*-
"""
Hydrogen and Deuterium Balmer-alpha analysis
===========================================

This script analyzes the H/D lamp data near the Balmer-alpha transition.
It intentionally does NOT use the full hydrogen spectrum for the article plots.
The full hydrogen file is only used to extract/check the Balmer-alpha region;
the isotope-shift result is extracted from the deuterium lamp itself, where the
residual hydrogen line is an internal wavelength reference.

Outputs saved in DATA_DIR:
    1) hydrogen_deuterium_scan_quality.png
    2) hydrogen_deuterium_best_double_gaussian_fit.png  (single-panel fit figure)
    3) hydrogen_deuterium_fit_results.csv
    4) hydrogen_deuterium_best_fit_curve.csv
    5) hydrogen_balmer_alpha_window.csv
    6) hydrogen_deuterium_article_numbers.txt
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import constants as const


# -----------------------------------------------------------------------------
# 1. USER SETTINGS
# -----------------------------------------------------------------------------

# Your local folder. The fallback below lets the script also run if you place it
# directly inside the data folder or run it in a notebook environment.
DATA_DIR = Path(r"C:\Users\orlyk\Desktop\Jonathan k\School\lab job\lab C\Spectroscopy\Data – Results – Graphs\Hydrogen and Deuterium")

if not DATA_DIR.exists():
    # Useful when the script is copied into the folder and run from there.
    DATA_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = DATA_DIR

# Set True to display the figures on screen after saving them.
# In PyCharm/regular Python this opens the matplotlib figure windows.
SHOW_FIGURES = True

# Theory: the useful visible isotope shift is at the Balmer-alpha transition,
# n_i = 3 -> n_f = 2, around 656 nm.
DEUTERIUM_ANALYSIS_WINDOW_NM = (655.40, 656.30)
DEUTERIUM_PLOT_WINDOW_NM = (655.30, 656.55)
HYDROGEN_ALPHA_SEARCH_WINDOW_NM = (640.0, 670.0)

# High-resolution scans should have a fitted Gaussian width much smaller than
# the H-D separation. The broad-slit files are still plotted, but not used for
# the final isotope-shift number if they fail this criterion.
MAX_ACCEPTED_SIGMA_NM = 0.08
MIN_ACCEPTED_SHIFT_NM = 0.10
MAX_ACCEPTED_SHIFT_NM = 0.25

# Values used only for reference annotations.
THEORY_H_ALPHA_AIR_NM = 656.281
THEORY_D_ALPHA_AIR_NM = 656.103
THEORY_SHIFT_NM = THEORY_H_ALPHA_AIR_NM - THEORY_D_ALPHA_AIR_NM

# Proton/electron mass ratio. For this undergraduate isotope-shift analysis,
# m_H is approximated by the proton mass, as in the standard derivation.
M_H_OVER_M_E = const.value("proton mass") / const.value("electron mass")
M_E_OVER_M_H = 1.0 / M_H_OVER_M_E
THEORY_M_D_OVER_M_H = const.value("deuteron mass") / const.value("proton mass")


# -----------------------------------------------------------------------------
# 2. MODELS AND GENERAL UTILITIES
# -----------------------------------------------------------------------------

def read_two_column_spectrum(path: Path) -> pd.DataFrame:
    """Read a two-column spectrum file: wavelength [nm], intensity [a.u.]."""
    df = pd.read_csv(
        path,
        sep=r"[\s,;]+",
        engine="python",
        header=None,
        comment="#",
        names=["wavelength_nm", "intensity_au"],
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("wavelength_nm").reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"File has too few numeric rows: {path.name}")
    return df


def gaussian(x: np.ndarray, c: float, a: float, mu: float, sigma: float) -> np.ndarray:
    return c + a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def double_gaussian_common_sigma(
    x: np.ndarray,
    c: float,
    a_d: float,
    mu_d: float,
    a_h: float,
    mu_h: float,
    sigma: float,
) -> np.ndarray:
    """Two Gaussians with one common instrumental width."""
    d_line = a_d * np.exp(-0.5 * ((x - mu_d) / sigma) ** 2)
    h_line = a_h * np.exp(-0.5 * ((x - mu_h) / sigma) ** 2)
    return c + d_line + h_line


def deuterium_component(x: np.ndarray, a_d: float, mu_d: float, sigma: float) -> np.ndarray:
    return a_d * np.exp(-0.5 * ((x - mu_d) / sigma) ** 2)


def hydrogen_component(x: np.ndarray, a_h: float, mu_h: float, sigma: float) -> np.ndarray:
    return a_h * np.exp(-0.5 * ((x - mu_h) / sigma) ** 2)


def discover_files(data_dir: Path) -> tuple[Path | None, list[Path]]:
    """Find the one hydrogen spectrum and all deuterium spectra in DATA_DIR."""
    candidates = []
    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if (
            name.startswith("hydrogen_deuterium_")
            or name.startswith("plot_hydrogen_deuterium")
            or name.startswith("hydrogen_balmer_alpha")
        ):
            continue
        if p.suffix.lower() in {".png", ".pdf", ".bib", ".py", ".docx", ".xlsx"}:
            continue
        # Raw files in this experiment may have no suffix, .csv, or names such as
        # Deuterium_in0.3mm_out0.2mm, where pathlib treats ".2mm" as a suffix.
        # Therefore we skip only clearly non-spectrum/generated file types above,
        # and then let read_two_column_spectrum() verify whether the file is numeric.
        candidates.append(p)

    hydrogen_candidates = []
    deuterium_files = []
    for p in candidates:
        name = p.name.lower()
        is_deuterium = ("deuterium" in name) or ("diatorium" in name) or ("deuter" in name)
        is_hydrogen = ("hydrogen" in name) and not is_deuterium
        if is_hydrogen:
            hydrogen_candidates.append(p)
        if is_deuterium:
            deuterium_files.append(p)

    hydrogen_file = None
    if hydrogen_candidates:
        # Prefer the raw file named exactly "hydrogen" if it exists.
        exact = [p for p in hydrogen_candidates if p.name.lower() == "hydrogen"]
        hydrogen_file = exact[0] if exact else sorted(hydrogen_candidates, key=lambda q: len(q.name))[0]

    deuterium_files = sorted(deuterium_files, key=lambda q: q.name.lower())
    return hydrogen_file, deuterium_files


def crop(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return df[(df["wavelength_nm"] >= lo) & (df["wavelength_nm"] <= hi)].copy()


# -----------------------------------------------------------------------------
# 3. FITTING FUNCTIONS
# -----------------------------------------------------------------------------

def fit_hydrogen_alpha_window(hydrogen_df: pd.DataFrame) -> dict:
    """
    Use only the Balmer-alpha region of the full hydrogen spectrum.
    This is a diagnostic check, not the isotope-shift measurement.
    """
    hwin = crop(hydrogen_df, *HYDROGEN_ALPHA_SEARCH_WINDOW_NM)
    if len(hwin) < 20:
        raise RuntimeError("Hydrogen Balmer-alpha search window has too few points.")

    x = hwin["wavelength_nm"].to_numpy()
    y = hwin["intensity_au"].to_numpy()

    # Select the largest local peak in the 640--670 nm region.
    peaks, _ = find_peaks(y, prominence=max(0.05 * np.ptp(y), 1e-6), distance=20)
    if len(peaks) == 0:
        peak_index = int(np.argmax(y))
    else:
        peak_index = int(peaks[np.argmax(y[peaks])])
    mu0 = float(x[peak_index])

    # Fit only around the selected line.
    fit_win = hwin[(hwin["wavelength_nm"] >= mu0 - 1.5) & (hwin["wavelength_nm"] <= mu0 + 1.5)]
    xfit = fit_win["wavelength_nm"].to_numpy()
    yfit = fit_win["intensity_au"].to_numpy()
    c0 = float(np.median(np.r_[yfit[:10], yfit[-10:]]))
    a0 = float(np.max(yfit) - c0)
    p0 = [c0, max(a0, 1e-9), mu0, 0.15]
    bounds = ([-np.inf, 0.0, mu0 - 0.5, 0.01], [np.inf, np.inf, mu0 + 0.5, 1.0])

    popt, pcov = curve_fit(gaussian, xfit, yfit, p0=p0, bounds=bounds, maxfev=100000)
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)

    out_df = pd.DataFrame({
        "wavelength_nm": xfit,
        "intensity_au": yfit,
        "fit_au": gaussian(xfit, *popt),
        "residual_au": yfit - gaussian(xfit, *popt),
    })
    out_df.to_csv(OUTPUT_DIR / "hydrogen_balmer_alpha_window.csv", index=False)

    return {
        "center_nm": popt[2],
        "center_unc_nm": perr[2],
        "sigma_nm": popt[3],
        "amplitude_au": popt[1],
        "window_df": out_df,
    }


def mass_ratio_from_shift(delta_nm: float, lambda_h_nm: float) -> float:
    """Khundkar-style Balmer isotope-shift mass-ratio formula."""
    factor = M_H_OVER_M_E * (1.0 + M_E_OVER_M_H) * (delta_nm / lambda_h_nm)
    return 1.0 / (1.0 - factor)


def mc_uncertainties_from_covariance(popt: np.ndarray, pcov: np.ndarray, n: int = 20000) -> dict:
    """Monte-Carlo error propagation from the double-Gaussian covariance matrix."""
    keys = ["c", "a_d", "lambda_d_nm", "a_h", "lambda_h_nm", "sigma_nm"]
    result = {f"{k}_unc": np.nan for k in keys}
    result.update({"delta_nm_unc": np.nan, "mass_ratio_unc": np.nan, "c_d_percent_unc": np.nan, "c_h_percent_unc": np.nan})

    if pcov is None or not np.all(np.isfinite(pcov)):
        return result

    # Make covariance numerically symmetric and positive semi-definite.
    pcov = 0.5 * (pcov + pcov.T)
    try:
        samples = np.random.default_rng(12345).multivariate_normal(popt, pcov, size=n, check_valid="ignore")
    except Exception:
        return result

    # Keep only physical samples.
    c, a_d, mu_d, a_h, mu_h, sigma = samples.T
    mask = (a_d > 0) & (a_h > 0) & (sigma > 0) & (mu_h > mu_d)
    samples = samples[mask]
    if len(samples) < 100:
        return result

    c, a_d, mu_d, a_h, mu_h, sigma = samples.T
    delta = mu_h - mu_d
    mass_ratio = mass_ratio_from_shift(delta, mu_h)
    area_d = np.sqrt(2.0 * np.pi) * a_d * sigma
    area_h = np.sqrt(2.0 * np.pi) * a_h * sigma
    c_d = 100.0 * area_d / (area_d + area_h)
    c_h = 100.0 * area_h / (area_d + area_h)

    vals = {
        "c": c,
        "a_d": a_d,
        "lambda_d_nm": mu_d,
        "a_h": a_h,
        "lambda_h_nm": mu_h,
        "sigma_nm": sigma,
    }
    for k, arr in vals.items():
        result[f"{k}_unc"] = float(np.std(arr, ddof=1))
    result["delta_nm_unc"] = float(np.std(delta, ddof=1))
    result["mass_ratio_unc"] = float(np.std(mass_ratio, ddof=1))
    result["c_d_percent_unc"] = float(np.std(c_d, ddof=1))
    result["c_h_percent_unc"] = float(np.std(c_h, ddof=1))
    return result


def analyze_deuterium_file(path: Path) -> tuple[dict, pd.DataFrame]:
    df = read_two_column_spectrum(path)
    dwin = crop(df, *DEUTERIUM_ANALYSIS_WINDOW_NM)
    if len(dwin) < 20:
        raise RuntimeError(f"Not enough points in analysis window for {path.name}")

    x = dwin["wavelength_nm"].to_numpy()
    y = dwin["intensity_au"].to_numpy()

    # Baseline and initial guesses.
    edge_mask = (x < 655.55) | (x > 656.20)
    c0 = float(np.median(y[edge_mask])) if np.any(edge_mask) else float(np.min(y))
    y0 = y - c0

    left_mask = (x > 655.65) & (x < 655.88)
    right_mask = (x > 655.88) & (x < 656.15)
    if not np.any(left_mask) or not np.any(right_mask):
        raise RuntimeError(f"Could not create left/right peak masks for {path.name}")

    mu_d0 = float(x[left_mask][np.argmax(y0[left_mask])])
    mu_h0 = float(x[right_mask][np.argmax(y0[right_mask])])
    a_d0 = float(max(np.max(y0[left_mask]), 0.1 * np.ptp(y)))
    a_h0 = float(max(np.max(y0[right_mask]), 0.03 * np.ptp(y)))
    sigma0 = 0.045

    p0 = [c0, a_d0, mu_d0, a_h0, mu_h0, sigma0]
    bounds = (
        [-np.inf, 0.0, 655.62, 0.0, 655.90, 0.010],
        [ np.inf, np.inf, 655.88, np.inf, 656.20, 0.250],
    )

    popt, pcov = curve_fit(
        double_gaussian_common_sigma,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=100000,
    )

    c, a_d, mu_d, a_h, mu_h, sigma = popt
    y_fit = double_gaussian_common_sigma(x, *popt)
    d_comp = deuterium_component(x, a_d, mu_d, sigma)
    h_comp = hydrogen_component(x, a_h, mu_h, sigma)
    resid = y - y_fit

    noise_mask = (x < 655.55) | (x > 656.23)
    noise = float(np.std(y[noise_mask])) if np.sum(noise_mask) >= 5 else float(np.std(resid))
    dof = len(x) - len(popt)
    reduced_chi2 = float(np.sum((resid / noise) ** 2) / dof) if noise > 0 and dof > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    dynamic_range = float(np.max(y) - np.min(y))
    nrmse = rmse / dynamic_range if dynamic_range > 0 else np.nan

    delta = float(mu_h - mu_d)
    mass_ratio = float(mass_ratio_from_shift(delta, mu_h))
    area_d = float(np.sqrt(2.0 * np.pi) * a_d * sigma)
    area_h = float(np.sqrt(2.0 * np.pi) * a_h * sigma)
    c_d_percent = float(100.0 * area_d / (area_d + area_h))
    c_h_percent = float(100.0 * area_h / (area_d + area_h))

    unc = mc_uncertainties_from_covariance(popt, pcov)

    accepted = (
        (sigma <= MAX_ACCEPTED_SIGMA_NM)
        and (MIN_ACCEPTED_SHIFT_NM <= delta <= MAX_ACCEPTED_SHIFT_NM)
        and (mu_h > mu_d)
    )
    if sigma > MAX_ACCEPTED_SIGMA_NM:
        quality_note = "excluded: broad/merged peaks; slit/instrument width too large for reliable isotope shift"
    elif not (MIN_ACCEPTED_SHIFT_NM <= delta <= MAX_ACCEPTED_SHIFT_NM):
        quality_note = "excluded: fitted isotope shift outside the expected Balmer-alpha range"
    else:
        quality_note = "accepted: resolved Balmer-alpha H/D doublet"

    result = {
        "file": path.name,
        "accepted_for_final_average": accepted,
        "quality_note": quality_note,
        "lambda_D_nm": float(mu_d),
        "lambda_D_unc_nm": unc["lambda_d_nm_unc"],
        "lambda_H_nm": float(mu_h),
        "lambda_H_unc_nm": unc["lambda_h_nm_unc"],
        "isotope_shift_nm": delta,
        "isotope_shift_unc_nm": unc["delta_nm_unc"],
        "mass_ratio_mD_over_mH": mass_ratio,
        "mass_ratio_unc": unc["mass_ratio_unc"],
        "C_D_percent": c_d_percent,
        "C_D_unc_percent": unc["c_d_percent_unc"],
        "C_H_percent": c_h_percent,
        "C_H_unc_percent": unc["c_h_percent_unc"],
        "area_D": area_d,
        "area_H": area_h,
        "amplitude_D": float(a_d),
        "amplitude_H": float(a_h),
        "common_sigma_nm": float(sigma),
        "reduced_chi2_noise_estimated": reduced_chi2,
        "rmse": rmse,
        "normalized_rmse": nrmse,
        "n_points": int(len(x)),
    }

    curve_df = pd.DataFrame({
        "wavelength_nm": x,
        "intensity_au": y,
        "fit_total_au": y_fit,
        "baseline_au": np.full_like(x, c),
        "deuterium_component_au": d_comp,
        "hydrogen_component_au": h_comp,
        "residual_au": resid,
    })
    return result, curve_df


# -----------------------------------------------------------------------------
# 4. FIGURE GENERATION
# -----------------------------------------------------------------------------

def make_scan_quality_figure(deuterium_data: dict[str, pd.DataFrame], results_df: pd.DataFrame, best_file: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for fname, df in deuterium_data.items():
        win = crop(df, *DEUTERIUM_PLOT_WINDOW_NM)
        x = win["wavelength_nm"].to_numpy()
        y = win["intensity_au"].to_numpy()
        yn = (y - np.min(y)) / (np.max(y) - np.min(y)) if np.max(y) > np.min(y) else y
        lw = 2.2 if fname == best_file else 1.2
        alpha = 1.0 if fname == best_file else 0.70
        ax.plot(x, yn, lw=lw, alpha=alpha, label=fname)

    ax.axvline(THEORY_D_ALPHA_AIR_NM, ls="--", lw=1.0, alpha=0.65, label="D$\\alpha$ reference")
    ax.axvline(THEORY_H_ALPHA_AIR_NM, ls=":", lw=1.0, alpha=0.65, label="H$\\alpha$ reference")
    ax.set_xlim(*DEUTERIUM_PLOT_WINDOW_NM)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Hydrogen--deuterium lamp near Balmer-$\\alpha$")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hydrogen_deuterium_scan_quality.png", dpi=300)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def make_best_fit_figure(best_curve: pd.DataFrame, best_result: dict) -> None:
    """
    Figure 2 for the article: the resolved Balmer-alpha doublet and fit only.

    The previous version also plotted a residual panel underneath. Per the
    article-figure request, the residual panel is removed from the figure, but
    the residual values are still saved in hydrogen_deuterium_best_fit_curve.csv
    for later checks and result writing.
    """
    # Original measured data points
    x = best_curve["wavelength_nm"].to_numpy()
    y = best_curve["intensity_au"].to_numpy()
    """
    total = best_curve["fit_total_au"].to_numpy()
    base = best_curve["baseline_au"].to_numpy()
    d_comp = best_curve["deuterium_component_au"].to_numpy() + base
    h_comp = best_curve["hydrogen_component_au"].to_numpy() + base
    """
    # Dense wavelength grid for smooth plotting of the fitted model
    x_smooth = np.linspace(np.min(x), np.max(x), 1500)

    # Get fit parameters from the best result
    c = best_curve["baseline_au"].iloc[0]
    a_d = best_result["amplitude_D"]
    mu_d = best_result["lambda_D_nm"]
    a_h = best_result["amplitude_H"]
    mu_h = best_result["lambda_H_nm"]
    sigma = best_result["common_sigma_nm"]

    # Smooth fitted curves
    total = double_gaussian_common_sigma(x_smooth, c, a_d, mu_d, a_h, mu_h, sigma)
    d_comp = deuterium_component(x_smooth, a_d, mu_d, sigma) + c
    h_comp = hydrogen_component(x_smooth, a_h, mu_h, sigma) + c



    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.plot(x_smooth, total, lw=4, color="blueviolet", label="double-Gaussian fit")
    ax.plot(x_smooth, d_comp, lw=2.3, ls="--", color="cyan", label="D component")
    ax.plot(x_smooth, h_comp, lw=2.3, ls="--", color="red", label="H component")
    ax.scatter(x, y, s=19, color="black", label="data")
    ax.axvline(best_result["lambda_D_nm"], lw=2, ls="--", alpha=0.65)
    ax.axvline(best_result["lambda_H_nm"], lw=2, ls="--", alpha=0.65)

    # Isotope-shift arrow between the fitted centers.
    y_arrow = np.max(y) * 0.63
    ax.annotate(
        "",
        xy=(best_result["lambda_D_nm"], y_arrow),
        xytext=(best_result["lambda_H_nm"], y_arrow),
        arrowprops=dict(arrowstyle="<->", lw=1.9),
    )
    ax.text(
        0.5 * (best_result["lambda_D_nm"] + best_result["lambda_H_nm"]),
        y_arrow * 1.02,
        f"$\\Delta\\lambda={best_result['isotope_shift_nm']:.4f}$ nm",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    ax.set_xlabel("Wavelength [nm]", fontsize=16)
    ax.set_ylabel("Intensity [A.U.]", fontsize=16)
    ax.set_title(f"Double Gaussian Fit Hydrogen Deuterium", fontsize=20)
    ax.legend(fontsize=13, frameon=False)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hydrogen_deuterium_best_double_gaussian_fit.png", dpi=300)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. MAIN SCRIPT
# -----------------------------------------------------------------------------

def main() -> None:
    print(f"Using data folder: {DATA_DIR}")
    hydrogen_file, deuterium_files = discover_files(DATA_DIR)

    if hydrogen_file is None:
        print("Warning: no standalone hydrogen spectrum was found.")
    else:
        print(f"Hydrogen file: {hydrogen_file.name}")

    if not deuterium_files:
        raise RuntimeError("No deuterium/diatorium files were found in DATA_DIR.")
    print("Deuterium files:")
    for p in deuterium_files:
        print(f"  - {p.name}")

    # Extract only the relevant Balmer-alpha region from the full hydrogen file.
    hydrogen_result = None
    if hydrogen_file is not None:
        try:
            hdf = read_two_column_spectrum(hydrogen_file)
            hydrogen_result = fit_hydrogen_alpha_window(hdf)
            print(
                "Standalone hydrogen Balmer-alpha diagnostic: "
                f"lambda = {hydrogen_result['center_nm']:.4f} ± {hydrogen_result['center_unc_nm']:.4f} nm"
            )
        except Exception as exc:
            print(f"Warning: could not fit standalone hydrogen Balmer-alpha window: {exc}")

    # Analyze all six deuterium-lamp files.
    all_results = []
    all_curves = {}
    deuterium_data = {}
    for p in deuterium_files:
        try:
            df = read_two_column_spectrum(p)
            deuterium_data[p.name] = df
            result, curve = analyze_deuterium_file(p)
            all_results.append(result)
            all_curves[p.name] = curve
            print(f"Analyzed {p.name}: shift = {result['isotope_shift_nm']:.5f} nm, {result['quality_note']}")
        except Exception as exc:
            print(f"Warning: failed to analyze {p.name}: {exc}")

    if not all_results:
        raise RuntimeError("None of the deuterium files could be analyzed.")

    results_df = pd.DataFrame(all_results)

    # Choose the main article scan: accepted scan with smallest normalized RMSE.
    accepted_df = results_df[results_df["accepted_for_final_average"]].copy()
    if len(accepted_df) == 0:
        warnings.warn("No scan passed the quality criteria; choosing the smallest normalized RMSE anyway.")
        best_idx = int(results_df["normalized_rmse"].idxmin())
    else:
        best_idx = int(accepted_df["normalized_rmse"].idxmin())

    best_result = results_df.loc[best_idx].to_dict()
    best_file = best_result["file"]
    best_curve = all_curves[best_file]

    # Save result tables.
    results_df.to_csv(OUTPUT_DIR / "hydrogen_deuterium_fit_results.csv", index=False)
    best_curve.to_csv(OUTPUT_DIR / "hydrogen_deuterium_best_fit_curve.csv", index=False)

    # Make figures.
    # Figure 1: scan-quality overview of all H/D lamp files.
    # Figure 2: the selected double-Gaussian Balmer-alpha fit.
    # No Figure 3 is generated here: the H/D references use the double-Gaussian
    # figure plus numerical tables/CSV outputs for the isotope shift, mass ratio,
    # and concentration. Adding another plot would mostly duplicate the CSV table.
    make_scan_quality_figure(deuterium_data, results_df, best_file)
    make_best_fit_figure(best_curve, best_result)

    # Average only accepted high-resolution scans. Use the spread across files as
    # a realistic repeatability estimate, because statistical fit errors alone can
    # be too optimistic for this experiment.
    summary_lines = []
    summary_lines.append("Hydrogen and Deuterium Balmer-alpha analysis")
    summary_lines.append("=" * 54)
    summary_lines.append(f"Data folder: {DATA_DIR}")
    summary_lines.append("")

    if hydrogen_result is not None:
        summary_lines.append("Standalone hydrogen file diagnostic, cropped to H-alpha region only:")
        summary_lines.append(
            f"  H-alpha measured center = {hydrogen_result['center_nm']:.6f} ± "
            f"{hydrogen_result['center_unc_nm']:.6f} nm"
        )
        summary_lines.append(
            "  This diagnostic is not used for the isotope shift; the residual H line "
            "inside the D lamp is the internal reference."
        )
        summary_lines.append("")

    summary_lines.append(f"Best article scan: {best_file}")
    summary_lines.append(f"  lambda_D = {best_result['lambda_D_nm']:.6f} ± {best_result['lambda_D_unc_nm']:.6f} nm")
    summary_lines.append(f"  lambda_H = {best_result['lambda_H_nm']:.6f} ± {best_result['lambda_H_unc_nm']:.6f} nm")
    summary_lines.append(f"  isotope shift Δλ = lambda_H - lambda_D = {best_result['isotope_shift_nm']:.6f} ± {best_result['isotope_shift_unc_nm']:.6f} nm")
    summary_lines.append(f"  m_D/m_H = {best_result['mass_ratio_mD_over_mH']:.6f} ± {best_result['mass_ratio_unc']:.6f}")
    summary_lines.append(f"  C_D = {best_result['C_D_percent']:.3f} ± {best_result['C_D_unc_percent']:.3f} %")
    summary_lines.append(f"  C_H = {best_result['C_H_percent']:.3f} ± {best_result['C_H_unc_percent']:.3f} %")
    summary_lines.append(f"  common Gaussian sigma = {best_result['common_sigma_nm']:.6f} nm")
    summary_lines.append(f"  normalized RMSE = {best_result['normalized_rmse']:.6f}")
    summary_lines.append("")

    summary_lines.append("Reference values used for comparison:")
    summary_lines.append(f"  H-alpha reference wavelength ≈ {THEORY_H_ALPHA_AIR_NM:.6f} nm")
    summary_lines.append(f"  D-alpha reference wavelength ≈ {THEORY_D_ALPHA_AIR_NM:.6f} nm")
    summary_lines.append(f"  expected isotope shift ≈ {THEORY_SHIFT_NM:.6f} nm")
    summary_lines.append(f"  theoretical m_D/m_H = {THEORY_M_D_OVER_M_H:.6f}")
    summary_lines.append("")

    if len(accepted_df) > 0:
        summary_lines.append("Average over accepted high-resolution scans:")
        for col, label, unit in [
            ("isotope_shift_nm", "Δλ", "nm"),
            ("mass_ratio_mD_over_mH", "m_D/m_H", ""),
            ("C_D_percent", "C_D", "%"),
            ("C_H_percent", "C_H", "%"),
        ]:
            mean = accepted_df[col].mean()
            std = accepted_df[col].std(ddof=1) if len(accepted_df) > 1 else np.nan
            if np.isfinite(std):
                summary_lines.append(f"  {label} = {mean:.6f} ± {std:.6f} {unit}".rstrip())
            else:
                summary_lines.append(f"  {label} = {mean:.6f} {unit}".rstrip())
        summary_lines.append("")

    summary_lines.append("Quality notes by file:")
    for _, row in results_df.iterrows():
        summary_lines.append(
            f"  {row['file']}: shift={row['isotope_shift_nm']:.6f} nm, "
            f"sigma={row['common_sigma_nm']:.6f} nm, "
            f"nRMSE={row['normalized_rmse']:.6f}, {row['quality_note']}"
        )

    summary_path = OUTPUT_DIR / "hydrogen_deuterium_article_numbers.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSaved outputs:")
    for fname in [
        "hydrogen_deuterium_scan_quality.png",
        "hydrogen_deuterium_best_double_gaussian_fit.png",
        "hydrogen_deuterium_fit_results.csv",
        "hydrogen_deuterium_best_fit_curve.csv",
        "hydrogen_balmer_alpha_window.csv",
        "hydrogen_deuterium_article_numbers.txt",
    ]:
        print(f"  {OUTPUT_DIR / fname}")


if __name__ == "__main__":
    main()
