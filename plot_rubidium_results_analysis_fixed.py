"""
Plot and analyze Rubidium spectroscopy data for the Rubidium Results section.

This version produces ONLY the two article figures requested:

    Figure 1: rubidium_spectrum_quantum_defect.png/pdf
        - measured visible Rb spectrum
        - wavelength-colored spectrum curve
        - fitted/identified Rb I lines
        - the lines used later in the Rydberg plot are highlighted

    Figure 2: rubidium_rydberg_formula_three_panel.png/pdf
        - three Luke--George--Tucker style Rydberg plots side by side
        - Sharp series:   2 fine-structure branches
        - Diffuse series: 2 fine-structure branches
        - Principal:      1 branch

The important correction compared with the previous script:
    The subset of n values is selected ONCE, jointly, and the same subset is
    used for both the sharp and diffuse series.  This avoids the inconsistent
    situation where the sharp plot and the diffuse plot use different n values.

Physics model:
    n* = n - delta_l
    nu_tilde = 1/lambda = T - R_Rb/(n*)^2

Plotting convention:
    For each family, delta_l is first estimated from pairwise wavenumber
    differences, which eliminates the unknown series limit T.  Then the figure
    plots nu_tilde versus 1/(n*)^2 and fits each fine-structure branch linearly:

        nu_tilde = a2 * [1/(n*)^2] + a1 .

    In the ideal Rydberg model, a1 is the branch series limit T and
    a2 should be close to -R_Rb.

Before running on your computer:
    Change DATA_DIR below to the folder containing your rubidium file.
    If you already have a wavelength calibration from Hg, set APPLY_CALIBRATION=True
    and insert CALIBRATION_A and CALIBRATION_B_NM.
"""

from __future__ import annotations

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import curve_fit, brentq
from scipy.stats import chi2

# =============================================================================
# 1) USER SETTINGS
# =============================================================================

DATA_DIR = Path(r"C:\Users\orlyk\Desktop\Jonathan k\School\lab job\lab C\Spectroscopy\Data – Results – Graphs\Rubidium")
RUBIDIUM_FILE = DATA_DIR / "rubidium"
OUTPUT_DIR = DATA_DIR

# Fallback for testing inside ChatGPT sandbox. This does not affect your computer.
if not RUBIDIUM_FILE.exists() and Path("/mnt/data/rubidium").exists():
    DATA_DIR = Path("/mnt/data")
    RUBIDIUM_FILE = DATA_DIR / "rubidium"
    OUTPUT_DIR = DATA_DIR

# Calibration convention: lambda_calibrated = a * lambda_measured + b.
APPLY_CALIBRATION = False
CALIBRATION_A = 1.0
CALIBRATION_B_NM = 0.0

R_RB_CM = 109737.3156816  # cm^{-1}

# Systematic uncertainty floor used for the Rydberg fits.
# The Gaussian center uncertainty alone is usually much smaller than the real
# uncertainty from calibration, finite resolution, and line identification.
WAVELENGTH_SYSTEMATIC_NM = 0.30

# Use measured fitted centers in the Rydberg analysis. If False, the script uses
# literature/reference wavelengths from the line list.
USE_FITTED_PEAK_CENTERS_FOR_RYDBERG = True

# Subset selection:
#   None  -> the script scans all common n-subsets and selects the best one.
#   list  -> force one shared subset, for example [7, 8, 10].
MANUAL_SHARED_SUBSET_N: list[int] | None = None

# The selected subset must contain at least this many n values because each of
# the sharp/diffuse branches is fit to a line.
SUBSET_MIN_N_VALUES = 3

# First select among statistically acceptable subsets. If none pass this cut,
# the script falls back to the best theory-agreement score and prints a warning.
SUBSET_MIN_P_VALUE_FOR_VALID = 0.05

# Theory-score weights. The score is dimensionless and lower is better.
# The score compares delta_l, T, fine-structure splitting, and slope to expected
# reference values, but only after requiring the fit to be statistically sane.
SUBSET_SCORE_WEIGHTS = {
    "delta": 1.0,
    "T": 1.0,
    "split": 1.0,
    "slope": 1.0,
}

OPTIONS = {
    "save_dpi": 300,
    "save_png": True,
    "save_pdf": True,
    "show_figures": True,

    "figure_size_spectrum": (13.0, 7.0),
    "figure_size_rydberg": (18.0, 6.8),

    "title_fontsize": 32,
    "subplot_title_fontsize": 26,
    "axis_label_fontsize": 24,
    "tick_fontsize": 17,
    "legend_fontsize": 19,

    "line_width": 5,
    "marker_size": 20,
    "grid": True,
    "grid_alpha": 0.25,

    "spectrum_title": "Rubidium Emission Spectrum",
    "spectrum_x_label": r"Wavelength $\lambda$ [nm]",
    "spectrum_y_label": "Intensity [A.U.]",
    "xlim_spectrum": (400, 790),
    # Keeping the same clipped range as before makes the weaker visible lines readable.
    # Set to None if you want the 780 nm principal line to appear at full height.
    "ylim_spectrum": (-0.15, 8.5),

    "fit_window_half_width_nm": 1.25,
    "max_allowed_fit_shift_nm_for_warning": 1.0,

    "reference_line_alpha": 0.34,
    "reference_line_width": 2,
    "line_label_fontsize": 13,
    "line_label_rotation": 90,
    "line_label_y_default": 0.97,
    "scatter_every_n": 20,
}

# =============================================================================
# 2) RUBIDIUM LINE LISTS
# =============================================================================

# General line list used in Figure 1. These are reference values used for line
# identification and as initial guesses for local Gaussian fits.
REFERENCE_LINES = [
    # Principal branch used in Fig. 2
    {"line_id": "principal_5p32_to_5s12", "lambda_ref_nm": 780.027, "series": "principal", "n": 5,
     "label": r"$5p\,^2P_{3/2}\to5s\,^2S_{1/2}$"},
    {"line_id": "principal_6p32_to_5s12", "lambda_ref_nm": 420.187, "series": "principal", "n": 6,
     "label": r"$6p\,^2P_{3/2}\to5s\,^2S_{1/2}$"},

    # Sharp series, two lower fine-structure branches
    {"line_id": "sharp_7s_to_5p12", "lambda_ref_nm": 728.007, "series": "sharp", "n": 7,
     "label": r"$7s\,^2S_{1/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "sharp_8s_to_5p12", "lambda_ref_nm": 607.077, "series": "sharp", "n": 8,
     "label": r"$8s\,^2S_{1/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "sharp_9s_to_5p12", "lambda_ref_nm": 557.893, "series": "sharp", "n": 9,
     "label": r"$9s\,^2S_{1/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "sharp_10s_to_5p12", "lambda_ref_nm": 532.255, "series": "sharp", "n": 10,
     "label": r"$10s\,^2S_{1/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "sharp_11s_to_5p12", "lambda_ref_nm": 516.994, "series": "sharp", "n": 11,
     "label": r"$11s\,^2S_{1/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "sharp_7s_to_5p32", "lambda_ref_nm": 740.828, "series": "sharp", "n": 7,
     "label": r"$7s\,^2S_{1/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "sharp_8s_to_5p32", "lambda_ref_nm": 615.964, "series": "sharp", "n": 8,
     "label": r"$8s\,^2S_{1/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "sharp_9s_to_5p32", "lambda_ref_nm": 565.385, "series": "sharp", "n": 9,
     "label": r"$9s\,^2S_{1/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "sharp_10s_to_5p32", "lambda_ref_nm": 539.068, "series": "sharp", "n": 10,
     "label": r"$10s\,^2S_{1/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "sharp_11s_to_5p32", "lambda_ref_nm": 523.412, "series": "sharp", "n": 11,
     "label": r"$11s\,^2S_{1/2}\to5p\,^2P_{3/2}$"},

    # Diffuse series, two branches following the notation used by Luke et al.
    {"line_id": "diffuse_5d32_to_5p12", "lambda_ref_nm": 761.899, "series": "diffuse", "n": 5,
     "label": r"$5d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_6d32_to_5p12", "lambda_ref_nm": 620.627, "series": "diffuse", "n": 6,
     "label": r"$6d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_7d32_to_5p12", "lambda_ref_nm": 564.791, "series": "diffuse", "n": 7,
     "label": r"$7d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_8d32_to_5p12", "lambda_ref_nm": 543.168, "series": "diffuse", "n": 8,
     "label": r"$8d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_9d32_to_5p12", "lambda_ref_nm": 526.019, "series": "diffuse", "n": 9,
     "label": r"$9d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_10d32_to_5p12", "lambda_ref_nm": 515.033, "series": "diffuse", "n": 10,
     "label": r"$10d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_11d32_to_5p12", "lambda_ref_nm": 507.547, "series": "diffuse", "n": 11,
     "label": r"$11d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_12d32_to_5p12", "lambda_ref_nm": 502.188, "series": "diffuse", "n": 12,
     "label": r"$12d\,^2D_{3/2}\to5p\,^2P_{1/2}$"},
    {"line_id": "diffuse_5d52_to_5p32", "lambda_ref_nm": 775.949, "series": "diffuse", "n": 5,
     "label": r"$5d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "diffuse_6d52_to_5p32", "lambda_ref_nm": 629.926, "series": "diffuse", "n": 6,
     "label": r"$6d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "diffuse_7d52_to_5p32", "lambda_ref_nm": 572.467, "series": "diffuse", "n": 7,
     "label": r"$7d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "diffuse_8d52_to_5p32", "lambda_ref_nm": 536.277, "series": "diffuse", "n": 8,
     "label": r"$8d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "diffuse_9d52_to_5p32", "lambda_ref_nm": 519.544, "series": "diffuse", "n": 9,
     "label": r"$9d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "diffuse_10d52_to_5p32", "lambda_ref_nm": 508.818, "series": "diffuse", "n": 10,
     "label": r"$10d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "diffuse_11d52_to_5p32", "lambda_ref_nm": 501.495, "series": "diffuse", "n": 11,
     "label": r"$11d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
    {"line_id": "diffuse_12d52_to_5p32", "lambda_ref_nm": 496.262, "series": "diffuse", "n": 12,
     "label": r"$12d\,^2D_{5/2}\to5p\,^2P_{3/2}$"},
]

SERIES_COLORS = {
    "principal": "tab:purple",
    "sharp": "tab:green",
    "diffuse": "tab:blue",
}

# Families used in Figure 2.  The legend labels are intentionally full
# spectroscopic branch names instead of short ambiguous labels such as nd -> 5p.
RYDBERG_FAMILIES = {
    "sharp": {
        "subplot_title_prefix": "Sharp series",
        "delta_symbol": r"\delta_s",
        "delta_theory": 3.139,
        "branches": [
            {
                "branch_key": "sharp_to_5p12",
                "legend_label": r"$ns\,^2S_{1/2}\rightarrow5p\,^2P_{1/2}$",
                "marker": "o",
                "color": "tab:red",
                "T_theory_cm_inv": 21112.06,
                "line_ids": [
                    "sharp_7s_to_5p12", "sharp_8s_to_5p12", "sharp_9s_to_5p12",
                    "sharp_10s_to_5p12", "sharp_11s_to_5p12",
                ],
            },
            {
                "branch_key": "sharp_to_5p32",
                "legend_label": r"$ns\,^2S_{1/2}\rightarrow5p\,^2P_{3/2}$",
                "marker": "^",
                "color": "tab:green",
                "T_theory_cm_inv": 20874.46,
                "line_ids": [
                    "sharp_7s_to_5p32", "sharp_8s_to_5p32", "sharp_9s_to_5p32",
                    "sharp_10s_to_5p32", "sharp_11s_to_5p32",
                ],
            },
        ],
    },
    "diffuse": {
        "subplot_title_prefix": "Diffuse series",
        "delta_symbol": r"\delta_d",
        "delta_theory": 1.339,
        "branches": [
            {
                "branch_key": "diffuse_to_5p12",
                "legend_label": r"$nd\,^2D_{3/2}\rightarrow5p\,^2P_{1/2}$",
                "marker": "^",
                "color": "tab:blue",
                "T_theory_cm_inv": 20874.46,
                "line_ids": [
                    "diffuse_5d32_to_5p12", "diffuse_6d32_to_5p12", "diffuse_7d32_to_5p12",
                    "diffuse_8d32_to_5p12", "diffuse_9d32_to_5p12", "diffuse_10d32_to_5p12",
                    "diffuse_11d32_to_5p12", "diffuse_12d32_to_5p12",
                ],
            },
            {
                "branch_key": "diffuse_to_5p32",
                "legend_label": r"$nd\,^2D_{5/2}\rightarrow5p\,^2P_{3/2}$",
                "marker": "o",
                "color": "tab:orange",
                "T_theory_cm_inv": 21112.06,
                "line_ids": [
                    "diffuse_5d52_to_5p32", "diffuse_6d52_to_5p32", "diffuse_7d52_to_5p32",
                    "diffuse_8d52_to_5p32", "diffuse_9d52_to_5p32", "diffuse_10d52_to_5p32",
                    "diffuse_11d52_to_5p32", "diffuse_12d52_to_5p32",
                ],
            },
        ],
    },
    "principal": {
        "subplot_title_prefix": "Principal series",
        "delta_symbol": r"\delta_p",
        "delta_theory": 2.680,
        "branches": [
            {
                "branch_key": "principal_to_5s12",
                "legend_label": r"$np\,^2P_{3/2}\rightarrow5s\,^2S_{1/2}$",
                "marker": "o",
                "color": "tab:purple",
                "T_theory_cm_inv": 33691.02,
                "line_ids": ["principal_5p32_to_5s12", "principal_6p32_to_5s12"],
            }
        ],
    },
}

# =============================================================================
# 3) BASIC HELPERS
# =============================================================================

def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column spectroscopy file: wavelength_nm, intensity."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+|,|;",
        engine="python",
        comment="#",
        header=None,
        names=["wavelength_nm", "intensity"],
    )
    df = df.dropna().apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 2:
        raise ValueError(f"File has fewer than 2 valid rows: {path}")
    x = df["wavelength_nm"].to_numpy(dtype=float)
    y = df["intensity"].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if APPLY_CALIBRATION:
        x = CALIBRATION_A * x + CALIBRATION_B_NM
    return x, y


def wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> tuple[float, float, float]:
    """Approximate visible wavelength to RGB for spectrum plotting."""
    w = float(wavelength_nm)
    if 380 <= w < 440:
        attenuation = 0.3 + 0.7 * (w - 380) / (440 - 380)
        r = ((-(w - 440) / (440 - 380)) * attenuation) ** gamma
        g = 0.0
        b = (1.0 * attenuation) ** gamma
    elif 440 <= w < 490:
        r = 0.0
        g = ((w - 440) / (490 - 440)) ** gamma
        b = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = (-(w - 510) / (510 - 490)) ** gamma
    elif 510 <= w < 580:
        r = ((w - 510) / (580 - 510)) ** gamma
        g = 1.0
        b = 0.0
    elif 580 <= w < 645:
        r = 1.0
        g = (-(w - 645) / (645 - 580)) ** gamma
        b = 0.0
    elif 645 <= w <= 790:
        attenuation = 0.3 + 0.7 * (790 - w) / (790 - 645)
        r = (1.0 * attenuation) ** gamma
        g = 0.0
        b = 0.0
    else:
        r = g = b = 1.0
    return float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1))


def wavelength_colors(x_nm: np.ndarray) -> np.ndarray:
    return np.array([wavelength_to_rgb(v) for v in x_nm])


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    if OPTIONS["save_png"]:
        fig.savefig(output_base.with_suffix(".png"), dpi=OPTIONS["save_dpi"], bbox_inches="tight")
    if OPTIONS["save_pdf"]:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def gaussian_linear_bg(x, b, m, A, mu, sigma):
    """Gaussian peak plus local linear background."""
    return b + m * (x - mu) + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_single_peak(x: np.ndarray, y: np.ndarray, lambda_ref_nm: float, window_half_width_nm: float) -> dict:
    """Fit one local spectral peak near a reference wavelength."""
    mask = (x >= lambda_ref_nm - window_half_width_nm) & (x <= lambda_ref_nm + window_half_width_nm)
    xx = x[mask]
    yy = y[mask]
    if len(xx) < 10:
        raise ValueError(f"Not enough points near {lambda_ref_nm:.3f} nm")

    mu0 = float(xx[np.argmax(yy)])
    baseline0 = float(np.percentile(yy, 10))
    amp0 = float(max(np.max(yy) - baseline0, 1e-12))
    p0 = [baseline0, 0.0, amp0, mu0, 0.12]
    lower = [-np.inf, -np.inf, 0.0, lambda_ref_nm - window_half_width_nm, 0.003]
    upper = [np.inf, np.inf, np.inf, lambda_ref_nm + window_half_width_nm, 1.2]

    popt, pcov = curve_fit(
        gaussian_linear_bg, xx, yy, p0=p0, bounds=(lower, upper), maxfev=50000
    )
    perr = np.sqrt(np.diag(pcov))
    b, m, A, mu, sigma = popt
    db, dm, dA, dmu, dsigma = perr
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    dfwhm = 2 * np.sqrt(2 * np.log(2)) * dsigma
    residuals = yy - gaussian_linear_bg(xx, *popt)
    dof = max(len(xx) - len(popt), 1)

    return {
        "lambda_fit_nm": float(mu),
        "d_lambda_fit_nm": float(dmu),
        "fit_shift_nm": float(mu - lambda_ref_nm),
        "amplitude": float(A),
        "d_amplitude": float(dA),
        "sigma_nm": float(sigma),
        "d_sigma_nm": float(dsigma),
        "fwhm_nm": float(fwhm),
        "d_fwhm_nm": float(dfwhm),
        "reduced_residual_variance": float(np.sum(residuals**2) / dof),
        "fit_parameters": popt,
    }


def nm_to_wavenumber_cm(lambda_nm: np.ndarray | float) -> np.ndarray | float:
    return 1.0e7 / lambda_nm


def d_wavenumber_from_d_lambda(lambda_nm: np.ndarray | float, d_lambda_nm: np.ndarray | float) -> np.ndarray | float:
    return 1.0e7 * d_lambda_nm / (np.asarray(lambda_nm) ** 2)


def local_intensity_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    if x0 < np.nanmin(x) or x0 > np.nanmax(x):
        return np.nan
    return float(np.interp(x0, x, y))

# =============================================================================
# 4) PEAK FITTING AND RYDBERG FITTING
# =============================================================================

def fit_all_reference_lines(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Fit all reference Rb lines in the visible range."""
    rows = []
    for line in REFERENCE_LINES:
        row = dict(line)
        try:
            fit = fit_single_peak(
                x, y,
                lambda_ref_nm=float(line["lambda_ref_nm"]),
                window_half_width_nm=OPTIONS["fit_window_half_width_nm"],
            )
            row.update({k: v for k, v in fit.items() if k != "fit_parameters"})
            row["fit_success"] = True
            row["fit_message"] = "ok"
            row["fit_warning"] = abs(row["fit_shift_nm"]) > OPTIONS["max_allowed_fit_shift_nm_for_warning"]
        except Exception as exc:
            row.update({
                "lambda_fit_nm": np.nan,
                "d_lambda_fit_nm": np.nan,
                "fit_shift_nm": np.nan,
                "amplitude": np.nan,
                "fwhm_nm": np.nan,
                "fit_success": False,
                "fit_message": str(exc),
                "fit_warning": True,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def make_family_input_table(peak_df: pd.DataFrame, family_key: str) -> pd.DataFrame:
    """Build the line table used by one Figure-2 family."""
    family = RYDBERG_FAMILIES[family_key]
    rows = []
    for branch in family["branches"]:
        for line_id in branch["line_ids"]:
            match = peak_df[peak_df["line_id"] == line_id]
            if len(match) == 0:
                continue
            line = match.iloc[0]
            if USE_FITTED_PEAK_CENTERS_FOR_RYDBERG and bool(line.get("fit_success", False)):
                lam = float(line["lambda_fit_nm"])
                dlam_stat = float(line["d_lambda_fit_nm"])
                source = "fit"
            else:
                lam = float(line["lambda_ref_nm"])
                dlam_stat = 0.001
                source = "reference"
            if not np.isfinite(lam):
                continue
            dlam = float(np.sqrt((dlam_stat if np.isfinite(dlam_stat) else 0.0) ** 2 + WAVELENGTH_SYSTEMATIC_NM ** 2))
            nu = float(nm_to_wavenumber_cm(lam))
            dnu = float(d_wavenumber_from_d_lambda(lam, dlam))
            rows.append({
                "family_key": family_key,
                "branch_key": branch["branch_key"],
                "branch_label": branch["legend_label"],
                "marker": branch["marker"],
                "color": branch["color"],
                "T_theory_cm_inv": branch.get("T_theory_cm_inv", np.nan),
                "delta_theory": family.get("delta_theory", np.nan),
                "line_id": line_id,
                "n": int(line["n"]),
                "lambda_ref_nm": float(line["lambda_ref_nm"]),
                "lambda_nm": lam,
                "d_lambda_nm": dlam,
                "wavenumber_cm_inv": nu,
                "d_wavenumber_cm_inv": dnu,
                "data_source": source,
            })
    return pd.DataFrame(rows).sort_values(["branch_key", "n"]).reset_index(drop=True)


def solve_delta_from_pair(n_low: float, n_high: float, nu_low_cm: float, nu_high_cm: float) -> float:
    """Solve delta from two lines after eliminating the series limit."""
    delta_nu = float(nu_high_cm - nu_low_cm)
    if delta_nu <= 0:
        raise ValueError("Expected wavenumber to increase with n inside a regular Rydberg series.")

    def f(delta):
        return R_RB_CM * (
            1.0 / (n_low - delta) ** 2 - 1.0 / (n_high - delta) ** 2
        ) - delta_nu

    upper = min(n_low, n_high) - 1e-4
    return float(brentq(f, 0.0, upper, maxiter=10000))


def estimate_delta_pairwise(df: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Estimate one delta_l for a family using all branch-internal line pairs."""
    pair_rows = []
    for branch_key, branch_rows in df.groupby("branch_key"):
        branch_rows = branch_rows.sort_values("n").reset_index(drop=True)
        for i, j in itertools.combinations(range(len(branch_rows)), 2):
            low = branch_rows.iloc[i]
            high = branch_rows.iloc[j]
            try:
                delta_pair = solve_delta_from_pair(
                    n_low=float(low["n"]), n_high=float(high["n"]),
                    nu_low_cm=float(low["wavenumber_cm_inv"]),
                    nu_high_cm=float(high["wavenumber_cm_inv"]),
                )
                ok = True
                msg = "ok"
            except Exception as exc:
                delta_pair = np.nan
                ok = False
                msg = str(exc)
            pair_rows.append({
                "family_key": str(low["family_key"]),
                "branch_key": branch_key,
                "n_low": int(low["n"]),
                "n_high": int(high["n"]),
                "delta_from_pair": delta_pair,
                "success": ok,
                "message": msg,
            })
    pair_df = pd.DataFrame(pair_rows)
    good = pair_df[(pair_df["success"] == True) & np.isfinite(pair_df["delta_from_pair"])]
    if len(good) == 0:
        raise ValueError("No valid pairwise delta estimates were found.")
    delta = float(good["delta_from_pair"].mean())
    d_delta = float(good["delta_from_pair"].std(ddof=1) / np.sqrt(len(good))) if len(good) > 1 else np.nan
    return delta, d_delta, pair_df


def weighted_linear_fit(x_vals: np.ndarray, y_vals: np.ndarray, y_err: np.ndarray) -> dict:
    """Fit y = a2*x + a1 with y uncertainties."""
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    y_err = np.asarray(y_err, dtype=float)
    if not np.any(np.isfinite(y_err) & (y_err > 0)):
        y_err = np.ones_like(y_vals)
    else:
        floor = np.nanmedian(y_err[np.isfinite(y_err) & (y_err > 0)])
        y_err = np.where(np.isfinite(y_err) & (y_err > 0), y_err, floor)

    def model(x, a2, a1):
        return a2 * x + a1

    if len(x_vals) >= 2:
        a2_0, a1_0 = np.polyfit(x_vals, y_vals, deg=1)
    else:
        raise ValueError("Need at least two points for a linear branch fit.")

    # With exactly two points, curve_fit covariance is formally underdetermined.
    # The fit is still useful for drawing the principal-series line.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(
            model, x_vals, y_vals, p0=[a2_0, a1_0], sigma=y_err,
            absolute_sigma=True, maxfev=100000,
        )
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.array([np.nan, np.nan])
    a2, a1 = popt
    da2, da1 = perr
    residuals = y_vals - model(x_vals, a2, a1)
    dof = len(x_vals) - 2
    if dof > 0:
        chi2_val = float(np.sum((residuals / y_err) ** 2))
        chi2_red = chi2_val / dof
        p_value = float(chi2.sf(chi2_val, dof))
    else:
        chi2_val = np.nan
        chi2_red = np.nan
        p_value = np.nan
    return {
        "a1_intercept_cm_inv": float(a1),
        "d_a1_intercept_cm_inv": float(da1),
        "a2_slope_cm_inv": float(a2),
        "d_a2_slope_cm_inv": float(da2),
        "R_from_slope_cm_inv": float(-a2),
        "d_R_from_slope_cm_inv": float(da2),
        "residuals": residuals,
        "chi2": chi2_val,
        "dof": int(max(dof, 0)),
        "chi2_red": float(chi2_red) if np.isfinite(chi2_red) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
    }


def fit_family_rydberg(input_df: pd.DataFrame, family_key: str, subset_n: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate delta, compute x=1/(n*)^2, and fit each branch linearly."""
    family = RYDBERG_FAMILIES[family_key]
    df = input_df.copy()
    if subset_n is not None:
        df = df[df["n"].astype(int).isin([int(n) for n in subset_n])].copy()
    if len(df) == 0:
        raise ValueError(f"No input rows left for {family_key} after subset selection.")

    # Require enough lines in every branch.
    for branch in family["branches"]:
        branch_rows = df[df["branch_key"] == branch["branch_key"]]
        if len(branch_rows) < 2:
            raise ValueError(f"Branch {branch['branch_key']} has fewer than two lines.")

    delta, d_delta, pair_df = estimate_delta_pairwise(df)
    df["delta_l"] = delta
    df["d_delta_l"] = d_delta
    df["n_star"] = df["n"].astype(float) - delta
    df["inv_nstar2"] = 1.0 / df["n_star"] ** 2
    if np.isfinite(d_delta) and d_delta > 0:
        df["d_inv_nstar2"] = 2.0 * d_delta / np.abs(df["n_star"]) ** 3
    else:
        df["d_inv_nstar2"] = np.nan

    fit_rows = []
    total_chi2 = 0.0
    total_dof = 0
    for branch in family["branches"]:
        branch_key = branch["branch_key"]
        branch_rows = df[df["branch_key"] == branch_key].sort_values("inv_nstar2").copy()
        fit = weighted_linear_fit(
            branch_rows["inv_nstar2"].to_numpy(dtype=float),
            branch_rows["wavenumber_cm_inv"].to_numpy(dtype=float),
            branch_rows["d_wavenumber_cm_inv"].to_numpy(dtype=float),
        )
        if np.isfinite(fit["chi2"]):
            total_chi2 += float(fit["chi2"])
            total_dof += int(fit["dof"])
        fit_rows.append({
            "family_key": family_key,
            "branch_key": branch_key,
            "branch_label": branch["legend_label"],
            "color": branch["color"],
            "marker": branch["marker"],
            "delta_symbol": family["delta_symbol"],
            "delta_l": delta,
            "d_delta_l": d_delta,
            "delta_theory": family.get("delta_theory", np.nan),
            "T_cm_inv": fit["a1_intercept_cm_inv"],
            "d_T_cm_inv": fit["d_a1_intercept_cm_inv"],
            "T_theory_cm_inv": branch.get("T_theory_cm_inv", np.nan),
            "a2_slope_cm_inv": fit["a2_slope_cm_inv"],
            "d_a2_slope_cm_inv": fit["d_a2_slope_cm_inv"],
            "R_from_slope_cm_inv": fit["R_from_slope_cm_inv"],
            "num_lines_branch": int(len(branch_rows)),
            "branch_chi2_red": fit["chi2_red"],
            "branch_p_value": fit["p_value"],
        })

    fit_df = pd.DataFrame(fit_rows)
    if total_dof > 0:
        fit_df["chi2"] = float(total_chi2)
        fit_df["dof"] = int(total_dof)
        fit_df["chi2_red"] = float(total_chi2 / total_dof)
        fit_df["p_value"] = float(chi2.sf(total_chi2, total_dof))
    else:
        fit_df["chi2"] = np.nan
        fit_df["dof"] = 0
        fit_df["chi2_red"] = np.nan
        fit_df["p_value"] = np.nan

    if len(fit_df) >= 2:
        fit_df["series_limit_splitting_cm_inv"] = float(fit_df["T_cm_inv"].max() - fit_df["T_cm_inv"].min())
    else:
        fit_df["series_limit_splitting_cm_inv"] = np.nan
    fit_df["R_mean_from_slopes_cm_inv"] = float(fit_df["R_from_slope_cm_inv"].mean())
    fit_df["selected_subset_n"] = ",".join(str(int(n)) for n in sorted(df["n"].unique()))
    return df.reset_index(drop=True), fit_df, pair_df


def theory_metrics(fit_df: pd.DataFrame, family_key: str) -> dict:
    """Compare fitted values to the reference values used for subset selection."""
    family = RYDBERG_FAMILIES[family_key]
    delta = float(fit_df.iloc[0]["delta_l"])
    delta_theory = float(family.get("delta_theory", np.nan))
    delta_relative_error = abs(delta - delta_theory) / abs(delta_theory) if np.isfinite(delta_theory) and delta_theory != 0 else 0.0

    T_errors = []
    T_theory_values = []
    for branch in family["branches"]:
        row = fit_df[fit_df["branch_key"] == branch["branch_key"]]
        if len(row) == 0:
            continue
        T_theory = float(branch.get("T_theory_cm_inv", np.nan))
        if np.isfinite(T_theory):
            T_errors.append(float(row.iloc[0]["T_cm_inv"]) - T_theory)
            T_theory_values.append(T_theory)
    if len(T_errors) > 0:
        T_rmse = float(np.sqrt(np.mean(np.asarray(T_errors) ** 2)))
        T_relative_rmse = T_rmse / float(np.mean(np.abs(T_theory_values)))
    else:
        T_rmse = np.nan
        T_relative_rmse = 0.0

    if len(T_theory_values) >= 2:
        split_theory = float(np.max(T_theory_values) - np.min(T_theory_values))
        split_fit = float(fit_df.iloc[0].get("series_limit_splitting_cm_inv", np.nan))
        split_relative_error = abs(split_fit - split_theory) / abs(split_theory) if split_theory != 0 else 0.0
    else:
        split_relative_error = 0.0

    slopes = fit_df["a2_slope_cm_inv"].to_numpy(dtype=float)
    slope_rmse = float(np.sqrt(np.nanmean((slopes + R_RB_CM) ** 2)))
    slope_relative_rmse = slope_rmse / R_RB_CM

    score = (
        SUBSET_SCORE_WEIGHTS["delta"] * delta_relative_error
        + SUBSET_SCORE_WEIGHTS["T"] * T_relative_rmse
        + SUBSET_SCORE_WEIGHTS["split"] * split_relative_error
        + SUBSET_SCORE_WEIGHTS["slope"] * slope_relative_rmse
    )
    return {
        "delta_relative_error": float(delta_relative_error),
        "T_rmse_cm_inv": float(T_rmse),
        "T_relative_rmse": float(T_relative_rmse),
        "split_relative_error": float(split_relative_error),
        "slope_rmse_cm_inv": float(slope_rmse),
        "slope_relative_rmse": float(slope_relative_rmse),
        "theory_score": float(score),
    }


def available_n_for_family(input_df: pd.DataFrame, family_key: str) -> set[int]:
    """Return n values present in every branch of the family."""
    family = RYDBERG_FAMILIES[family_key]
    branch_sets = []
    for branch in family["branches"]:
        branch_rows = input_df[input_df["branch_key"] == branch["branch_key"]]
        branch_sets.append(set(int(n) for n in branch_rows["n"].dropna().unique()))
    if not branch_sets:
        return set()
    return set.intersection(*branch_sets)


def scan_shared_subsets(sharp_input: pd.DataFrame, diffuse_input: pd.DataFrame) -> pd.DataFrame:
    """Scan one common n-subset for both sharp and diffuse families."""
    common_n = sorted(available_n_for_family(sharp_input, "sharp") & available_n_for_family(diffuse_input, "diffuse"))
    if len(common_n) < SUBSET_MIN_N_VALUES:
        raise ValueError(f"Not enough common n values for sharp and diffuse: {common_n}")

    rows = []
    for subset_size in range(SUBSET_MIN_N_VALUES, len(common_n) + 1):
        for subset_tuple in itertools.combinations(common_n, subset_size):
            subset = list(subset_tuple)
            try:
                _, sharp_fit, _ = fit_family_rydberg(sharp_input, "sharp", subset)
                _, diffuse_fit, _ = fit_family_rydberg(diffuse_input, "diffuse", subset)
                sharp_metrics = theory_metrics(sharp_fit, "sharp")
                diffuse_metrics = theory_metrics(diffuse_fit, "diffuse")
                joint_score = sharp_metrics["theory_score"] + diffuse_metrics["theory_score"]
                min_p = min(float(sharp_fit.iloc[0]["p_value"]), float(diffuse_fit.iloc[0]["p_value"]))
                rows.append({
                    "subset_n": ",".join(str(n) for n in subset),
                    "subset_size": len(subset),
                    "joint_score": joint_score,
                    "min_p_value": min_p,
                    "sharp_p_value": float(sharp_fit.iloc[0]["p_value"]),
                    "diffuse_p_value": float(diffuse_fit.iloc[0]["p_value"]),
                    "sharp_delta_l": float(sharp_fit.iloc[0]["delta_l"]),
                    "diffuse_delta_l": float(diffuse_fit.iloc[0]["delta_l"]),
                    "sharp_theory_score": sharp_metrics["theory_score"],
                    "diffuse_theory_score": diffuse_metrics["theory_score"],
                    "sharp_T_rmse_cm_inv": sharp_metrics["T_rmse_cm_inv"],
                    "diffuse_T_rmse_cm_inv": diffuse_metrics["T_rmse_cm_inv"],
                    "sharp_slope_rmse_cm_inv": sharp_metrics["slope_rmse_cm_inv"],
                    "diffuse_slope_rmse_cm_inv": diffuse_metrics["slope_rmse_cm_inv"],
                    "fit_success": True,
                    "fit_message": "ok",
                })
            except Exception as exc:
                rows.append({
                    "subset_n": ",".join(str(n) for n in subset),
                    "subset_size": len(subset),
                    "joint_score": np.inf,
                    "min_p_value": 0.0,
                    "fit_success": False,
                    "fit_message": str(exc),
                })
    scan_df = pd.DataFrame(rows)
    return scan_df.sort_values(["joint_score", "min_p_value"], ascending=[True, False]).reset_index(drop=True)


def select_shared_subset(scan_df: pd.DataFrame) -> tuple[list[int], pd.Series, str]:
    """Choose one shared subset for sharp and diffuse."""
    good = scan_df[scan_df["fit_success"] == True].copy()
    if len(good) == 0:
        raise ValueError("No successful shared-subset fits were found.")
    valid = good[good["min_p_value"] >= SUBSET_MIN_P_VALUE_FOR_VALID].copy()
    if len(valid) > 0:
        row = valid.sort_values(["joint_score", "min_p_value"], ascending=[True, False]).iloc[0]
        reason = "best joint theory score among statistically acceptable shared subsets"
    else:
        row = good.sort_values(["joint_score", "min_p_value"], ascending=[True, False]).iloc[0]
        reason = "best joint theory score; warning: no shared subset passed the p-value threshold"
    subset = [int(v) for v in str(row["subset_n"]).split(",")]
    return subset, row, reason

# =============================================================================
# 5) PLOTTING
# =============================================================================

def plot_spectrum(x: np.ndarray, y: np.ndarray, peak_df: pd.DataFrame, selected_lines_df: pd.DataFrame) -> None:
    """Create Figure 1: measured spectrum and identified Rb lines."""
    fig, ax = plt.subplots(figsize=OPTIONS["figure_size_spectrum"])

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        colors=wavelength_colors(0.5 * (x[:-1] + x[1:])),
        linewidths=OPTIONS["line_width"],
        alpha=1.0,
        label="Measured Rb spectrum",
    )
    ax.add_collection(lc)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale_view()

    step = max(int(OPTIONS["scatter_every_n"]), 1)
    ax.scatter(x[::step], y[::step], s=10, c=wavelength_colors(x[::step]), edgecolors="none", alpha=0.9)

    label_transform = blended_transform_factory(ax.transData, ax.transAxes)
    used_series_labels = set()
    for _, line in peak_df.iterrows():
        lam_ref = float(line["lambda_ref_nm"])
        if not (np.nanmin(x) <= lam_ref <= np.nanmax(x)):
            continue
        series = str(line["series"])
        color = SERIES_COLORS.get(series, "gray")
        legend_label = f"{series.capitalize()} reference lines"
        if legend_label in used_series_labels:
            legend_label = None
        else:
            used_series_labels.add(legend_label)
        ax.axvline(lam_ref, color=color, lw=OPTIONS["reference_line_width"], alpha=OPTIONS["reference_line_alpha"], label=legend_label)


    ok = peak_df[peak_df["fit_success"] == True].copy()
    ax.scatter(
        ok["lambda_fit_nm"],
        [local_intensity_at(x, y, v) for v in ok["lambda_fit_nm"]],
        s=40,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Fitted peak centers",
        zorder=5,
    )

    # Highlight the exact lines used in Figure 2.
    used = set()
    for (family_key, branch_key), rows in selected_lines_df.groupby(["family_key", "branch_key"]):
        rows = rows.copy()
        branch_label = str(rows.iloc[0]["branch_label"])
        marker = str(rows.iloc[0]["marker"])
        color = str(rows.iloc[0]["color"])
        label = f"{branch_label}"
        if label in used:
            label = None
        else:
            used.add(label)
        ax.scatter(
            rows["lambda_nm"],
            [local_intensity_at(x, y, v) for v in rows["lambda_nm"]],
            marker=marker,
            s=115,
            facecolors=color,
            edgecolors=color,
            linewidths=2.1,
            label=label,
            zorder=8,
        )

    title = OPTIONS["spectrum_title"]
    if APPLY_CALIBRATION:
        title += " (calibrated wavelength axis)"
    ax.set_title(title, fontsize=OPTIONS["title_fontsize"])
    ax.set_xlabel(OPTIONS["spectrum_x_label"], fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_ylabel(OPTIONS["spectrum_y_label"], fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_xlim(*OPTIONS["xlim_spectrum"])
    if OPTIONS["ylim_spectrum"] is not None:
        ax.set_ylim(*OPTIONS["ylim_spectrum"])
    ax.tick_params(axis="both", labelsize=OPTIONS["tick_fontsize"])
    if OPTIONS["grid"]:
        ax.grid(True, alpha=OPTIONS["grid_alpha"])
    ax.legend(fontsize=OPTIONS["legend_fontsize"] - 2, loc="upper left", ncol=1)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "rubidium_spectrum_quantum_defect")
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)


def plot_family_panel(ax: plt.Axes, family_key: str, input_df: pd.DataFrame, fit_df: pd.DataFrame, subplot_title: str) -> None:
    """Draw one Figure-2 panel."""
    x_max = float(input_df["inv_nstar2"].max() * 1.08)
    x_grid = np.linspace(0.0, x_max, 400)
    y_for_limits = []

    for _, fit_row in fit_df.iterrows():
        branch_key = fit_row["branch_key"]
        branch_rows = input_df[input_df["branch_key"] == branch_key].sort_values("inv_nstar2")
        color = str(fit_row["color"])
        marker = str(fit_row["marker"])
        label = str(fit_row["branch_label"])
        a1 = float(fit_row["T_cm_inv"])
        a2 = float(fit_row["a2_slope_cm_inv"])
        y_line = a2 * x_grid + a1
        ax.plot(x_grid, y_line, color=color, lw=5.5, label=label, zorder=3)
        ax.errorbar(
            branch_rows["inv_nstar2"],
            branch_rows["wavenumber_cm_inv"],
            xerr=branch_rows["d_inv_nstar2"] if np.all(np.isfinite(branch_rows["d_inv_nstar2"])) else None,
            yerr=branch_rows["d_wavenumber_cm_inv"],
            fmt=marker,
            color=color,
            markerfacecolor="black",
            markeredgecolor=color,
            markersize=13,
            capsize=1,
            linestyle="none",
            zorder=12,
        )
        for _, r in branch_rows.iterrows():
            ax.annotate(
                f"{int(r['n'])}",
                xy=(float(r["inv_nstar2"]), float(r["wavenumber_cm_inv"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=19,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.3),
            )
        y_for_limits.extend(y_line.tolist())
        y_for_limits.extend(branch_rows["wavenumber_cm_inv"].tolist())

    family = RYDBERG_FAMILIES[family_key]
    subset_text = str(fit_df.iloc[0]["selected_subset_n"])
    ax.set_title(f"{subplot_title}  (n=\u007b{subset_text}\u007d)", fontsize=OPTIONS["subplot_title_fontsize"])
    ax.set_xlabel(r"$1/(n^*)^2$", fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_ylabel(r"$\tilde{\nu}$ [$\mathrm{cm}^{-1}$]", fontsize=OPTIONS["axis_label_fontsize"])
    ax.tick_params(axis="both", labelsize=OPTIONS["tick_fontsize"])
    if OPTIONS["grid"]:
        ax.grid(True, alpha=OPTIONS["grid_alpha"])
    ax.legend(fontsize=OPTIONS["legend_fontsize"], loc="best")
    ax.set_xlim(0.0, x_max)

    if y_for_limits:
        y_arr = np.array(y_for_limits, dtype=float)
        margin = 0.06 * (np.nanmax(y_arr) - np.nanmin(y_arr)) if np.nanmax(y_arr) > np.nanmin(y_arr) else 100
        ax.set_ylim(np.nanmin(y_arr) - margin, np.nanmax(y_arr) + margin)

    # A small physics-result box; no T marker is plotted on the graph.
    p_value = float(fit_df.iloc[0]["p_value"]) if np.isfinite(fit_df.iloc[0]["p_value"]) else np.nan
    chi_red = float(fit_df.iloc[0]["chi2_red"]) if np.isfinite(fit_df.iloc[0]["chi2_red"]) else np.nan
    delta = float(fit_df.iloc[0]["delta_l"])
    d_delta = float(fit_df.iloc[0]["d_delta_l"])
    box_lines = [
        rf"${family['delta_symbol']}={delta:.4f}$" if np.isfinite(d_delta) else rf"${family['delta_symbol']}={delta:.4f}$",
    ]
    ax.text(
        0.03,
        0.04,
        "\n".join(box_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=OPTIONS["legend_fontsize"],
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.82),
    )


def plot_three_panel_rydberg(sharp_input: pd.DataFrame, sharp_fit: pd.DataFrame,
                             diffuse_input: pd.DataFrame, diffuse_fit: pd.DataFrame,
                             principal_input: pd.DataFrame, principal_fit: pd.DataFrame) -> None:
    """Create Figure 2: three side-by-side Rydberg plots with five total lines."""
    fig, axes = plt.subplots(1, 3, figsize=OPTIONS["figure_size_rydberg"])
    fig.suptitle("Rubidium Rydberg formula plots", fontsize=OPTIONS["title_fontsize"] + 3, y=0.97)

    plot_family_panel(axes[0], "sharp", sharp_input, sharp_fit, RYDBERG_FAMILIES["sharp"]["subplot_title_prefix"])
    plot_family_panel(axes[1], "diffuse", diffuse_input, diffuse_fit, RYDBERG_FAMILIES["diffuse"]["subplot_title_prefix"])
    plot_family_panel(axes[2], "principal", principal_input, principal_fit, RYDBERG_FAMILIES["principal"]["subplot_title_prefix"])

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "rubidium_rydberg_formula_three_panel")
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)

# =============================================================================
# 6) LATEX TABLE HELPERS
# =============================================================================

def _latex_cell_formatter(value):
    """Format table cells for LaTeX output without triggering pandas FutureWarning.

    Pandas warns that DataFrame.to_latex will change in a future version.  The
    recommended replacement is Styler.to_latex, so the helper below uses Styler
    by default and keeps a safe fallback for older pandas versions.
    """
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "--"
        return f"{value:.5g}"
    return value


def dataframe_to_latex_string(df: pd.DataFrame) -> str:
    """Return a LaTeX tabular string using pandas Styler when available."""
    table_df = df.copy()

    try:
        styler = table_df.style.format(_latex_cell_formatter, escape=None)

        # hide(axis="index") is the modern Styler API.  hide_index() is kept as
        # a fallback for older pandas installations.
        if hasattr(styler, "hide"):
            styler = styler.hide(axis="index")
        elif hasattr(styler, "hide_index"):
            styler = styler.hide_index()

        return styler.to_latex(hrules=True)

    except Exception:
        # Very old pandas fallback.  The warning is intentionally suppressed so
        # the script still runs cleanly even when Styler.to_latex is unavailable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return table_df.to_latex(
                index=False,
                float_format=lambda v: f"{v:.5g}",
                escape=False,
            )


def save_latex_tables(peak_df: pd.DataFrame, rydberg_input: pd.DataFrame, rydberg_fit: pd.DataFrame, shared_subset_scan: pd.DataFrame) -> None:
    peak_cols = [
        "line_id", "series", "n", "lambda_ref_nm", "lambda_fit_nm",
        "d_lambda_fit_nm", "fit_shift_nm", "fwhm_nm", "fit_success", "fit_warning",
    ]
    fit_cols = [
        "family_key", "branch_label", "selected_subset_n", "delta_l", "d_delta_l",
        "T_cm_inv", "d_T_cm_inv", "T_theory_cm_inv", "a2_slope_cm_inv",
        "R_from_slope_cm_inv", "chi2_red", "p_value",
    ]

    with open(OUTPUT_DIR / "rubidium_latex_peak_table.txt", "w", encoding="utf-8") as f:
        f.write(dataframe_to_latex_string(peak_df[peak_cols]))

    with open(OUTPUT_DIR / "rubidium_latex_rydberg_fit_table.txt", "w", encoding="utf-8") as f:
        f.write(dataframe_to_latex_string(rydberg_fit[fit_cols]))

    with open(OUTPUT_DIR / "rubidium_latex_shared_subset_scan_top.txt", "w", encoding="utf-8") as f:
        top = shared_subset_scan.head(10)
        f.write(dataframe_to_latex_string(top))

# =============================================================================
# 7) RUN
# =============================================================================

if __name__ == "__main__":
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": OPTIONS["tick_fontsize"],
        "axes.titlesize": OPTIONS["subplot_title_fontsize"],
        "axes.labelsize": OPTIONS["axis_label_fontsize"],
        "legend.fontsize": OPTIONS["legend_fontsize"],
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, y = load_spectrum(RUBIDIUM_FILE)
    peak_df = fit_all_reference_lines(x, y)
    peak_df.to_csv(OUTPUT_DIR / "rubidium_peak_fit_results.csv", index=False)

    sharp_all = make_family_input_table(peak_df, "sharp")
    diffuse_all = make_family_input_table(peak_df, "diffuse")
    principal_all = make_family_input_table(peak_df, "principal")

    if MANUAL_SHARED_SUBSET_N is None:
        shared_scan = scan_shared_subsets(sharp_all, diffuse_all)
        shared_scan.to_csv(OUTPUT_DIR / "rubidium_shared_subset_scan.csv", index=False)
        shared_subset_n, selected_scan_row, selection_reason = select_shared_subset(shared_scan)
    else:
        shared_subset_n = [int(n) for n in MANUAL_SHARED_SUBSET_N]
        shared_scan = scan_shared_subsets(sharp_all, diffuse_all)
        shared_scan.to_csv(OUTPUT_DIR / "rubidium_shared_subset_scan.csv", index=False)
        selected_scan_row = pd.Series({"subset_n": ",".join(str(n) for n in shared_subset_n)})
        selection_reason = "manual shared subset from MANUAL_SHARED_SUBSET_N"

    sharp_input, sharp_fit, sharp_pairs = fit_family_rydberg(sharp_all, "sharp", shared_subset_n)
    diffuse_input, diffuse_fit, diffuse_pairs = fit_family_rydberg(diffuse_all, "diffuse", shared_subset_n)
    principal_input, principal_fit, principal_pairs = fit_family_rydberg(principal_all, "principal", None)

    # Attach theory metrics and selection reason to output tables.
    fit_tables = []
    input_tables = []
    pair_tables = []
    for fam_key, inp, fit, pairs in [
        ("sharp", sharp_input, sharp_fit, sharp_pairs),
        ("diffuse", diffuse_input, diffuse_fit, diffuse_pairs),
        ("principal", principal_input, principal_fit, principal_pairs),
    ]:
        metrics = theory_metrics(fit, fam_key)
        for k, v in metrics.items():
            fit[k] = v
        fit["subset_selection_reason"] = selection_reason
        fit_tables.append(fit)
        input_tables.append(inp)
        pairs["family_key"] = fam_key
        pair_tables.append(pairs)

    rydberg_input = pd.concat(input_tables, ignore_index=True)
    rydberg_fit = pd.concat(fit_tables, ignore_index=True)
    pairwise_delta = pd.concat(pair_tables, ignore_index=True)

    rydberg_input.to_csv(OUTPUT_DIR / "rubidium_rydberg_input_lines.csv", index=False)
    rydberg_fit.to_csv(OUTPUT_DIR / "rubidium_rydberg_fit_results.csv", index=False)
    pairwise_delta.to_csv(OUTPUT_DIR / "rubidium_pairwise_delta_results.csv", index=False)
    save_latex_tables(peak_df, rydberg_input, rydberg_fit, shared_scan)

    # Figure 1: spectrum. Highlight exactly the lines used in Figure 2.
    plot_spectrum(x, y, peak_df, rydberg_input)

    # Figure 2: exactly three panels and five fitted lines.
    plot_three_panel_rydberg(
        sharp_input, sharp_fit,
        diffuse_input, diffuse_fit,
        principal_input, principal_fit,
    )

    print("Saved outputs in:", OUTPUT_DIR)
    print("\nSelected shared subset for sharp and diffuse:", shared_subset_n)
    print("Selection reason:", selection_reason)
    if MANUAL_SHARED_SUBSET_N is None:
        print("\nTop shared-subset candidates:")
        print(shared_scan.head(10).to_string(index=False))
    print("\nRydberg fit summary:")
    print(rydberg_fit[[
        "family_key", "branch_label", "selected_subset_n", "delta_l", "d_delta_l",
        "T_cm_inv", "T_theory_cm_inv", "a2_slope_cm_inv", "R_from_slope_cm_inv",
        "chi2_red", "p_value", "theory_score",
    ]].to_string(index=False))

    print("\nCreated figures:")
    print("  - rubidium_spectrum_quantum_defect.png/pdf")
    print("  - rubidium_rydberg_formula_three_panel.png/pdf")
