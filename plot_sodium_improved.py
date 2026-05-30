"""
Improved sodium doublet / line-broadening analysis.

Input data format:
    two columns, no header:
        wavelength_nm    intensity

Main outputs:
    figures/
        sodium_all_times_raw.png/pdf
        sodium_all_times_normalized.png/pdf
        sodium_double_gaussian_fits.png/pdf
        sodium_fwhm_vs_time.png/pdf
        sodium_broadening_factor_vs_time.png/pdf
        sodium_doublet_splitting_vs_time.png/pdf
        sodium_doppler_temperature_check.png/pdf
        sodium_line_shape_aic_comparison.png/pdf
    data/
        sodium_fit_results.csv
        sodium_line_shape_comparison.csv
        sodium_theory_checks.csv
        sodium_normalized_spectra.csv

Physics included:
    - Sodium D-line doublet fitting.
    - Gaussian FWHM extraction.
    - Optional instrumental Gaussian deconvolution.
    - Doppler-implied temperature check.
    - Gaussian / Lorentzian / Voigt line-shape comparison using AIC/BIC.

Notes:
    1) A Gaussian fit is useful for extracting widths, but it does not prove
       that the broadening is physically Doppler broadening.
    2) If the Doppler-implied temperature is far above the expected lamp
       temperature, the observed width is dominated by other effects such as
       pressure/self-absorption/instrumental broadening.
    3) If you have a wavelength calibration from Hg/Rb/etc., fill the
       WAVELENGTH_CALIBRATION section below.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from scipy.optimize import curve_fit
    from scipy.special import wofz
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# =============================================================================
# 1) USER SETTINGS
# =============================================================================

# Keep your original folder here. The script will use it if it exists.
USER_DATA_DIR = Path(
    r"C:\Users\orlyk\Desktop\Jonathan k\School\lab job\lab C\Spectroscopy\Data – Results – Graphs\experiment 3\all data"
)

# If True, the script first tries USER_DATA_DIR, then the script folder, then
# the current working directory. This makes the code easier to move between PCs.
AUTO_DETECT_DATA_DIR = True

# Sodium files. The keys are nominal collection times in seconds.
SODIUM_FILE_NAMES = {
    0: "Sodium t=0",
    30: "Sodium t=30",
    60: "Sodium t=60",
    90: "Sodium t=90",
    120: "Sodium t=120",
    150: "Sodium t=150",
    180: "Sodium t=180",
    240: "Sodium t=240",
    300: "Sodium t=300",
    600: "Sodium t=600",
}

# Optional wavelength calibration. Leave enabled=False unless you know the
# calibration coefficients from your calibration section.
# Formula used: lambda_corrected = a0 + a1*lambda_raw + a2*lambda_raw^2.
WAVELENGTH_CALIBRATION = {
    "enabled": False,
    "a0": 0.0,
    "a1": 1.0,
    "a2": 0.0,
}

OPTIONS = {
    # Output control
    "output_subfolder": "sodium_analysis_improved",
    "save_png": True,
    "save_pdf": True,
    "save_dpi": 300,
    "show_figures": True,

    # Figure style
    "figure_size_sodium": (12, 6),
    "figure_size_summary": (9, 5),
    "title_fontsize": 28,
    "axis_label_fontsize": 22,
    "tick_fontsize": 16,
    "legend_fontsize": 15,
    "line_width": 3.8,
    "marker_size": 9,
    "grid": True,
    "grid_alpha": 0.25,
    "colormap": "viridis", #viridis

    # Plot ranges
    "x_label": r"Wavelength $\lambda$ [nm]",
    "y_label": "Intensity [A.U.]",
    "xlim_sodium": (587.5, 590.0),
    "ylim_normalized": (-0.05, 1.08),

    # Fitting windows. These are intentionally slightly wider than the peaks.
    "fit_window_nm": (587.50, 590.00),
    "left_peak_bounds_nm": (588.25, 588.90),
    "right_peak_bounds_nm": (588.95, 589.60),
    "sigma_bounds_nm": (0.005, 0.50),
    "gamma_bounds_nm": (0.005, 0.50),
    "min_signal_for_fit": 0.02,

    # Broadening-factor reference.
    # Use "first" to follow B(t)=FWHM(t)/FWHM(t0).
    # Use "first_reliable" to avoid using very noisy early data as the reference.
    "broadening_reference": "first",
    "reliable_snr_threshold": 20.0,

    # Theory checks
    "sodium_atomic_mass_u": 22.98976928,
    "theoretical_D_splitting_nm": 0.5974,
    "expected_temperature_range_K": (1900.0, 2800.0),

    # Instrument correction. Put your instrumental FWHM here if you measured it.
    # Leave as None if unknown. The measured width is then only an upper bound.
    "instrument_fwhm_nm": None,

    # Line-shape comparison. Main results still use the Gaussian FWHM, but the
    # comparison helps check whether Lorentzian/Voigt descriptions are preferred.
    "compare_line_shapes": True,
}

# =============================================================================
# 2) CONSTANTS
# =============================================================================

HC_EV_NM = 1239.8419843320026
C_M_S = 299792458.0
KB_J_K = 1.380649e-23
U_KG = 1.66053906660e-27
FWHM_GAUSSIAN_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))

# =============================================================================
# 3) PATHS AND DATA LOADING
# =============================================================================

def choose_data_dir() -> Path:
    """Return the first candidate directory that contains all sodium files."""
    candidates = [USER_DATA_DIR]

    if AUTO_DETECT_DATA_DIR:
        try:
            candidates.append(Path(__file__).resolve().parent)
        except NameError:
            pass
        candidates.append(Path.cwd())
        candidates.append(Path("/mnt/data"))  # useful when testing in ChatGPT's sandbox

    for candidate in candidates:
        if candidate and candidate.exists():
            if all((candidate / name).exists() for name in SODIUM_FILE_NAMES.values()):
                return candidate

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find a directory containing all sodium files. Checked:\n" + checked
    )


def apply_wavelength_calibration(lambda_nm: np.ndarray) -> np.ndarray:
    """Apply optional quadratic wavelength calibration."""
    x = np.asarray(lambda_nm, dtype=float)
    if not WAVELENGTH_CALIBRATION["enabled"]:
        return x

    a0 = float(WAVELENGTH_CALIBRATION["a0"])
    a1 = float(WAVELENGTH_CALIBRATION["a1"])
    a2 = float(WAVELENGTH_CALIBRATION["a2"])
    return a0 + a1 * x + a2 * x**2


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column spectroscopy file: wavelength_nm, intensity."""
    df = pd.read_csv(
        path,
        sep=r"\s+|,|;",
        engine="python",
        comment="#",
        header=None,
        names=["wavelength_nm", "intensity"],
    )
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    if len(df) < 5:
        raise ValueError(f"Too few valid rows in file: {path}")

    x_raw = df["wavelength_nm"].to_numpy(dtype=float)
    y = df["intensity"].to_numpy(dtype=float)
    x = apply_wavelength_calibration(x_raw)

    order = np.argsort(x)
    return x[order], y[order]


def load_all_spectra(data_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load all sodium spectra into a dictionary indexed by collection time."""
    spectra = {}
    for time_s, name in sorted(SODIUM_FILE_NAMES.items()):
        spectra[time_s] = load_spectrum(data_dir / name)
    return spectra


def normalize_intensity(y: np.ndarray) -> np.ndarray:
    """Subtract the minimum and normalize to the maximum."""
    y = np.asarray(y, dtype=float)
    y0 = y - np.nanmin(y)
    denom = np.nanmax(y0)
    if denom <= 0:
        return y0
    return y0 / denom


def estimate_noise(y: np.ndarray) -> float:
    """Robust noise estimate from the low-signal tails."""
    y = np.asarray(y, dtype=float)
    q20 = np.percentile(y, 20)
    tail = y[y <= q20]
    if len(tail) < 5:
        tail = y
    mad = np.median(np.abs(tail - np.median(tail)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= 0:
        noise = np.std(tail)
    if not np.isfinite(noise) or noise <= 0:
        # Quantized data can have perfectly flat tails. In that case, use the
        # smallest positive intensity step as an uncertainty scale.
        unique_y = np.unique(np.round(y, 12))
        diffs = np.diff(unique_y)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            noise = float(np.median(diffs) / np.sqrt(12.0))
    if not np.isfinite(noise) or noise <= 0:
        noise = 1e-12
    return float(noise)

# =============================================================================
# 4) MODEL FUNCTIONS
# =============================================================================

def gaussian_peak(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lorentzian_peak(x: np.ndarray, mu: float, gamma: float) -> np.ndarray:
    return 1.0 / (1.0 + ((x - mu) / gamma) ** 2)


def voigt_peak(x: np.ndarray, mu: float, sigma: float, gamma: float) -> np.ndarray:
    """Peak-normalized Voigt profile."""
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2.0))
    v = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    vmax = np.nanmax(v)
    if vmax <= 0 or not np.isfinite(vmax):
        return v
    return v / vmax


def double_gaussian_linear_bg(x, b0, b1, A1, mu1, s1, A2, mu2, s2):
    x0 = np.mean(OPTIONS["fit_window_nm"])
    return (
        b0 + b1 * (x - x0)
        + A1 * gaussian_peak(x, mu1, s1)
        + A2 * gaussian_peak(x, mu2, s2)
    )


def double_lorentzian_linear_bg(x, b0, b1, A1, mu1, g1, A2, mu2, g2):
    x0 = np.mean(OPTIONS["fit_window_nm"])
    return (
        b0 + b1 * (x - x0)
        + A1 * lorentzian_peak(x, mu1, g1)
        + A2 * lorentzian_peak(x, mu2, g2)
    )


def double_voigt_linear_bg(x, b0, b1, A1, mu1, s1, g1, A2, mu2, s2, g2):
    x0 = np.mean(OPTIONS["fit_window_nm"])
    return (
        b0 + b1 * (x - x0)
        + A1 * voigt_peak(x, mu1, s1, g1)
        + A2 * voigt_peak(x, mu2, s2, g2)
    )

# =============================================================================
# 5) FITTING AND THEORY CHECKS
# =============================================================================

def initial_guesses(x: np.ndarray, y: np.ndarray) -> dict:
    """Create robust initial guesses for the doublet fit."""
    b0 = float(np.percentile(y, 5))
    y0 = y - b0

    left_min, left_max = OPTIONS["left_peak_bounds_nm"]
    right_min, right_max = OPTIONS["right_peak_bounds_nm"]
    left = (x >= left_min) & (x <= left_max)
    right = (x >= right_min) & (x <= right_max)

    if left.sum() < 3 or right.sum() < 3:
        raise ValueError("Not enough points in one of the two peak windows.")

    A1 = float(max(np.max(y0[left]), 1e-12))
    A2 = float(max(np.max(y0[right]), 1e-12))
    mu1 = float(x[left][np.argmax(y0[left])])
    mu2 = float(x[right][np.argmax(y0[right])])

    return {"b0": b0, "b1": 0.0, "A1": A1, "A2": A2, "mu1": mu1, "mu2": mu2}


def goodness_of_fit(y: np.ndarray, y_fit: np.ndarray, n_params: int) -> dict:
    """Return RSS, RMSE, R^2, AIC, and BIC."""
    residuals = np.asarray(y) - np.asarray(y_fit)
    n = len(residuals)
    rss = float(np.sum(residuals**2))
    rmse = float(np.sqrt(rss / max(n, 1)))
    tss = float(np.sum((y - np.mean(y))**2))
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan
    rss_for_log = max(rss, 1e-300)
    aic = float(n * np.log(rss_for_log / n) + 2 * n_params)
    bic = float(n * np.log(rss_for_log / n) + n_params * np.log(n))
    return {"rss": rss, "rmse": rmse, "r2": r2, "aic": aic, "bic": bic}


def fit_gaussian_main(x: np.ndarray, y: np.ndarray, time_s: float) -> dict:
    """Main fit used for widths: two Gaussians plus linear background."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for fitting. Install with: pip install scipy")

    fit_min, fit_max = OPTIONS["fit_window_nm"]
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit_data = y[mask]

    signal = float(np.nanmax(y_fit_data) - np.nanmin(y_fit_data))
    noise = estimate_noise(y_fit_data)
    snr = signal / noise if noise > 0 else np.inf

    if signal < OPTIONS["min_signal_for_fit"]:
        return {
            "time_s": time_s,
            "fit_success": False,
            "fit_message": "signal below threshold",
            "signal": signal,
            "noise_est": noise,
            "snr_est": snr,
        }

    guess = initial_guesses(x_fit, y_fit_data)
    left_min, left_max = OPTIONS["left_peak_bounds_nm"]
    right_min, right_max = OPTIONS["right_peak_bounds_nm"]
    sigma_min, sigma_max = OPTIONS["sigma_bounds_nm"]
    yrange = float(np.ptp(y_fit_data))

    p0 = [
        guess["b0"], guess["b1"],
        guess["A1"], guess["mu1"], 0.15,
        guess["A2"], guess["mu2"], 0.15,
    ]
    lower = [
        np.min(y_fit_data) - 2 * yrange, -np.inf,
        0.0, left_min, sigma_min,
        0.0, right_min, sigma_min,
    ]
    upper = [
        np.max(y_fit_data) + 2 * yrange, np.inf,
        np.inf, left_max, sigma_max,
        np.inf, right_max, sigma_max,
    ]

    try:
        popt, pcov = curve_fit(
            double_gaussian_linear_bg,
            x_fit,
            y_fit_data,
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000,
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:
        return {
            "time_s": time_s,
            "fit_success": False,
            "fit_message": str(exc),
            "signal": signal,
            "noise_est": noise,
            "snr_est": snr,
        }

    b0, b1, A1, mu1, s1, A2, mu2, s2 = popt
    db0, db1, dA1, dmu1, ds1, dA2, dmu2, ds2 = perr
    fit_y = double_gaussian_linear_bg(x_fit, *popt)
    gof = goodness_of_fit(y_fit_data, fit_y, n_params=len(popt))

    fwhm1 = FWHM_GAUSSIAN_FACTOR * s1
    fwhm2 = FWHM_GAUSSIAN_FACTOR * s2
    dfwhm1 = FWHM_GAUSSIAN_FACTOR * ds1
    dfwhm2 = FWHM_GAUSSIAN_FACTOR * ds2
    mean_fwhm = 0.5 * (fwhm1 + fwhm2)
    dmean_fwhm = 0.5 * np.sqrt(dfwhm1**2 + dfwhm2**2)

    inst = OPTIONS["instrument_fwhm_nm"]
    if inst is not None and np.isfinite(inst):
        phys1 = np.sqrt(max(fwhm1**2 - inst**2, 0.0))
        phys2 = np.sqrt(max(fwhm2**2 - inst**2, 0.0))
        phys_mean = np.sqrt(max(mean_fwhm**2 - inst**2, 0.0))
    else:
        phys1 = np.nan
        phys2 = np.nan
        phys_mean = np.nan

    splitting_nm = mu2 - mu1
    dsplitting_nm = np.sqrt(dmu1**2 + dmu2**2)
    delta_E_SO_eV = HC_EV_NM * abs(1.0 / mu1 - 1.0 / mu2)

    # Uncertainty propagation for E = hc |1/mu1 - 1/mu2|.
    dE_dmu1 = HC_EV_NM / (mu1**2)
    dE_dmu2 = HC_EV_NM / (mu2**2)
    d_delta_E_SO_eV = np.sqrt((dE_dmu1 * dmu1) ** 2 + (dE_dmu2 * dmu2) ** 2)

    return {
        "time_s": time_s,
        "fit_success": True,
        "fit_message": "ok",
        "signal": signal,
        "noise_est": noise,
        "snr_est": snr,
        "background_b0": b0,
        "background_b1": b1,
        "A_left": A1,
        "lambda_left_nm": mu1,
        "sigma_left_nm": s1,
        "fwhm_left_nm": fwhm1,
        "fwhm_left_deconvolved_nm": phys1,
        "A_right": A2,
        "lambda_right_nm": mu2,
        "sigma_right_nm": s2,
        "fwhm_right_nm": fwhm2,
        "fwhm_right_deconvolved_nm": phys2,
        "mean_fwhm_nm": mean_fwhm,
        "mean_fwhm_deconvolved_nm": phys_mean,
        "splitting_nm": splitting_nm,
        "delta_E_SO_eV": delta_E_SO_eV,
        "d_lambda_left_nm": dmu1,
        "d_lambda_right_nm": dmu2,
        "d_fwhm_left_nm": dfwhm1,
        "d_fwhm_right_nm": dfwhm2,
        "d_mean_fwhm_nm": dmean_fwhm,
        "d_splitting_nm": dsplitting_nm,
        "d_delta_E_SO_eV": d_delta_E_SO_eV,
        **gof,
        "fit_parameters": popt,
    }


def doppler_fwhm_nm(lambda0_nm: float, temperature_K: float) -> float:
    """Doppler FWHM in wavelength units."""
    M = OPTIONS["sodium_atomic_mass_u"] * U_KG
    lambda0_m = lambda0_nm * 1e-9
    width_m = (lambda0_m / C_M_S) * np.sqrt(8.0 * KB_J_K * temperature_K * np.log(2.0) / M)
    return float(width_m * 1e9)


def doppler_temperature_from_fwhm(lambda0_nm: float, fwhm_nm: float) -> float:
    """Temperature implied if the observed FWHM were entirely Doppler broadening."""
    M = OPTIONS["sodium_atomic_mass_u"] * U_KG
    return float(
        (M * C_M_S**2 / (8.0 * KB_J_K * np.log(2.0)))
        * (fwhm_nm / lambda0_nm) ** 2
    )


def add_theory_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Doppler temperature checks and expected thermal widths."""
    df = df.copy()
    lambda_mean = 0.5 * (df["lambda_left_nm"] + df["lambda_right_nm"])

    doppler_column_map = {
        "fwhm_left_nm": "T_doppler_from_fwhm_left_K",
        "fwhm_right_nm": "T_doppler_from_fwhm_right_K",
        "mean_fwhm_nm": "T_doppler_from_fwhm_mean_K",
    }
    for col, out_col in doppler_column_map.items():
        df[out_col] = [doppler_temperature_from_fwhm(lam, w) for lam, w in zip(lambda_mean, df[col])]

    if OPTIONS["instrument_fwhm_nm"] is not None:
        col = "mean_fwhm_deconvolved_nm"
        df["T_doppler_from_deconvolved_mean_fwhm_K"] = [
            doppler_temperature_from_fwhm(lam, w) if np.isfinite(w) else np.nan
            for lam, w in zip(lambda_mean, df[col])
        ]

    T_low, T_high = OPTIONS["expected_temperature_range_K"]
    df["expected_doppler_fwhm_lowT_nm"] = [doppler_fwhm_nm(lam, T_low) for lam in lambda_mean]
    df["expected_doppler_fwhm_highT_nm"] = [doppler_fwhm_nm(lam, T_high) for lam in lambda_mean]

    return df


def fit_line_shape_model(x: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    """Fit Gaussian/Lorentzian/Voigt models for AIC/BIC comparison."""
    guess = initial_guesses(x, y)
    left_min, left_max = OPTIONS["left_peak_bounds_nm"]
    right_min, right_max = OPTIONS["right_peak_bounds_nm"]
    sigma_min, sigma_max = OPTIONS["sigma_bounds_nm"]
    gamma_min, gamma_max = OPTIONS["gamma_bounds_nm"]
    yrange = float(np.ptp(y))

    if model_name == "Gaussian":
        fun = double_gaussian_linear_bg
        p0 = [guess["b0"], 0.0, guess["A1"], guess["mu1"], 0.15, guess["A2"], guess["mu2"], 0.15]
        lower = [np.min(y) - 2 * yrange, -np.inf, 0, left_min, sigma_min, 0, right_min, sigma_min]
        upper = [np.max(y) + 2 * yrange, np.inf, np.inf, left_max, sigma_max, np.inf, right_max, sigma_max]
    elif model_name == "Lorentzian":
        fun = double_lorentzian_linear_bg
        p0 = [guess["b0"], 0.0, guess["A1"], guess["mu1"], 0.15, guess["A2"], guess["mu2"], 0.15]
        lower = [np.min(y) - 2 * yrange, -np.inf, 0, left_min, gamma_min, 0, right_min, gamma_min]
        upper = [np.max(y) + 2 * yrange, np.inf, np.inf, left_max, gamma_max, np.inf, right_max, gamma_max]
    elif model_name == "Voigt":
        fun = double_voigt_linear_bg
        p0 = [guess["b0"], 0.0, guess["A1"], guess["mu1"], 0.12, 0.04, guess["A2"], guess["mu2"], 0.12, 0.04]
        lower = [np.min(y) - 2 * yrange, -np.inf, 0, left_min, sigma_min, gamma_min, 0, right_min, sigma_min, gamma_min]
        upper = [np.max(y) + 2 * yrange, np.inf, np.inf, left_max, sigma_max, gamma_max, np.inf, right_max, sigma_max, gamma_max]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    popt, _ = curve_fit(fun, x, y, p0=p0, bounds=(lower, upper), maxfev=100000)
    y_fit = fun(x, *popt)
    gof = goodness_of_fit(y, y_fit, n_params=len(popt))
    return {"model": model_name, **gof}


def compare_line_shapes(spectra: dict[int, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Compare Gaussian, Lorentzian, and Voigt fits for each measurement."""
    if not SCIPY_AVAILABLE or not OPTIONS["compare_line_shapes"]:
        return pd.DataFrame()

    rows = []
    fit_min, fit_max = OPTIONS["fit_window_nm"]
    for time_s, (x, y) in spectra.items():
        mask = (x >= fit_min) & (x <= fit_max)
        x_fit, y_fit = x[mask], y[mask]
        for model_name in ["Gaussian", "Lorentzian", "Voigt"]:
            try:
                result = fit_line_shape_model(x_fit, y_fit, model_name)
                rows.append({"time_s": time_s, **result, "fit_success": True})
            except Exception as exc:
                rows.append({
                    "time_s": time_s,
                    "model": model_name,
                    "fit_success": False,
                    "fit_message": str(exc),
                })

    return pd.DataFrame(rows).sort_values(["time_s", "model"])

# =============================================================================
# 6) PLOTTING
# =============================================================================

def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if OPTIONS["save_png"]:
        fig.savefig(output_base.with_suffix(".png"), dpi=OPTIONS["save_dpi"], bbox_inches="tight")
    if OPTIONS["save_pdf"]:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def style_axis(ax, title: str | None = None, ylabel: str | None = None, xlim=None, ylim=None):
    ax.set_xlabel(OPTIONS["x_label"], fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_ylabel(ylabel or OPTIONS["y_label"], fontsize=OPTIONS["axis_label_fontsize"])
    if title:
        ax.set_title(title, fontsize=OPTIONS["title_fontsize"])
    ax.tick_params(axis="both", labelsize=OPTIONS["tick_fontsize"])
    if OPTIONS["grid"]:
        ax.grid(True, alpha=OPTIONS["grid_alpha"])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def get_time_cmap(times):
    cmap = plt.get_cmap(OPTIONS["colormap"])
    norm = Normalize(vmin=min(times), vmax=max(times))
    return cmap, norm


def plot_all_times(spectra, fig_dir: Path, normalize=False):
    times = sorted(spectra)
    cmap, norm = get_time_cmap(times)
    fig, ax = plt.subplots(figsize=OPTIONS["figure_size_sodium"])

    for t in times:
        x, y = spectra[t]
        y_plot = normalize_intensity(y) if normalize else y
        ax.plot(x, y_plot, color=cmap(norm(t)), lw=OPTIONS["line_width"], label=f"{t:g} s")

    title = "Normalized sodium doublet during lamp warm-up" if normalize else "Sodium Doublet During Lamp Warm-Up"
    ylabel = "Normalized intensity" if normalize else OPTIONS["y_label"]
    ylim = OPTIONS["ylim_normalized"] if normalize else None
    style_axis(ax, title=title, ylabel=ylabel, xlim=OPTIONS["xlim_sodium"], ylim=ylim)
    ax.legend(fontsize=OPTIONS["legend_fontsize"], ncol=2, loc="best")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Collection time [s]", fontsize=OPTIONS["axis_label_fontsize"])
    cbar.ax.tick_params(labelsize=OPTIONS["tick_fontsize"])

    fig.tight_layout()
    name = "sodium_all_times_normalized" if normalize else "sodium_all_times_raw"
    save_figure(fig, fig_dir / name)
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)


def plot_gaussian_fits(spectra, fit_df: pd.DataFrame, fig_dir: Path):
    times = sorted(spectra)
    cmap, norm = get_time_cmap(times)
    fig, ax = plt.subplots(figsize=OPTIONS["figure_size_sodium"])

    for t in times:
        x, y = spectra[t]
        row = fit_df[fit_df["time_s"] == t]
        color = cmap(norm(t))
        ax.scatter(x, y, s=OPTIONS["marker_size"], color=color, alpha=0.45, edgecolors="none")

        if len(row) and bool(row.iloc[0]["fit_success"]):
            params = row.iloc[0]["fit_parameters"]
            x_dense = np.linspace(OPTIONS["fit_window_nm"][0], OPTIONS["fit_window_nm"][1], 1000)
            y_dense = double_gaussian_linear_bg(x_dense, *params)
            ax.plot(x_dense, y_dense, color=color, lw=OPTIONS["line_width"], label=f"{t:g} s")

    style_axis(ax, title="Double-Gaussian Fits", xlim=OPTIONS["xlim_sodium"])
    ax.legend(fontsize=OPTIONS["legend_fontsize"], ncol=2, loc="best")
    fig.tight_layout()
    save_figure(fig, fig_dir / "sodium_double_gaussian_fits")
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)


def summary_error_plot(df, ycols, yerrcols, labels, ylabel, title, filename, fig_dir, hline=None):
    fig, ax = plt.subplots(figsize=OPTIONS["figure_size_summary"])
    for ycol, yerrcol, label in zip(ycols, yerrcols, labels):
        yerr = df[yerrcol].to_numpy() if yerrcol and yerrcol in df.columns else None
        ax.errorbar(
            df["time_s"],
            df[ycol],
            yerr=yerr,
            fmt="o-",
            lw=OPTIONS["line_width"],
            markersize=5,
            capsize=3,
            label=label,
        )
    if hline is not None:
        ax.axhline(hline, ls="--", lw=1.5, label="theory/reference")
    ax.set_xlabel("Collection time [s]", fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_title(title, fontsize=OPTIONS["title_fontsize"])
    ax.tick_params(axis="both", labelsize=OPTIONS["tick_fontsize"])
    if OPTIONS["grid"]:
        ax.grid(True, alpha=OPTIONS["grid_alpha"])
    ax.legend(fontsize=OPTIONS["legend_fontsize"])
    fig.tight_layout()
    save_figure(fig, fig_dir / filename)
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)


def plot_doppler_temperature_check(df: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=OPTIONS["figure_size_summary"])
    ax.semilogy(df["time_s"], df["T_doppler_from_fwhm_left_K"], "o-", label="left line")
    ax.semilogy(df["time_s"], df["T_doppler_from_fwhm_right_K"], "o-", label="right line")
    ax.semilogy(df["time_s"], df["T_doppler_from_fwhm_mean_K"], "o-", label="mean")
    T_low, T_high = OPTIONS["expected_temperature_range_K"]
    ax.axhspan(T_low, T_high, alpha=0.15, label=f"typical lamp range {T_low:g}--{T_high:g} K")
    ax.set_xlabel("Collection time [s]", fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_ylabel(r"Doppler-implied temperature $T_{\mathrm{Doppler}}$ [K]", fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_title("Doppler broadening consistency check", fontsize=OPTIONS["title_fontsize"])
    ax.tick_params(axis="both", labelsize=OPTIONS["tick_fontsize"])
    if OPTIONS["grid"]:
        ax.grid(True, which="both", alpha=OPTIONS["grid_alpha"])
    ax.legend(fontsize=OPTIONS["legend_fontsize"])
    fig.tight_layout()
    save_figure(fig, fig_dir / "sodium_doppler_temperature_check")
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)


def plot_line_shape_aic(comparison_df: pd.DataFrame, fig_dir: Path):
    if comparison_df.empty:
        return
    ok = comparison_df[comparison_df["fit_success"] == True].copy()
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=OPTIONS["figure_size_summary"])
    for model in ["Gaussian", "Lorentzian", "Voigt"]:
        sub = ok[ok["model"] == model]
        ax.plot(sub["time_s"], sub["aic"], "o-", label=model)
    ax.set_xlabel("Collection time [s]", fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_ylabel("AIC value", fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_title("Line-shape comparison: lower AIC is better", fontsize=OPTIONS["title_fontsize"])
    ax.tick_params(axis="both", labelsize=OPTIONS["tick_fontsize"])
    if OPTIONS["grid"]:
        ax.grid(True, alpha=OPTIONS["grid_alpha"])
    ax.legend(fontsize=OPTIONS["legend_fontsize"])
    fig.tight_layout()
    save_figure(fig, fig_dir / "sodium_line_shape_aic_comparison")
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)

# =============================================================================
# 7) OUTPUT TABLES
# =============================================================================

def save_normalized_spectra(spectra, data_dir: Path):
    rows = []
    for t, (x, y) in spectra.items():
        yn = normalize_intensity(y)
        for xi, yi, yni in zip(x, y, yn):
            rows.append({"time_s": t, "wavelength_nm": xi, "intensity": yi, "normalized_intensity": yni})
    pd.DataFrame(rows).to_csv(data_dir / "sodium_normalized_spectra.csv", index=False)


def choose_broadening_reference(df_ok: pd.DataFrame) -> tuple[float, str]:
    mode = OPTIONS["broadening_reference"]
    if mode == "first_reliable":
        reliable = df_ok[df_ok["snr_est"] >= OPTIONS["reliable_snr_threshold"]]
        if len(reliable) > 0:
            row = reliable.iloc[0]
            return float(row["mean_fwhm_nm"]), f"first reliable fit, t={row['time_s']:g} s"
    row = df_ok.iloc[0]
    return float(row["mean_fwhm_nm"]), f"first fit, t={row['time_s']:g} s"

# =============================================================================
# 8) MAIN
# =============================================================================

def main():
    plt.rcParams.update({
        "font.size": OPTIONS["tick_fontsize"],
        "axes.titlesize": OPTIONS["title_fontsize"],
        "axes.labelsize": OPTIONS["axis_label_fontsize"],
        "legend.fontsize": OPTIONS["legend_fontsize"],
        "figure.dpi": 120,
    })

    data_root = choose_data_dir()
    output_root = data_root / OPTIONS["output_subfolder"]
    fig_dir = output_root / "figures"
    table_dir = output_root / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using data directory: {data_root}")
    print(f"Saving outputs to: {output_root}")

    spectra = load_all_spectra(data_root)
    save_normalized_spectra(spectra, table_dir)

    plot_all_times(spectra, fig_dir, normalize=False)
    plot_all_times(spectra, fig_dir, normalize=True)

    if not SCIPY_AVAILABLE:
        warnings.warn("scipy is not installed, so fits were skipped. Install with: pip install scipy")
        return

    results = []
    for t, (x, y) in spectra.items():
        result = fit_gaussian_main(x, y, t)
        results.append(result)
        if not result.get("fit_success"):
            print(f"Fit failed/skipped for t={t:g} s: {result.get('fit_message')}")

    fit_df_full = pd.DataFrame(results).sort_values("time_s")
    fit_df_ok = fit_df_full[fit_df_full["fit_success"] == True].copy()

    if not fit_df_ok.empty:
        ref_width, ref_description = choose_broadening_reference(fit_df_ok)
        fit_df_ok["broadening_factor_left"] = fit_df_ok["fwhm_left_nm"] / ref_width
        fit_df_ok["broadening_factor_right"] = fit_df_ok["fwhm_right_nm"] / ref_width
        fit_df_ok["broadening_factor_mean"] = fit_df_ok["mean_fwhm_nm"] / ref_width
        fit_df_ok["broadening_reference_width_nm"] = ref_width
        fit_df_ok["broadening_reference_description"] = ref_description
        fit_df_ok = add_theory_columns(fit_df_ok)

    # Save CSV without the raw numpy fit-parameter array.
    save_df = fit_df_ok.drop(columns=["fit_parameters"], errors="ignore")
    save_df.to_csv(table_dir / "sodium_fit_results.csv", index=False)

    # Theory summary table.
    if not fit_df_ok.empty:
        lambda_typical = float(np.nanmean(0.5 * (fit_df_ok["lambda_left_nm"] + fit_df_ok["lambda_right_nm"])))
    else:
        lambda_typical = 589.0
    T_low, T_high = OPTIONS["expected_temperature_range_K"]
    theory_rows = [
        {
            "quantity": "typical_lambda_nm",
            "value": lambda_typical,
            "comment": "Mean fitted sodium wavelength used for the Doppler check.",
        },
        {
            "quantity": f"expected_Doppler_FWHM_at_{T_low:g}_K_nm",
            "value": doppler_fwhm_nm(lambda_typical, T_low),
            "comment": "Thermal Doppler width expected for sodium if T is in the usual lamp range.",
        },
        {
            "quantity": f"expected_Doppler_FWHM_at_{T_high:g}_K_nm",
            "value": doppler_fwhm_nm(lambda_typical, T_high),
            "comment": "Thermal Doppler width expected for sodium if T is in the usual lamp range.",
        },
        {
            "quantity": "theoretical_D_line_splitting_nm",
            "value": OPTIONS["theoretical_D_splitting_nm"],
            "comment": "Reference value for the Na D-line separation; calibration affects the comparison.",
        },
        {
            "quantity": "instrument_fwhm_nm",
            "value": OPTIONS["instrument_fwhm_nm"] if OPTIONS["instrument_fwhm_nm"] is not None else np.nan,
            "comment": "If unknown, measured widths are upper bounds on the physical Gaussian width.",
        },
    ]
    pd.DataFrame(theory_rows).to_csv(table_dir / "sodium_theory_checks.csv", index=False)

    comparison_df = compare_line_shapes(spectra)
    if not comparison_df.empty:
        comparison_df.to_csv(table_dir / "sodium_line_shape_comparison.csv", index=False)

    # Figures based on fits.
    if not fit_df_ok.empty:
        plot_gaussian_fits(spectra, fit_df_ok, fig_dir)

        summary_error_plot(
            fit_df_ok,
            ycols=["fwhm_left_nm", "fwhm_right_nm", "mean_fwhm_nm"],
            yerrcols=["d_fwhm_left_nm", "d_fwhm_right_nm", "d_mean_fwhm_nm"],
            labels=["left line", "right line", "mean"],
            ylabel=r"FWHM $\Delta\lambda$ [nm]",
            title="Sodium linewidth during lamp warm-up",
            filename="sodium_fwhm_vs_time",
            fig_dir=fig_dir,
        )

        summary_error_plot(
            fit_df_ok,
            ycols=["broadening_factor_left", "broadening_factor_right", "broadening_factor_mean"],
            yerrcols=[None, None, None],
            labels=["left line", "right line", "mean"],
            ylabel=r"Broadening factor $B(t)$",
            title="Relative sodium broadening during warm-up",
            filename="sodium_broadening_factor_vs_time",
            fig_dir=fig_dir,
        )

        summary_error_plot(
            fit_df_ok,
            ycols=["splitting_nm"],
            yerrcols=["d_splitting_nm"],
            labels=["measured splitting"],
            ylabel=r"Doublet splitting $\Delta\lambda$ [nm]",
            title="Sodium doublet splitting stability check",
            filename="sodium_doublet_splitting_vs_time",
            fig_dir=fig_dir,
            hline=OPTIONS["theoretical_D_splitting_nm"],
        )

        summary_error_plot(
            fit_df_ok,
            ycols=["A_left", "A_right"],
            yerrcols=[None, None],
            labels=["left line", "right line"],
            ylabel="Fitted amplitude [arb. units]",
            title="Sodium line intensity during lamp warm-up",
            filename="sodium_amplitude_vs_time",
            fig_dir=fig_dir,
        )

        plot_doppler_temperature_check(fit_df_ok, fig_dir)
        plot_line_shape_aic(comparison_df, fig_dir)

    print("\nSaved data tables:")
    for p in sorted(table_dir.glob("*.csv")):
        print("  ", p)

    print("\nSaved figures:")
    for p in sorted(fig_dir.glob("*.png")):
        print("  ", p)

    if not fit_df_ok.empty:
        compact_cols = [
            "time_s", "lambda_left_nm", "lambda_right_nm", "splitting_nm",
            "fwhm_left_nm", "fwhm_right_nm", "mean_fwhm_nm",
            "T_doppler_from_fwhm_mean_K", "r2", "snr_est",
        ]
        print("\nMain fit summary:")
        print(fit_df_ok[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
