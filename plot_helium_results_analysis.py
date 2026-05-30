"""
Plot and analyze Helium spectroscopy data for the Helium Results section.

Data format expected:
    two columns, no header:
        wavelength_nm    intensity

What the script creates:
    1. helium_spectrum_visible_colored.png/pdf
       - one full-spectrum figure
       - wavelength-colored spectrum
       - known He I line markers
       - fitted peak centers
       - horizontal arrows marking the exchange-energy pairs
    2. helium_peak_fit_results.csv
    3. helium_exchange_energy_results.csv
    4. helium_latex_peak_table.txt
    5. helium_latex_exchange_table.txt

Before running:
    Change DATA_DIR below to the folder containing your helium file.
    If you already have wavelength calibration from Hg, set APPLY_CALIBRATION=True
    and insert your calibration coefficients.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import curve_fit

# =============================================================================
# 1) USER SETTINGS: edit this section
# =============================================================================

DATA_DIR = Path(r"C:\Users\orlyk\Desktop\Jonathan k\School\lab job\lab C\Spectroscopy\Data – Results – Graphs\Helium")
HELIUM_FILE = DATA_DIR / "helium"
OUTPUT_DIR = DATA_DIR

# Fallback for testing inside ChatGPT sandbox. This will not affect your computer.
if not HELIUM_FILE.exists() and Path("/mnt/data/helium").exists():
    DATA_DIR = Path("/mnt/data")
    HELIUM_FILE = DATA_DIR / "helium"
    OUTPUT_DIR = DATA_DIR

# Calibration convention: lambda_calibrated = a * lambda_measured + b
# Use this ONLY if you already found a,b from your Hg calibration.
APPLY_CALIBRATION = False
CALIBRATION_A = 1.0
CALIBRATION_B_NM = 0.0

OPTIONS = {
    # General figure control
    "save_dpi": 300,
    "save_pdf": True,
    "save_png": True,
    "show_figures": True,
    "figure_size_helium": (13, 7),

    # Font sizes
    "title_fontsize": 32,
    "axis_label_fontsize": 24,
    "tick_fontsize": 17,
    "legend_fontsize": 15.4,

    # Lines and points
    "line_width": 5,
    "marker_size": 20,
    "marker_alpha": 1,
    "line_alpha": 1,
    "scatter_every_n_helium": 20,
    "show_wavelength_colored_points": True,
    "show_lines": True,

    # Axes and grid
    "grid": True,
    "grid_alpha": 0.25,
    "x_label": r"Wavelength $\lambda$ [nm]",
    "y_label": "Intensity [A.U.]",
    "helium_title": "Helium Emission Spectrum",
    "xlim_helium": (392, 795),        # example: (380, 790)
    "ylim_helium": None,
    "legend_location": "center left",

    # Markers and labels
    "mark_known_helium_lines": True,
    "helium_label_fontsize": 18.6,
    "helium_label_rotation": 90,
    "helium_line_marker_alpha": 0.54,

    # Vertical positions of the wavelength labels above each helium reference line.
    # These values use axes coordinates:
    #     0.00 = bottom of the plot
    #     1.00 = top of the plot
    #
    # Edit these numbers to move each individual helium wavelength label up/down.
    # The keys are the reference wavelengths written as strings.
    "helium_line_label_y_default": 0.97,
    "helium_line_label_y_positions": {
        "382.0": 0.80,
        "388.9": 0.96,
        "402.6": 0.80,
        "414.4": 0.96,
        "438.8": 0.80,
        "447.148": 0.96,
        "471.315": 0.80,
        "492.193": 0.96,
        "501.568": 0.80,
        "504.8": 0.96,
        "587.562": 0.80,
        "667.815": 0.96,
        "706.519": 0.81,
        "728.135": 0.96,
        "777.5": 0.80,
    },

    # Optional: control the vertical length of each gray helium marker line.
    # Each value is (ymin, ymax) in axes coordinates. Use (0.0, 1.0) for a full-height line.
    # Usually you only need to change the label heights above, not these marker ranges.
    "helium_line_marker_y_ranges": {
        # "447.148": (0.0, 0.95),
        # "492.193": (0.0, 0.90),
    },

    # Peak fitting
    "fit_known_helium_lines": True,
    "fit_window_half_width_nm": 1.5,

    # Exchange-pair arrows on the one and only figure
    "draw_exchange_arrows": True,
    "annotate_only_final_pairs": True,  # False will also draw optional/check pairs

    # Arrow appearance.
    # The arrows are one-sided because each pair is read from the triplet/ortho line
    # to the corresponding singlet/para line on the wavelength axis.
    "exchange_arrow_fontsize": 18,
    "exchange_arrow_color": "black",
    "exchange_arrow_alpha": 1,
    "exchange_arrow_lw": 3.2,
    "exchange_arrow_style": "->",       # one-sided arrow; use "<->" only for a two-sided distance marker

    # Label control.
    # Set this to False if you want only arrows and no text above them.
    # The formula text was removed from the label; the formula belongs in the article text/table.
    "exchange_arrow_show_labels": True,
    "exchange_arrow_label_offset": 0.014,

    # Vertical arrow positions in axes coordinates:
    # 0.00 is the bottom of the plot and 1.00 is the top of the plot.
    # Change these numbers to move individual arrows up/down.
    "exchange_arrow_y_positions": {
        "2s_from_3p": 0.12,
        "2p_from_5d": 0.20,
        "2p_from_4d": 0.34,
        "2p_from_3d": 0.50,
        "2p_from_6d": 0.60,
    },

    # Fallback spacing if a pair is missing from exchange_arrow_y_positions.
    "exchange_arrow_y_start": 0.87,
    "exchange_arrow_y_step": 0.065,

    # The strong line near 777 nm in some scans is not used for exchange-energy calculation.
    # Keep it as an observed feature only, unless your instructor identifies it as helium.
    "include_observed_777_feature": True,
}

# Helium singlet-triplet exchange-energy pairs.
# ortho/triplet line = lower-energy triplet system
# para/singlet line = higher-energy singlet system
#
# K = 0.5 * hc * |1/lambda_para - 1/lambda_ortho|
HELIUM_EXCHANGE_PAIRS = [
    {
        "K_pair": "2s_from_3p",
        "name": r"$1s3p \rightarrow 1s2s$",
        "quantity": r"$K_{2s}$",
        "ortho_nm": 388.9,
        "para_nm": 501.6,
        "use_for_final": True,
        "note": "Use only if the 388.9 nm line is inside the measured range and clearly visible.",
    },
    {
        "K_pair": "2p_from_5d",
        "name": r"$1s5d \rightarrow 1s2p$",
        "quantity": r"$K_{2p}$",
        "ortho_nm": 402.6,
        "para_nm": 438.8,
        "use_for_final": True,
        "note": "Additional 2p exchange pair; useful if both blue/violet peaks are resolved.",
    },
    {
        "K_pair": "2p_from_4d",
        "name": r"$1s4d \rightarrow 1s2p$",
        "quantity": r"$K_{2p}$",
        "ortho_nm": 447.1,
        "para_nm": 492.2,
        "use_for_final": True,
        "note": "Standard reliable 2p exchange pair.",
    },
    {
        "K_pair": "2p_from_3d",
        "name": r"$1s3d \rightarrow 1s2p$",
        "quantity": r"$K_{2p}$",
        "ortho_nm": 587.6,
        "para_nm": 667.8,
        "use_for_final": True,
        "note": "Standard reliable 2p exchange pair.",
    },
    {
        "K_pair": "2p_from_6d",
        "name": r"$1s6d \rightarrow 1s2p$",
        "quantity": r"$K_{2p}$",
        "ortho_nm": 382.0,
        "para_nm": 414.4,
        "use_for_final": False,
        "note": "Optional/check only; 382.0 nm may be outside the scan or weak.",
    },
]

PAIR_INFO = {pair["K_pair"]: pair for pair in HELIUM_EXCHANGE_PAIRS}

# Bright visible/near-visible He I lines relevant to this experiment.
# lambda_ref_nm values are reference line positions used only for identification/initial guesses.
HELIUM_LINES = [
    # Optional/check 2p pair: 1s6d -> 1s2p
    {
        "lambda_ref_nm": 382.0,
        "label": r"$1s6d\,^3D \rightarrow 1s2p\,^3P$",
        "transition_group": "1s6d -> 1s2p",
        "type": "ortho",
        "use_for_K": True,
        "K_pair": "2p_from_6d",
    },
    {
        "lambda_ref_nm": 414.4,
        "label": r"$1s6d\,^1D \rightarrow 1s2p\,^1P$",
        "transition_group": "1s6d -> 1s2p",
        "type": "para",
        "use_for_K": True,
        "K_pair": "2p_from_6d",
    },

    # 2s exchange pair: 1s3p -> 1s2s
    {
        "lambda_ref_nm": 388.9,
        "label": r"$1s3p\,^3P \rightarrow 1s2s\,^3S$",
        "transition_group": "1s3p -> 1s2s",
        "type": "ortho",
        "use_for_K": True,
        "K_pair": "2s_from_3p",
    },
    {
        "lambda_ref_nm": 501.568,
        "label": r"$1s3p\,^1P \rightarrow 1s2s\,^1S$",
        "transition_group": "1s3p -> 1s2s",
        "type": "para",
        "use_for_K": True,
        "K_pair": "2s_from_3p",
    },

    # Additional 2p pair: 1s5d -> 1s2p
    {
        "lambda_ref_nm": 402.6,
        "label": r"$1s5d\,^3D \rightarrow 1s2p\,^3P$",
        "transition_group": "1s5d -> 1s2p",
        "type": "ortho",
        "use_for_K": True,
        "K_pair": "2p_from_5d",
    },
    {
        "lambda_ref_nm": 438.8,
        "label": r"$1s5d\,^1D \rightarrow 1s2p\,^1P$",
        "transition_group": "1s5d -> 1s2p",
        "type": "para",
        "use_for_K": True,
        "K_pair": "2p_from_5d",
    },

    # Standard 2p pair: 1s4d -> 1s2p
    {
        "lambda_ref_nm": 447.148,
        "label": r"$1s4d\,^3D \rightarrow 1s2p\,^3P$",
        "transition_group": "1s4d -> 1s2p",
        "type": "ortho",
        "use_for_K": True,
        "K_pair": "2p_from_4d",
    },
    {
        "lambda_ref_nm": 492.193,
        "label": r"$1s4d\,^1D \rightarrow 1s2p\,^1P$",
        "transition_group": "1s4d -> 1s2p",
        "type": "para",
        "use_for_K": True,
        "K_pair": "2p_from_4d",
    },

    # Lines visible in the spectrum, but not clean K pairs with the simple formula
    {
        "lambda_ref_nm": 471.315,
        "label": r"$1s4s\,^3S \rightarrow 1s2p\,^3P$",
        "transition_group": "1s4s -> 1s2p",
        "type": "ortho",
        "use_for_K": True,
        "K_pair": "",
    },
    {
        "lambda_ref_nm": 504.8,
        "label": r"$1s4s\,^1S \rightarrow 1s2p\,^1P$",
        "transition_group": "1s4s -> 1s2p",
        "type": "para",
        "use_for_K": True,
        "K_pair": "",
    },

    # Standard 2p pair: 1s3d -> 1s2p
    {
        "lambda_ref_nm": 587.562,
        "label": r"$1s3d\,^3D \rightarrow 1s2p\,^3P$",
        "transition_group": "1s3d -> 1s2p",
        "type": "ortho",
        "use_for_K": True,
        "K_pair": "2p_from_3d",
    },
    {
        "lambda_ref_nm": 667.815,
        "label": r"$1s3d\,^1D \rightarrow 1s2p\,^1P$",
        "transition_group": "1s3d -> 1s2p",
        "type": "para",
        "use_for_K": True,
        "K_pair": "2p_from_3d",
    },

    # Visible but not clean K pairs with the simple formula
    {
        "lambda_ref_nm": 706.519,
        "label": r"$1s3s\,^3S \rightarrow 1s2p\,^3P$",
        "transition_group": "1s3s -> 1s2p",
        "type": "ortho",
        "use_for_K": True,
        "K_pair": "",
    },
    {
        "lambda_ref_nm": 728.135,
        "label": r"$1s3s\,^1S \rightarrow 1s2p\,^1P$",
        "transition_group": "1s3s -> 1s2p",
        "type": "para",
        "use_for_K": True,
        "K_pair": "",
    },
]

if OPTIONS["include_observed_777_feature"]:
    HELIUM_LINES.append({
        "lambda_ref_nm": 777.5,
        "label": r"observed feature near $777.5$ nm",
        "transition_group": "observed near 777.5 nm; not used for K",
        "type": "observed",
        "use_for_K": True,
        "K_pair": "",
    })

# Literature exchange energies for comparison. Update these if your course uses different references.
K_THEORY_EV = {
    "2s_from_3p": 0.39808,
    "2p_from_6d": 0.12690,
    "2p_from_5d": 0.12690,
    "2p_from_4d": 0.12690,
    "2p_from_3d": 0.12681,
}

# =============================================================================
# 2) HELPER FUNCTIONS
# =============================================================================

def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column spectroscopy file: wavelength_nm, intensity."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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
    if len(df) < 2:
        raise ValueError(f"File has fewer than 2 valid data rows: {path}")

    x = df["wavelength_nm"].to_numpy(dtype=float)
    y = df["intensity"].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if APPLY_CALIBRATION:
        x = CALIBRATION_A * x + CALIBRATION_B_NM

    return x, y


def wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> tuple[float, float, float]:
    """Approximate visible wavelength to RGB. Valid mainly for 380--790 nm."""
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
    return (float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1)))


def wavelength_colors(x_nm: np.ndarray) -> np.ndarray:
    return np.array([wavelength_to_rgb(v) for v in x_nm])


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base = Path(output_base)
    if OPTIONS["save_png"]:
        fig.savefig(output_base.with_suffix(".png"), dpi=OPTIONS["save_dpi"], bbox_inches="tight")
    if OPTIONS["save_pdf"]:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def apply_axis_style(ax, title=None, xlim=None, ylim=None):
    ax.set_xlabel(OPTIONS["x_label"], fontsize=OPTIONS["axis_label_fontsize"])
    ax.set_ylabel(OPTIONS["y_label"], fontsize=OPTIONS["axis_label_fontsize"])
    if title:
        ax.set_title(title, fontsize=OPTIONS["title_fontsize"])
    ax.tick_params(axis="both", labelsize=OPTIONS["tick_fontsize"])
    if OPTIONS["grid"]:
        ax.grid(True, alpha=OPTIONS["grid_alpha"])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def gaussian_linear_bg(x, b, m, A, mu, sigma):
    return b + m * (x - mu) + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_single_peak(x, y, lambda_ref_nm, window_half_width_nm):
    mask = (x >= lambda_ref_nm - window_half_width_nm) & (x <= lambda_ref_nm + window_half_width_nm)
    xx = x[mask]
    yy = y[mask]
    if len(xx) < 10:
        raise ValueError(f"Not enough points near {lambda_ref_nm:.3f} nm")

    # Use the local maximum as the starting point for the peak center.
    mu0 = float(xx[np.argmax(yy)])
    baseline0 = float(np.percentile(yy, 10))
    amp0 = float(np.max(yy) - baseline0)
    sigma0 = 0.12
    slope0 = 0.0

    p0 = [baseline0, slope0, max(amp0, 1e-12), mu0, sigma0]
    lower = [-np.inf, -np.inf, 0.0, lambda_ref_nm - window_half_width_nm, 0.005]
    upper = [np.inf, np.inf, np.inf, lambda_ref_nm + window_half_width_nm, 1.0]

    popt, pcov = curve_fit(
        gaussian_linear_bg,
        xx,
        yy,
        p0=p0,
        bounds=(lower, upper),
        maxfev=50000,
    )
    perr = np.sqrt(np.diag(pcov))

    b, m, A, mu, sigma = popt
    db, dm, dA, dmu, dsigma = perr
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    dfwhm = 2 * np.sqrt(2 * np.log(2)) * dsigma
    area = A * sigma * np.sqrt(2 * np.pi)
    darea = area * np.sqrt((dA / A) ** 2 + (dsigma / sigma) ** 2) if A > 0 and sigma > 0 else np.nan

    residuals = yy - gaussian_linear_bg(xx, *popt)
    dof = max(len(xx) - len(popt), 1)
    # This is an unweighted reduced residual variance, not a formal chi-square unless sigma_i is known.
    reduced_residual_variance = float(np.sum(residuals**2) / dof)

    return {
        "lambda_ref_nm": lambda_ref_nm,
        "lambda_fit_nm": float(mu),
        "d_lambda_fit_nm": float(dmu),
        "amplitude": float(A),
        "d_amplitude": float(dA),
        "sigma_nm": float(sigma),
        "d_sigma_nm": float(dsigma),
        "fwhm_nm": float(fwhm),
        "d_fwhm_nm": float(dfwhm),
        "area": float(area),
        "d_area": float(darea),
        "background": float(b),
        "slope": float(m),
        "reduced_residual_variance": reduced_residual_variance,
        "x_fit": xx,
        "y_fit": yy,
        "fit_parameters": popt,
    }


def propagate_K(lambda_ortho, d_lambda_ortho, lambda_para, d_lambda_para):
    """K = 0.5 * hc * |1/lambda_para - 1/lambda_ortho|, wavelengths in nm."""
    hc_eV_nm = 1239.841984
    K = 0.5 * hc_eV_nm * abs(1.0 / lambda_para - 1.0 / lambda_ortho)
    dK_dlo = 0.5 * hc_eV_nm / (lambda_ortho**2)
    dK_dlp = 0.5 * hc_eV_nm / (lambda_para**2)
    dK = np.sqrt((dK_dlo * d_lambda_ortho) ** 2 + (dK_dlp * d_lambda_para) ** 2)
    return float(K), float(dK)

# =============================================================================
# 3) ANALYSIS AND ONE-FIGURE PLOTTING FUNCTIONS
# =============================================================================

def fit_helium_peaks(x, y) -> pd.DataFrame:
    rows = []
    for line in HELIUM_LINES:
        lambda_ref = float(line["lambda_ref_nm"])
        try:
            r = fit_single_peak(
                x,
                y,
                lambda_ref_nm=lambda_ref,
                window_half_width_nm=OPTIONS["fit_window_half_width_nm"],
            )
            row = {k: v for k, v in r.items() if k not in ["x_fit", "y_fit", "fit_parameters"]}
            row.update({
                "transition": line["label"],
                "transition_group": line["transition_group"],
                "type": line["type"],
                "use_for_K": line["use_for_K"],
                "K_pair": line["K_pair"],
                "fit_success": True,
                "fit_message": "ok",
            })
            rows.append(row)
        except Exception as exc:
            rows.append({
                "lambda_ref_nm": lambda_ref,
                "lambda_fit_nm": np.nan,
                "d_lambda_fit_nm": np.nan,
                "transition": line["label"],
                "transition_group": line["transition_group"],
                "type": line["type"],
                "use_for_K": line["use_for_K"],
                "K_pair": line["K_pair"],
                "fit_success": False,
                "fit_message": str(exc),
            })
    return pd.DataFrame(rows)


def calculate_exchange_table(peak_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in HELIUM_EXCHANGE_PAIRS:
        pair_name = pair["K_pair"]
        pair_df = peak_df[
            (peak_df["K_pair"] == pair_name)
            & (peak_df["fit_success"] == True)
            & (peak_df["use_for_K"] == True)
        ]
        if len(pair_df) < 2:
            rows.append({
                "K_pair": pair_name,
                "quantity": pair["quantity"],
                "transition_group": pair["name"],
                "use_for_final": pair["use_for_final"],
                "fit_success": False,
                "fit_message": "Could not fit both ortho and para lines.",
                "lambda_ortho_nm": np.nan,
                "d_lambda_ortho_nm": np.nan,
                "lambda_para_nm": np.nan,
                "d_lambda_para_nm": np.nan,
                "K_measured_eV": np.nan,
                "d_K_measured_eV": np.nan,
                "K_theory_eV": K_THEORY_EV.get(pair_name, np.nan),
                "relative_error_percent": np.nan,
                "N_sigma_using_fit_error_only": np.nan,
                "note": pair["note"],
            })
            continue

        ortho_rows = pair_df[pair_df["type"] == "ortho"]
        para_rows = pair_df[pair_df["type"] == "para"]
        if len(ortho_rows) == 0 or len(para_rows) == 0:
            continue

        ortho = ortho_rows.iloc[0]
        para = para_rows.iloc[0]
        K, dK = propagate_K(
            lambda_ortho=ortho["lambda_fit_nm"],
            d_lambda_ortho=ortho["d_lambda_fit_nm"],
            lambda_para=para["lambda_fit_nm"],
            d_lambda_para=para["d_lambda_fit_nm"],
        )
        K_theory = K_THEORY_EV.get(pair_name, np.nan)
        rel_error_percent = abs(K - K_theory) / K_theory * 100 if np.isfinite(K_theory) else np.nan
        N_sigma = abs(K - K_theory) / dK if dK > 0 and np.isfinite(K_theory) else np.nan

        rows.append({
            "K_pair": pair_name,
            "quantity": pair["quantity"],
            "transition_group": pair["name"],
            "use_for_final": pair["use_for_final"],
            "fit_success": True,
            "fit_message": "ok",
            "lambda_ortho_nm": ortho["lambda_fit_nm"],
            "d_lambda_ortho_nm": ortho["d_lambda_fit_nm"],
            "lambda_para_nm": para["lambda_fit_nm"],
            "d_lambda_para_nm": para["d_lambda_fit_nm"],
            "K_measured_eV": K,
            "d_K_measured_eV": dK,
            "K_theory_eV": K_theory,
            "relative_error_percent": rel_error_percent,
            "N_sigma_using_fit_error_only": N_sigma,
            "note": pair["note"],
        })
    return pd.DataFrame(rows)



def _get_helium_line_option(line: dict, option_name: str, default):
    """Return a per-line plotting option using flexible wavelength keys.

    The options dictionaries use wavelength strings such as "447.148" or "504.8"
    as keys. This helper prevents crashes and also allows small formatting
    differences, for example "402.6", "402.600", or "402.6000".
    """
    option_dict = OPTIONS.get(option_name, {})
    if not isinstance(option_dict, dict):
        return default

    lam = float(line["lambda_ref_nm"])

    # Try the most common string formats first.
    possible_keys = [
        str(line["lambda_ref_nm"]),
        f"{lam:.4f}",
        f"{lam:.3f}",
        f"{lam:.2f}",
        f"{lam:.1f}",
    ]

    for key in possible_keys:
        if key in option_dict:
            return option_dict[key]

    # Last fallback: compare numeric values of the keys.
    # This makes "402.6" match 402.6000000000.
    for key, value in option_dict.items():
        try:
            if np.isclose(float(key), lam, rtol=0.0, atol=1e-6):
                return value
        except (TypeError, ValueError):
            continue

    return default

def _line_y_at(x, y, x0):
    """Interpolate spectrum intensity at x0. Returns nan if x0 is outside range."""
    if x0 < np.nanmin(x) or x0 > np.nanmax(x):
        return np.nan
    return float(np.interp(x0, x, y))


def add_exchange_arrows(ax, x, y, peak_df: pd.DataFrame, exchange_df: pd.DataFrame) -> None:
    """Draw one-sided horizontal arrows for the exchange-energy line pairs.

    Physics convention used here
    ----------------------------
    For the useful helium pairs, the triplet/ortho transition is at the shorter
    wavelength and the singlet/para transition is at the longer wavelength.
    Since E_gamma = hc/lambda, this means the triplet line has the larger photon
    energy. The plotted arrow therefore runs from ortho -> para on the wavelength
    axis. The numerical K value is still calculated separately in the table.

    Position convention used here
    -----------------------------
    x-coordinates are data coordinates, so the arrow starts/ends at fitted
    wavelengths. y-coordinates are axes coordinates, so 0 is the bottom of the
    plot and 1 is the top. This makes vertical placement independent of the
    measured intensity scale.
    """
    if not OPTIONS["draw_exchange_arrows"] or exchange_df is None or len(exchange_df) == 0:
        return

    # x uses physical wavelength units, y uses a normalized 0--1 axis fraction.
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # Build a list of pairs that were successfully fitted and should be drawn.
    pairs_to_draw = []
    for pair in HELIUM_EXCHANGE_PAIRS:
        # Keep only the final/report pairs unless optional/check pairs are requested.
        if OPTIONS["annotate_only_final_pairs"] and not pair["use_for_final"]:
            continue

        row = exchange_df[exchange_df["K_pair"] == pair["K_pair"]]
        if len(row) == 0 or not bool(row.iloc[0].get("fit_success", False)):
            continue

        lambda_ortho = row.iloc[0]["lambda_ortho_nm"]  # triplet line
        lambda_para = row.iloc[0]["lambda_para_nm"]    # singlet line
        if not np.isfinite(lambda_ortho) or not np.isfinite(lambda_para):
            continue

        # Skip arrows whose endpoints fall outside the measured x range.
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if min(lambda_ortho, lambda_para) < x_min or max(lambda_ortho, lambda_para) > x_max:
            continue

        pairs_to_draw.append((pair, row.iloc[0], lambda_ortho, lambda_para))

    for i, (pair, row, lambda_ortho, lambda_para) in enumerate(pairs_to_draw):
        # First try the explicit per-pair y-position. If it is not given, use the fallback ladder.
        y_positions = OPTIONS.get("exchange_arrow_y_positions", {})
        y_frac = y_positions.get(
            pair["K_pair"],
            OPTIONS["exchange_arrow_y_start"] - i * OPTIONS["exchange_arrow_y_step"],
        )
        y_frac = float(np.clip(y_frac, 0.02, 0.98))

        # One-sided physics direction: ortho/triplet -> para/singlet.
        # Do NOT sort the endpoints; sorting would erase the physical direction.
        x_start = lambda_ortho
        x_end = lambda_para

        ax.annotate(
            "",
            xy=(x_end, y_frac),       # arrow head: para/singlet line
            xytext=(x_start, y_frac), # arrow tail: ortho/triplet line
            xycoords=trans,
            textcoords=trans,
            arrowprops=dict(
                arrowstyle=OPTIONS.get("exchange_arrow_style", "->"),
                color=OPTIONS["exchange_arrow_color"],
                lw=OPTIONS["exchange_arrow_lw"],
                alpha=OPTIONS["exchange_arrow_alpha"],
                shrinkA=0,
                shrinkB=0,
            ),
        )

        # Optional short label. The exchange-energy formula is intentionally not printed here.
        if OPTIONS.get("exchange_arrow_show_labels", True):
            x_mid = 0.5 * (x_start + x_end)
            label = rf"{pair['quantity']}: {pair['name']}"
            ax.text(
                x_mid,
                y_frac + OPTIONS["exchange_arrow_label_offset"],
                label,
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=OPTIONS["exchange_arrow_fontsize"],
                color=OPTIONS["exchange_arrow_color"],
                alpha=OPTIONS["exchange_arrow_alpha"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
            )


def plot_helium_one_figure(x, y, peak_df=None, exchange_df=None) -> None:
    """Create the only figure: full Helium spectrum with peak centers and exchange arrows."""
    fig, ax = plt.subplots(figsize=OPTIONS["figure_size_helium"])

    if OPTIONS["show_lines"]:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_wavelengths = 0.5 * (x[:-1] + x[1:])
        segment_colors = wavelength_colors(segment_wavelengths)
        lc = LineCollection(
            segments,
            colors=segment_colors,
            linewidths=OPTIONS["line_width"],
            alpha=OPTIONS["line_alpha"],
            label="Helium spectrum colored by wavelength",
        )
        ax.add_collection(lc)
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale_view()

    if OPTIONS["show_wavelength_colored_points"]:
        step = max(int(OPTIONS["scatter_every_n_helium"]), 1)
        xs = x[::step]
        ys = y[::step]
        ax.scatter(
            xs,
            ys,
            s=OPTIONS["marker_size"],
            c=wavelength_colors(xs),
            alpha=OPTIONS["marker_alpha"],
            edgecolors="none",
        )

    if OPTIONS["mark_known_helium_lines"]:
        # The vertical line positions are controlled in axes coordinates.
        # This means 0 is the bottom of the plot and 1 is the top of the plot,
        # independent of the intensity scale.
        line_label_transform = blended_transform_factory(ax.transData, ax.transAxes)

        for line in HELIUM_LINES:
            lam = float(line["lambda_ref_nm"])
            if x.min() <= lam <= x.max():
                # Optional control over each marker line height.
                # Default is a full-height gray marker line.
                marker_range = _get_helium_line_option(
                    line,
                    "helium_line_marker_y_ranges",
                    (0.0, 1.0),
                )
                marker_ymin, marker_ymax = marker_range

                ax.axvline(
                    lam,
                    ymin=float(marker_ymin),
                    ymax=float(marker_ymax),
                    color="gray",
                    lw=1.35,
                    alpha=OPTIONS["helium_line_marker_alpha"],
                )

                # Per-line label height. Change OPTIONS["helium_line_label_y_positions"]
                # to move each wavelength label individually.
                label_y = _get_helium_line_option(
                    line,
                    "helium_line_label_y_positions",
                    OPTIONS["helium_line_label_y_default"],
                )
                label_y = float(np.clip(label_y, 0.02, 0.98))

                ax.text(
                    lam,
                    label_y,
                    f"{lam:.1f}",
                    transform=line_label_transform,
                    rotation=OPTIONS["helium_label_rotation"],
                    ha="right",
                    va="top",
                    fontsize=OPTIONS["helium_label_fontsize"],
                    color="black",
                )

    if peak_df is not None and len(peak_df) > 0:
        ok = peak_df[peak_df["fit_success"] == True]
        ax.scatter(
            ok["lambda_fit_nm"],
            [_line_y_at(x, y, v) for v in ok["lambda_fit_nm"]],
            s=50,
            facecolors="none",
            edgecolors="black",
            linewidths=5.6,
            label="Fitted peak centers",
        )

    title = OPTIONS["helium_title"]
    if APPLY_CALIBRATION:
        title += " (calibrated wavelength axis)"
    apply_axis_style(ax, title=title, xlim=OPTIONS["xlim_helium"], ylim=OPTIONS["ylim_helium"])

    add_exchange_arrows(ax, x, y, peak_df, exchange_df)

    ax.legend(fontsize=OPTIONS["legend_fontsize"], loc=OPTIONS["legend_location"])
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "helium_spectrum_visible_colored")
    if OPTIONS["show_figures"]:
        plt.show()
    else:
        plt.close(fig)


def dataframe_to_latex_tables(peak_df: pd.DataFrame, exchange_df: pd.DataFrame) -> None:
    peak_cols = [
        "transition_group", "type", "lambda_ref_nm", "lambda_fit_nm", "d_lambda_fit_nm", "fwhm_nm"
    ]
    exchange_cols = [
        "transition_group", "quantity", "use_for_final", "fit_success",
        "lambda_ortho_nm", "lambda_para_nm", "K_measured_eV", "d_K_measured_eV",
        "K_theory_eV", "relative_error_percent"
    ]

    peak_table = peak_df[peak_df["fit_success"] == True][peak_cols].copy()
    exchange_table = exchange_df[exchange_cols].copy() if len(exchange_df) > 0 else pd.DataFrame(columns=exchange_cols)

    with open(OUTPUT_DIR / "helium_latex_peak_table.txt", "w", encoding="utf-8") as f:
        f.write(peak_table.to_latex(index=False, float_format=lambda v: f"{v:.4g}", escape=False))

    with open(OUTPUT_DIR / "helium_latex_exchange_table.txt", "w", encoding="utf-8") as f:
        f.write(exchange_table.to_latex(index=False, float_format=lambda v: f"{v:.4g}", escape=False))

# =============================================================================
# 4) RUN
# =============================================================================

if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": OPTIONS["tick_fontsize"],
        "axes.titlesize": OPTIONS["title_fontsize"],
        "axes.labelsize": OPTIONS["axis_label_fontsize"],
        "legend.fontsize": OPTIONS["legend_fontsize"],
        "figure.dpi": 120,
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_spectrum(HELIUM_FILE)

    if OPTIONS["fit_known_helium_lines"]:
        peak_df = fit_helium_peaks(x, y)
        exchange_df = calculate_exchange_table(peak_df)

        peak_csv = OUTPUT_DIR / "helium_peak_fit_results.csv"
        exchange_csv = OUTPUT_DIR / "helium_exchange_energy_results.csv"
        peak_df.to_csv(peak_csv, index=False)
        exchange_df.to_csv(exchange_csv, index=False)
        dataframe_to_latex_tables(peak_df, exchange_df)

        print("Saved peak-fit table to:", peak_csv)
        print("Saved exchange-energy table to:", exchange_csv)
        print("\nPeak fit results:")
        print(peak_df[["lambda_ref_nm", "lambda_fit_nm", "d_lambda_fit_nm", "transition_group", "type", "fit_success"]])
        print("\nExchange energy results:")
        print(exchange_df)
    else:
        peak_df = None
        exchange_df = None

    # The only figure produced by this script.
    plot_helium_one_figure(x, y, peak_df=peak_df, exchange_df=exchange_df)

    print("Done. One Helium figure and all tables were saved in:", OUTPUT_DIR)
