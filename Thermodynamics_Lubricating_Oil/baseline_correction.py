# -*- coding: utf-8 -*-
"""
Smoothed baseline-corrected spectra (340–390 nm)
Two panels on one canvas: AGED vs FRESH

Pipeline:
1) RAW spectra
2) light smoothing → baseline estimation (AsLS, pybaselines)
3) baseline subtraction from RAW
4) final smoothing
5) save processed data to CSV
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pybaselines import Baseline


# ============================================================
# PARAMETERS
# ============================================================

INPUT_CSV = "alina_active_thermodynamics_18_12_25.csv"
OUTPUT_CSV = "baseline_corrected_340_390nm_smoothed.csv"

WL_MIN = 340.0
WL_MAX = 390.0

# smoothing for baseline estimation
SG_BASELINE_WINDOW = 9
SG_BASELINE_POLY = 2

# smoothing for final signal
SG_FINAL_WINDOW = 15
SG_FINAL_POLY = 3

# AsLS baseline parameters
ASLS_LAM = 1e7
ASLS_P = 0.001


# ============================================================
# SAMPLE GROUPS
# ============================================================

AGED_SAMPLES = {
    "alina_aged_oil_BHT_0_8",
    "alina_aged_oil_BHT_0_8_t_28_55",
    "alina_aged_oil_BHT_0_8_t_31_25",
    "alina_aged_oil_BHT_0_8_t_32_15",
    "alina_aged_oil_BHT_0_8_t_37_25",
    "alina_aged_oil_BHT_0_8_t_44_5",
    "alina_aged_oil_BHT_0_8_t_50",
    "alina_aged_oil_BHT_0_8_t_59",
}

FRESH_SAMPLES = {
    "alina_fresh_oil",
    "alina_fresh_oil_BHT_0_8",
}


# ============================================================
# LABELS (as requested)
# ============================================================

LABELS = {
    "alina_aged_oil_BHT_0_8": "Aged (reference)",
    "alina_aged_oil_BHT_0_8_t_28_55": "28.55 °C",
    "alina_aged_oil_BHT_0_8_t_31_25": "31.25 °C",
    "alina_aged_oil_BHT_0_8_t_32_15": "32.15 °C",
    "alina_aged_oil_BHT_0_8_t_37_25": "37.25 °C",
    "alina_aged_oil_BHT_0_8_t_44_5": "44.5 °C",
    "alina_aged_oil_BHT_0_8_t_50": "50 °C",
    "alina_aged_oil_BHT_0_8_t_59": "59 °C",
    "alina_fresh_oil": "Fresh oil",
    "alina_fresh_oil_BHT_0_8": "Fresh + BHT",
}


# ============================================================
# IEEE STYLE
# ============================================================

def set_ieee_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
    })


# ============================================================
# LOAD RAW SPECTRA
# ============================================================

def load_multispectrum_csv(path):
    df = pd.read_csv(path, header=None, encoding="utf-8")
    header = df.iloc[0]

    spectra = {}
    for c in range(0, df.shape[1], 2):
        if pd.isna(header[c]):
            continue

        name = str(header[c]).strip().lstrip("\ufeff")
        if name == "Baseline 100%R":
            continue

        wl = pd.to_numeric(df.iloc[2:, c], errors="coerce").to_numpy()
        r = pd.to_numeric(df.iloc[2:, c + 1], errors="coerce").to_numpy()

        mask = np.isfinite(wl) & np.isfinite(r)
        wl, r = wl[mask], r[mask]

        order = np.argsort(wl)
        spectra[name] = (wl[order], r[order])

    return spectra


# ============================================================
# PROCESSING (baseline + smoothing)
# ============================================================

def process_window(spectra):
    processed = {}
    wl_ref = None

    for name, (wl, r_raw) in spectra.items():
        mask = (wl >= WL_MIN) & (wl <= WL_MAX)
        if np.count_nonzero(mask) < SG_FINAL_WINDOW:
            continue

        wl_win = wl[mask]
        r_win = r_raw[mask]

        # smoothing for baseline estimation
        r_smooth = savgol_filter(r_win, SG_BASELINE_WINDOW, SG_BASELINE_POLY)

        # baseline (AsLS)
        bl = Baseline(wl_win)
        baseline, _ = bl.asls(r_smooth, lam=ASLS_LAM, p=ASLS_P)

        # subtract baseline from RAW
        r_corr = r_win - baseline

        # final smoothing
        r_corr_smooth = savgol_filter(r_corr, SG_FINAL_WINDOW, SG_FINAL_POLY)

        processed[name] = r_corr_smooth
        if wl_ref is None:
            wl_ref = wl_win

    df = pd.DataFrame({"Wavelength (nm)": wl_ref})
    for name, values in processed.items():
        df[name] = values

    return df


# ============================================================
# PLOTTING: TWO PANELS
# ============================================================

def plot_two_panels(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)

    wl = df["Wavelength (nm)"].values

    # ---- LEFT: AGED ----
    ax = axes[0]
    for name in AGED_SAMPLES:
        if name in df.columns:
            ax.plot(wl, df[name], label=LABELS[name])

    ax.set_title("Aged corn oil")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Baseline-corrected Reflectance (%R)")
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.grid(True, linewidth=0.4)

    ax.legend(frameon=False)

    # ---- RIGHT: FRESH ----
    ax = axes[1]
    for name in FRESH_SAMPLES:
        if name in df.columns:
            ax.plot(wl, df[name], label=LABELS[name])

    ax.set_title("Fresh corn oil")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.grid(True, linewidth=0.4)

    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig("baseline_corrected_340_390nm_two_panels.png", dpi=600)
    fig.savefig("baseline_corrected_340_390nm_two_panels.pdf")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    set_ieee_style()

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    spectra = load_multispectrum_csv(INPUT_CSV)
    df_processed = process_window(spectra)

    df_processed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    plot_two_panels(df_processed)

    print("✔ Processing completed")
    print(f"✔ CSV saved: {OUTPUT_CSV}")
    print("✔ Figure saved: baseline_corrected_340_390nm_two_panels.png / .pdf")


if __name__ == "__main__":
    main()
