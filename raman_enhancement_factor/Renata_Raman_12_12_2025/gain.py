import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# SAMPLES + CONCENTRATIONS
# ============================================================

samples = {
    # -------- glass --------
    "0.5% BHT glass": {
        "file": "Oil_BHT_0_5_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
        "C": 0.005,
        "substrate": "glass",
    },
    "0.8% BHT glass": {
        "file": "Oil_BHT_0_8_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
        "C": 0.008,
        "substrate": "glass",
    },
    "1.0% BHT glass": {
        "file": "Oil_BHT_1%_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
        "C": 0.010,
        "substrate": "glass",
    },

    # -------- Au --------
    "0.08% BHT + Au": {
        "file": "Oil_BHT_0_08Au_sub.txt",
        "C": 0.0008,
        "substrate": "Au",
    },
    "0.008% BHT + Au": {
        "file": "Oil_BHT_0_008Au_sub.txt",
        "C": 0.00008,
        "substrate": "Au",
    },
}

# ============================================================
# SPECTRAL WINDOWS (cm^-1)
# ============================================================

AM_MIN, AM_MAX = 700.0, 1800.0        # matrix region
AB_MIN, AB_MAX = 1420.0, 1480.0       # BHT band

# ============================================================
# PROCESSING PARAMETERS
# ============================================================

BASELINE_WINDOW = 101   # only for wide region
BASELINE_POLY   = 3

SMOOTH_WINDOW = 15
SMOOTH_POLY   = 3

# ============================================================
# FUNCTIONS
# ============================================================

def process_wide_region(filename, wmin, wmax):
    """Baseline + smoothing for A_M"""
    df = pd.read_csv(filename, sep="\t", header=None, names=["Shift", "I"])
    x = df["Shift"].values
    y = df["I"].values

    mask = (x >= wmin) & (x <= wmax)
    x_roi = x[mask]
    y_roi = y[mask]

    baseline = savgol_filter(y_roi, BASELINE_WINDOW, BASELINE_POLY)
    y_corr = y_roi - baseline
    y_smooth = savgol_filter(y_corr, SMOOTH_WINDOW, SMOOTH_POLY)

    y_shifted = y_smooth - np.min(y_smooth)
    return x_roi, y_shifted


def process_narrow_band(filename, wmin, wmax):
    """ONLY smoothing for A_B"""
    df = pd.read_csv(filename, sep="\t", header=None, names=["Shift", "I"])
    x = df["Shift"].values
    y = df["I"].values

    mask = (x >= wmin) & (x <= wmax)
    x_roi = x[mask]
    y_roi = y[mask]

    y_smooth = savgol_filter(y_roi, SMOOTH_WINDOW, SMOOTH_POLY)
    y_shifted = y_smooth - np.min(y_smooth)
    return x_roi, y_shifted

# ============================================================
# CALCULATIONS
# ============================================================

rows = []
spectra_for_plot = {}

for name, s in samples.items():
    file = s["file"]
    C = s["C"]
    substrate = s["substrate"]

    # ---- A_M
    xM, yM = process_wide_region(file, AM_MIN, AM_MAX)
    A_M = np.trapz(yM, xM)

    # ---- A_B
    xB, yB = process_narrow_band(file, AB_MIN, AB_MAX)
    A_B = np.trapz(yB, xB)

    spectra_for_plot[name] = (xM, yM)

    rows.append({
        "Sample": name,
        "Substrate": substrate,
        "C": C,
        "A_M": A_M,
        "A_B": A_B,
        "(A_B/A_M)/C": (A_B / A_M) / C
    })

df = pd.DataFrame(rows)
print("\n=== Integrated quantities ===")
print(df)

# ============================================================
# GS,dB
# ============================================================

glass_ref = df[df["Substrate"] == "glass"]["(A_B/A_M)/C"].mean()

au_df = df[df["Substrate"] == "Au"].copy()
au_df["GS_dB"] = 10 * np.log10(
    au_df["(A_B/A_M)/C"] / glass_ref
)

print("\n=== GS,dB (Au vs glass) ===")
print(au_df[["Sample", "GS_dB"]])

# ============================================================
# PLOT (NO NORMALIZATION)
# ============================================================

plt.figure(figsize=(10, 6))

for name, (xM, yM) in spectra_for_plot.items():
    plt.plot(xM, yM, label=name)

plt.xlabel("Raman shift (cm$^{-1}$)")
plt.ylabel("Intensity (baseline-subtracted, shifted)")
plt.title("Processed Raman spectra (700–1800 cm$^{-1}$)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
