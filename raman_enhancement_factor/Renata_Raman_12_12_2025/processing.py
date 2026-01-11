import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# FILE GROUPS
# ============================================================

au_files = {
    "0.08% BHT + Au":  "Oil_BHT_0_08Au_sub.txt",
    "0.008% BHT + Au": "Oil_BHT_0_008Au_sub.txt",
}

glass_files = {
    "0.5% BHT": "Oil_BHT_0_5_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "0.8% BHT": "Oil_BHT_0_8_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "1.0% BHT": "Oil_BHT_1%_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
}

# ============================================================
# PARAMETERS
# ============================================================

RAMAN_MIN = 700
RAMAN_MAX = 1800

BASELINE_WINDOW = 101   # odd
BASELINE_POLY   = 3

SMOOTH_WINDOW = 15      # odd
SMOOTH_POLY   = 3

# ============================================================
# PROCESSING FUNCTION
# ============================================================

def process_and_normalize(filename):
    df = pd.read_csv(
        filename,
        sep="\t",
        header=None,
        names=["RamanShift", "Intensity"]
    )

    x = df["RamanShift"].values
    y = df["Intensity"].values

    # ROI
    mask = (x >= RAMAN_MIN) & (x <= RAMAN_MAX)
    x_roi = x[mask]
    y_roi = y[mask]

    # Baseline (SG)
    baseline = savgol_filter(
        y_roi,
        window_length=BASELINE_WINDOW,
        polyorder=BASELINE_POLY
    )

    y_corr = y_roi - baseline

    # Smoothing
    y_smooth = savgol_filter(
        y_corr,
        window_length=SMOOTH_WINDOW,
        polyorder=SMOOTH_POLY
    )

    # Normalization [0, 1]
    y_min, y_max = y_smooth.min(), y_smooth.max()
    y_norm = (y_smooth - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y_smooth)

    return x_roi, y_norm

# ============================================================
# FIGURE: ONE CANVAS, TWO PANELS
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(13, 5),
    sharey=True
)

# ---- LEFT: Au-enhanced spectra
ax_au = axes[0]
for label, filename in au_files.items():
    x_roi, y_norm = process_and_normalize(filename)
    ax_au.plot(x_roi, y_norm, label=label, linewidth=2)

ax_au.set_title("BHT + Au (SERS)")
ax_au.set_xlabel("Raman shift (cm$^{-1}$)")
ax_au.set_ylabel("Normalized intensity (0–1)")
ax_au.legend()
ax_au.grid(alpha=0.3)

# ---- RIGHT: Glass spectra
ax_glass = axes[1]
for label, filename in glass_files.items():
    x_roi, y_norm = process_and_normalize(filename)
    ax_glass.plot(x_roi, y_norm, label=label, linewidth=2)

ax_glass.set_title("BHT on glass")
ax_glass.set_xlabel("Raman shift (cm$^{-1}$)")
ax_glass.legend()
ax_glass.grid(alpha=0.3)

# ============================================================
# FINAL LAYOUT
# ============================================================

fig.suptitle(
    "Baseline-subtracted, smoothed, normalized Raman spectra\n"
    "CH-stretch region (2500–3000 cm$^{-1}$)",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
