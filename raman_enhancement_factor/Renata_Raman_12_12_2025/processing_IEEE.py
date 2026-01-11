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

ROI_MIN, ROI_MAX = 700.0, 1800.0

BASELINE_WINDOW = 101
BASELINE_POLY   = 3

SMOOTH_WINDOW = 15
SMOOTH_POLY   = 3

OUTPUT_PDF = "Figure_SERS_vs_Glass_700_1800cm-1.pdf"

# ============================================================
# PROCESSING FUNCTION
# ============================================================

def process_wide_region(filename):
    df = pd.read_csv(filename, sep="\t", header=None, names=["Shift", "Intensity"])

    x = df["Shift"].values
    y = df["Intensity"].values

    mask = (x >= ROI_MIN) & (x <= ROI_MAX)
    x_roi = x[mask]
    y_roi = y[mask]

    baseline = savgol_filter(y_roi, BASELINE_WINDOW, BASELINE_POLY)
    y_corr = y_roi - baseline
    y_smooth = savgol_filter(y_corr, SMOOTH_WINDOW, SMOOTH_POLY)

    y_shifted = y_smooth - np.min(y_smooth)
    return x_roi, y_shifted

# ============================================================
# IEEE STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype": 42,   # TrueType (IEEE recommended)
    "ps.fonttype": 42,
})

# ============================================================
# FIGURE: TWO STACKED PANELS
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(3.5, 5.0),   # IEEE single-column
    sharex=True
)

# ---------- (a) Au ----------
ax1 = axes[0]
for label, filename in au_files.items():
    x, y = process_wide_region(filename)
    ax1.plot(x, y, label=label)

ax1.set_ylabel("Intensity (a.u.)")
ax1.set_title("(a) BHT on Au substrate (SERS)", loc="left")
ax1.legend(frameon=False)
ax1.grid(False)

# ---------- (b) Glass ----------
ax2 = axes[1]
for label, filename in glass_files.items():
    x, y = process_wide_region(filename)
    ax2.plot(x, y, label=label)

ax2.set_xlabel("Raman shift (cm$^{-1}$)")
ax2.set_ylabel("Intensity (a.u.)")
ax2.set_title("(b) BHT on glass substrate", loc="left")
ax2.legend(frameon=False)
ax2.grid(False)

# ============================================================
# SAVE + SHOW
# ============================================================

plt.tight_layout(pad=1.0)

plt.savefig(
    OUTPUT_PDF,
    format="pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved as: {OUTPUT_PDF}")
