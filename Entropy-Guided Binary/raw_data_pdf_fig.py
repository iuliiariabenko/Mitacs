import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Input files
# -----------------------------
files = {
    "0.08% BHT": "Oil_BHT_0_08_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "0.5% BHT":  "Oil_BHT_0_5_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "0.8% BHT":  "Oil_BHT_0_8_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "1.0% BHT":  "Oil_BHT_1%_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
}

# -----------------------------
# Raman ROI
# -----------------------------
RAMAN_MIN = 100
RAMAN_MAX = 3000

# -----------------------------
# IEEE figure style
# -----------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

#
fig, ax = plt.subplots(figsize=(5.0, 2.8))  # inches

# -----------------------------
# Plot spectra
# -----------------------------
for label, filename in files.items():
    df = pd.read_csv(
        filename,
        sep="\t",
        header=None,
        names=["RamanShift", "Intensity"]
    )

    mask = (df["RamanShift"] >= RAMAN_MIN) & (df["RamanShift"] <= RAMAN_MAX)
    df_roi = df.loc[mask].copy()

    # Shift intensity so min = 0
    df_roi["Intensity"] -= df_roi["Intensity"].min()

    ax.plot(
        df_roi["RamanShift"],
        df_roi["Intensity"],
        linewidth=1.1,
        label=label
    )

# -----------------------------
# Formatting
# -----------------------------
ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Raw Raman spectra")

ax.set_xlim(RAMAN_MIN, RAMAN_MAX)
ax.grid(alpha=0.25)

# -----------------------------
# Legend outside plot
# -----------------------------
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

# Reserve space for legend
fig.tight_layout(rect=[0, 0, 0.78, 1])

# -----------------------------
# Save PDF
# -----------------------------
fig.savefig(
    "Raman_raw_spectra_BHT.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
