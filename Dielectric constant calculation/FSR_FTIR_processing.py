import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# === PARAMETERS ===
n_cell = 2.4
k_cell = 0.0
t_total_um = (2 * 2.32 + 0.5) * 1000  # 2 ZnSe + 0.5 mm oil in micrometers

# === LOAD DATA ===
df = pd.read_excel("12_06_25_Oil.xlsx")
wavenumbers_cm1 = df["XDP"].values
T_measured = df["T_0.5 mm Oil"].values
T_measured_norm = T_measured / np.max(T_measured)
wavelengths_um = 1e4 / wavenumbers_cm1

# === MURMAN INTERFERENCE FUNCTION ===
def murman_transmission(n, k, wl_um, t_um, n0=1.0, n2=1.0):
    delta = 4 * np.pi * n * t_um / wl_um
    R = ((n - n0) / (n + n0))**2
    F = 4 * R / (1 - R)**2
    T = (1 / (1 + F * np.sin(delta / 2)**2)) * ((1 - R)**2 / (1 - R**2))
    return T

# === MODEL INTERFERENCE FROM ZnSe–OIL–ZnSe ===
T_murman = np.array([
    murman_transmission(n_cell, k_cell, wl, t_total_um, n0=1.0, n2=n_cell)
    for wl in wavelengths_um
])
T_filtered = np.clip(T_measured_norm - T_murman, 1e-6, 1.0)

# === ABSORBANCE ===
absorbance = -np.log10(T_filtered)

# === PEAK DETECTION ===
peaks, _ = find_peaks(absorbance, height=0.02, distance=10)
widths_result = peak_widths(absorbance, peaks, rel_height=0.5)
peak_positions = wavenumbers_cm1[peaks]
fwhm_cm1 = widths_result[0] * np.mean(np.diff(wavenumbers_cm1))
peak_heights = absorbance[peaks]

# === PEAK DATA ===
peak_data = pd.DataFrame({
    "Peak Position (cm^-1)": peak_positions,
    "FWHM (cm^-1)": fwhm_cm1,
    "Height (Absorbance)": peak_heights
})
ch_mask = (peak_data["Peak Position (cm^-1)"] >= 2800) & (peak_data["Peak Position (cm^-1)"] <= 3000)
ch_peaks = peak_data[ch_mask]

# === SAVE DATA ===
peak_data.to_csv("hexane_all_peaks.csv", index=False)
ch_peaks.to_csv("hexane_CH3_CH2_peaks.csv", index=False)
df_result = pd.DataFrame({
    "Wavenumber (cm^-1)": wavenumbers_cm1,
    "T_measured_norm": T_measured_norm,
    "T_murman": T_murman,
    "T_filtered": T_filtered,
    "Absorbance": absorbance
})
df_result.to_csv("T_filtered_Murman_final.csv", index=False)

# === PLOT ABSORBANCE AND PEAKS ===
plt.figure(figsize=(12, 6))
plt.plot(wavenumbers_cm1, absorbance, label='Absorbance A = -log₁₀(T)', color='black')
plt.plot(peak_positions, peak_heights, "ro", label='Detected Peaks')
for x, y in zip(peak_positions, peak_heights):
    if 2800 <= x <= 3000:
        plt.text(x, y + 0.02, f"{x:.0f}", ha='center', fontsize=8, color='darkred')
plt.gca().invert_xaxis()
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorbance")
plt.title("Hexane Absorption Peaks (CH₃/CH₂ Stretch Region Highlighted)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("hexane_absorption_peaks_plot.png")
plt.show()
