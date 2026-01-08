import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Constants and cuvette parameters ===
PI = np.pi
t = 500.0  # cuvette thickness in micrometers
n_cell = 1.5  # refractive index of cuvette
k_cell = 0.0  # assume cuvette has no absorption
R_reflection = 0.04  # reflection at interface (4%)

# === Load experimental data ===
file_path = "12_06_25_Oil.xlsx"  # path to your Excel file
df = pd.read_excel(file_path)

# === Extract and convert spectral data ===
wavelengths_cm1 = df['XDP'].values
T_measured = df['T_0.5 mm Oil'].values/100
wavelengths_um = 1e4 / wavelengths_cm1  # convert cm1 to µm

# === Function to compute Murman-based interference transmission ===
def murman_transmission(n, k, y, y0=1.0, y2=1.0, t=500.0):
    k = max(k, 1e-10)  # prevent division by zero
    q = np.exp(np.clip(4 * PI * t * k / y, -700, 700))
    q0 = np.exp(np.clip(-4 * PI * t * k / y, -700, 700))
    p = np.cos(4 * PI * t * n / y)
    p0 = np.sin(4 * PI * t * n / y)
    mod2 = n**2 + k**2

    A = ((n - y0)**2 + k**2) * ((n + y2)**2 + k**2)
    B = ((n + y0)**2 + k**2)**2 * ((n - y2)**2 + k**2)
    E = ((n + y0)**2 + k**2)**2 * ((n + y2)**2 + k**2)
    F = ((n - y0)**2 + k**2) * ((n - y2)**2 + k**2)
    C = mod2 * (y0**2 + y2**2) - mod2**2 - y0**2 * y2**2 - 4 * y0 * y2 * k**2
    G = mod2 * (y0**2 + y2**2) - mod2**2 - y0**2 * y2**2 + 4 * y0 * y2 * k**2
    D = k * (y2 - y0) * (mod2 + y0 * y2)
    H = k * (y2 + y0) * (mod2 - y0 * y2)

    ZR = A * q + B * q0 + 2 * C * p + 4 * D * p0
    ZT = 16 * y0 * y2 * mod2
    Z = E * q + F * q0 + 2 * G * p + 4 * H * p0
    return ZT / Z

# === Compute Murman-modeled interference spectrum ===
T_murman = np.array([murman_transmission(n_cell, k_cell, wl, y0=1.0, y2=n_cell, t=t) for wl in wavelengths_um])

# === Subtract interference to isolate absorption-only spectrum ===
T_filtered = T_measured - T_murman
T_filtered = np.clip(T_filtered, 0, 1)  # ensure values are within [0, 1]

# === Save result to CSV ===
output_df = pd.DataFrame({
    'Wavelength (Å)': wavelengths_cm1,
    'T_measured': T_measured,
    'T_interference_Murman': T_murman,
    'T_filtered': T_filtered
})
output_df.to_csv("T_filtered_Murman.csv", index=False)

# === Plot the results ===
plt.figure()
plt.plot(wavelengths_cm1, T_measured, label='Measured T(λ)')
plt.plot(wavelengths_cm1, T_murman, label='Murman Interference T(λ)', linestyle='--')
plt.plot(wavelengths_cm1, T_filtered, label='Filtered T(λ)', linestyle='-.')
plt.xlabel("Wavelength (Å)")
plt.ylabel("Transmission")
plt.title("FTIR Spectra: Murman Interference Filtering")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("T_filtered_Murman_plot.png")
plt.show()
