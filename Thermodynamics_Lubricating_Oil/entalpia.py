# -*- coding: utf-8 -*-
"""
Eyring analysis using Aged corn oil optical data

Builds ln(A/T) vs 1/T plot and extracts:
ΔH‡, ΔS‡ from linear regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ------------------------------
# CONSTANTS
# ------------------------------
R = 8.314  # J/(mol·K)
kB = 1.380649e-23
h = 6.62607015e-34

# ------------------------------
# INPUT
# ------------------------------
CSV_FILE = "baseline_corrected_340_390nm_smoothed.csv"

TEMPERATURES_C = {
    "alina_aged_oil_BHT_0_8_t_28_55": 28.55,
    "alina_aged_oil_BHT_0_8_t_31_25": 31.25,
    "alina_aged_oil_BHT_0_8_t_32_15": 32.15,
    "alina_aged_oil_BHT_0_8_t_37_25": 37.25,
    "alina_aged_oil_BHT_0_8_t_44_5": 44.5,
    "alina_aged_oil_BHT_0_8_t_50": 50.0,
    "alina_aged_oil_BHT_0_8_t_59": 59.0,
}

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(CSV_FILE)
wl = df["Wavelength (nm)"].values

A_vals = []
invT = []

for name, T_C in TEMPERATURES_C.items():
    if name not in df.columns:
        continue

    T = T_C + 273.15  # K
    spectrum = df[name].values

    # area under curve A(T)
    A = np.trapz(spectrum, wl)

    A_vals.append(np.log(A / T))
    invT.append(1.0 / T)

A_vals = np.array(A_vals)
invT = np.array(invT)

# ------------------------------
# LINEAR REGRESSION
# ------------------------------
slope, intercept, r, p, stderr = linregress(invT, A_vals)

DeltaH = slope * R           # J/mol
DeltaS = R * (intercept - np.log(kB / h))  # J/(mol·K)

# ------------------------------
# PLOT
# ------------------------------
plt.figure(figsize=(5.2, 3.6))
plt.scatter(invT, A_vals, color="orange", label="Experimental data")
plt.plot(invT, intercept + slope * invT, color="tab:red",
         label="Theoretical calculation")

plt.xlabel("1/T (1/K)")
plt.ylabel("ln(A/T)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.title(
    f"ΔH‡ = {DeltaH/1000:.2f} kJ/mol, "
    f"ΔS‡ = {DeltaS:.2f} J/mol·K"
)

plt.legend()
plt.tight_layout()
plt.savefig("eyring_aged_oil.png", dpi=600)
plt.savefig("eyring_aged_oil.pdf")
plt.show()

# ------------------------------
# OUTPUT
# ------------------------------
print("Eyring analysis (Aged corn oil)")
print(f"ΔH‡ = {DeltaH/1000:.2f} kJ/mol")
print(f"ΔS‡ = {DeltaS:.2f} J/(mol·K)")
print(f"R²  = {r**2:.4f}")
