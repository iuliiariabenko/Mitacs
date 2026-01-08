import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
df_reflectance = pd.read_csv("corrected_reflectance_no_baseline.csv")
df_transmission = pd.read_csv("oil_transmission_only.csv")

# === Prepare and merge data ===
R_df = df_reflectance[['Wavelength', 'Oil_0.8']].rename(columns={'Wavelength': 'Wavelength_nm', 'Oil_0.8': 'R'})
T_df = df_transmission[['Wavelength', 'T_Oil_0.8']].rename(columns={'Wavelength': 'Wavelength_nm', 'T_Oil_0.8': 'T'})
R_df['Wavelength_nm'] = R_df['Wavelength_nm'].round().astype(int)
T_df['Wavelength_nm'] = T_df['Wavelength_nm'].round().astype(int)

df_merged = pd.merge(R_df, T_df, on='Wavelength_nm').dropna()
df_merged['lambda_um'] = df_merged['Wavelength_nm'] / 1000.0
lambda_um = df_merged['lambda_um'].values
wavelength_m = lambda_um * 1e-6

# === Plot raw R and T curves ===
plt.figure(figsize=(10, 5))
plt.plot(lambda_um, df_merged['R'], label='Reflectance R(λ)', linewidth=2)
plt.plot(lambda_um, df_merged['T'], label='Transmittance T(λ)', linewidth=2)
plt.xlabel('Wavelength (μm)')
plt.ylabel('R and T (arb. units)')
plt.title('Optical Response of Oil_0.8 Layer')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Sellmeier model for fused silica ===
def fused_silica_n(lambda_um):
    B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
    C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2
    lambda_sq = lambda_um ** 2
    n_sq = 1 + (B1 * lambda_sq) / (lambda_sq - C1) + \
              (B2 * lambda_sq) / (lambda_sq - C2) + \
              (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_sq)

# === Dielectric function estimation ===
def epsilon_real(n0, nn, R, T, wavelength, thickness):
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (n0 ** 2 + nn ** 2) / 2
        term2 = 2 * n0 * nn * (2 * R) / T
        term3 = (n0 - nn) ** 2
        term4 = (nn*(1 - R - T) / T) ** 2
        term5 = ( (2 * np.pi * thickness/wavelength)** 2)*(n0** 2 - nn** 2)
        total = term2 + term3 + term4 + term5
        total[total < 0] = np.nan
        return term1 + (wavelength / (2 * np.pi * thickness)) * np.sqrt(total)

def epsilon_imaginary(nn, R, T, wavelength, thickness):
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = (wavelength / 2 / np.pi / thickness) * (1-R-T) /T
        return  nn * kappa

# === Constants and computation ===
thickness_mm = 4.0
thickness_m = thickness_mm / 1000.0
n_substrate = fused_silica_n(lambda_um)

R = df_merged['R'].values
T = df_merged['T'].values
eps1 = epsilon_real(n_substrate, n_substrate, R, T, wavelength_m, thickness_m)
eps2 = epsilon_imaginary(n_substrate, R, T, wavelength_m, thickness_m)

df_merged['epsilon_real'] = eps1
df_merged['epsilon_imag'] = eps2

# === Plot both real and imaginary parts with dual Y-axis ===
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Wavelength (μm)')
ax1.set_ylabel('Re(ε)', color=color1)
ax1.plot(df_merged['lambda_um'], df_merged['epsilon_real'], color=color1, label='Re(ε)', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Im(ε)', color=color2)
ax2.plot(df_merged['lambda_um'], df_merged['epsilon_imag'], color=color2, label='Im(ε)', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Complex Dielectric Function of Oil_0.8 vs Wavelength')
fig.tight_layout()
plt.grid(True)
plt.show()

# === Save output data ===
df_eps = df_merged[['lambda_um', 'epsilon_real', 'epsilon_imag']]
df_eps.to_csv("epsilon_Oil_0.8_only.csv", index=False)

# === Plot only imaginary part with full visibility ===
plt.figure(figsize=(10, 6))
plt.plot(df_merged['lambda_um'], df_merged['epsilon_imag'], label='Im(ε)', linewidth=2, color='orange')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Im(ε)')
plt.title('Imaginary Part of the Dielectric Function of Oil_0.8')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
