import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === Load data ===
file_path = "oil_transmission_only.csv"
df = pd.read_csv(file_path)

df = df[['Wavelength', 'T_Oil_2.0']].dropna()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.astype(float)
df = df.sort_values('Wavelength')

wavelength_nm = df['Wavelength'].values
transmission_raw = df['T_Oil_2.0'].values

# === Apply Savitzky-Golay smoothing to transmission before derivation ===
# Choose a larger window for better smoothing (must be odd)
window_length = 31
polyorder = 3

transmission_smooth = savgol_filter(transmission_raw, window_length=window_length, polyorder=polyorder)

# === Compute second derivative from smoothed data ===
delta_lambda = np.mean(np.gradient(wavelength_nm))
d2T = - savgol_filter(transmission_smooth, window_length=window_length, polyorder=polyorder,
                    deriv=2, delta=delta_lambda)

# === Plot on dual Y-axes ===
fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'black'
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Transmission (a.u.)', color=color1, fontsize=12)
ax1.plot(wavelength_nm, transmission_smooth, color=color1, label='Smoothed T_Oil_0.1', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

# Second axis
ax2 = ax1.twinx()
color2 = 'blue'
ax2.set_ylabel(r'Second derivative $d^2T/d\lambda^2$ (a.u./nm²)', color=color2, fontsize=12)
ax2.plot(wavelength_nm, d2T, color=color2, label='Second derivative', linewidth=1)
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Smoothed T_Oil_0.1 and its Second Derivative vs Wavelength', fontsize=14)
fig.tight_layout()
plt.show()
