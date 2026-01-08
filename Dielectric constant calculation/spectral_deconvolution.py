import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.integrate import simpson

# === Step 1: Load experimental spectrum ===
file_path = "oil_transmission_only.csv"  # Замените на свой путь
df = pd.read_csv(file_path)
wavelength = df['Wavelength'].values.astype(float)
intensity = df['T_Oil_0.1'].values.astype(float)

# === Step 2: Peak detection for automatic bands ===
peaks, _ = find_peaks(-np.gradient(intensity), prominence=0.005)

# === Step 3: Siano–Metzler asymmetric log-normal function ===
def siano_metzler(nu, I_max, nu_max, W, gamma):
    ln2 = np.log(2)
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = ((nu_max - nu) / W) * np.sqrt((gamma**2 - 1) / (gamma**2 + 1)) + 1
        arg = np.where(arg <= 0, np.nan, arg)
        exponent = - (ln2**2 / np.log(gamma)**2) * np.log(arg)**2
    return np.nan_to_num(I_max * np.exp(exponent), nan=0.0, neginf=0.0, posinf=0.0)

# === Step 4: Combined model from multiple bands ===
def total_model_spectrum(nu, params, n_bands):
    spectrum = np.zeros_like(nu)
    for i in range(n_bands):
        I_max, nu_max, W, gamma = params[i*4:(i+1)*4]
        spectrum += siano_metzler(nu, I_max, nu_max, W, gamma)
    return spectrum

# === Step 5: Objective function for optimizer ===
def objective_function(params, nu, exp_spectrum, n_bands):
    model = total_model_spectrum(nu, params, n_bands)
    return np.sum((exp_spectrum - model) ** 2)

# === Step 6: Initialize parameters ===
initial_guess = [
    0.4, 230, 25, 1.5  # Manually added UV-band
]
bounds = [
    (0, 2), (190, 250), (5, 80), (1.01, 3.0)
]

# Add automatically detected bands
for p in peaks:
    nu_max_guess = wavelength[p]
    I_max_guess = intensity[p]
    W_guess = 30
    gamma_guess = 1.5
    initial_guess += [I_max_guess, nu_max_guess, W_guess, gamma_guess]
    bounds += [(0, 2), (190, 800), (5, 300), (1.01, 3.0)]

n_bands = len(initial_guess) // 4

# === Step 7: Fit using L-BFGS-B ===
result = minimize(objective_function, initial_guess,
                  args=(wavelength, intensity, n_bands),
                  method='L-BFGS-B',
                  bounds=bounds)

optimized_params = result.x
fitted_spectrum = total_model_spectrum(wavelength, optimized_params, n_bands)

# === Step 8: Compute area under each band and error metrics ===
band_integrals = []
band_curves = []
for i in range(n_bands):
    I_max, nu_max, W, gamma = optimized_params[i*4:(i+1)*4]
    band = siano_metzler(wavelength, I_max, nu_max, W, gamma)
    area = simpson(band, wavelength)
    band_integrals.append(area)
    band_curves.append(band)

residual = intensity - fitted_spectrum
rmse = np.sqrt(np.mean(residual**2))
mae = np.mean(np.abs(residual))

# === Step 9: Plot ===
plt.figure(figsize=(12, 6))
plt.plot(wavelength, intensity, label='Experimental spectrum', color='orange', linewidth=2)
plt.plot(wavelength, fitted_spectrum, '--', label='Fitted spectrum', color='blue', linewidth=2)

colors = plt.cm.viridis(np.linspace(0, 1, n_bands))
for i, band in enumerate(band_curves):
    plt.plot(wavelength, band, linestyle=':', color=colors[i],
             label=f'Band {i+1} (A={band_integrals[i]:.2f})')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'Spectral Deconvolution using Siano–Metzler\nEstimated bands: {n_bands} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 10: Print band data ===
band_data = pd.DataFrame({
    'Band': [f'Band {i+1}' for i in range(n_bands)],
    'I_max': optimized_params[::4],
    'nu_max (nm)': optimized_params[1::4],
    'W': optimized_params[2::4],
    'gamma': optimized_params[3::4],
    'Area (a.u.*nm)': band_integrals
})
print("\n=== Fitted Band Parameters ===")
print(band_data.to_string(index=False))
print(f"\nRMSE = {rmse:.5f}, MAE = {mae:.5f}")
