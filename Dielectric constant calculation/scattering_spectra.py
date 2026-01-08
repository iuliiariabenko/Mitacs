import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# === Загрузка данных ===
file_path = "09-06-25 With sphere.csv"
df = pd.read_csv(file_path, skiprows=1, usecols=[
    'Wavelength (nm)', 'Abs', 'Abs.1', 'Abs.2', 'Abs.3', 'Abs.4', 'Abs.5', 'Abs.6'])

# === Переименование столбцов ===
df.columns = ['Wavelength', 'Baseline', 'C01', 'C02', 'C05', 'C06', 'C2', 'Oil']
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# === Вычитание базовой линии ===
for col in ['C01', 'C02', 'C05', 'C06', 'C2', 'Oil']:
    df[col] = df[col] - df['Baseline']

# === Сглаживание и производные ===
window_length = 21
polyorder = 3
smooth_data = {}
derivatives = {}

for col in ['C01', 'C02', 'C05', 'C06', 'C2', 'Oil']:
    smooth = savgol_filter(df[col], window_length, polyorder)
    deriv = np.gradient(smooth, df['Wavelength'])
    smooth_data[col] = smooth
    derivatives[col] = deriv

# === Функция нахождения главного пика ===
def find_main_peak(wavelengths, signal, derivative, wl_min=240, wl_max=290):
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wl = np.array(wavelengths[mask])
    sig = np.array(signal[mask])
    der = derivative[mask]
    zero_crossings = np.where(np.diff(np.sign(der)))[0]
    if len(zero_crossings) == 0:
        return np.nan, np.nan
    peak_wavelengths = (wl[zero_crossings] + wl[zero_crossings + 1]) / 2
    peak_values = [np.interp(w, wl, sig) for w in peak_wavelengths]
    max_index = np.argmax(peak_values)
    return peak_wavelengths[max_index], peak_values[max_index]

# === Поиск пиков рассеяния ===
concentrations = [0.1, 0.2, 0.5, 0.8, 2.0, 100.0]  # Oil = 100%
labels = ['C01', 'C02', 'C05', 'C06', 'C2', 'Oil']
peak_wavelengths = []
peak_scattering = []

for label in labels:
    wl, sc = find_main_peak(df['Wavelength'], df[label], derivatives[label])
    peak_wavelengths.append(wl)
    peak_scattering.append(sc)

# === Сводная таблица ===
peaks_summary = pd.DataFrame({
    'Concentration (%)': concentrations,
    'Peak Wavelength (nm)': peak_wavelengths,
    'Scattering at Peak': peak_scattering
})

# === 1. График рассеяния (линейный масштаб) ===
plt.figure(figsize=(12, 6))
for label in labels:
    plt.plot(df['Wavelength'], df[label], label=label)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative Scattering (a.u.)")
plt.title("Scattering Spectra (Linear Scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. График в dBm с лазерными зонами ===
plt.figure(figsize=(14, 7))
for label in labels:
    scattering_dBm = 10 * np.log10(np.clip(df[label], a_min=1e-12, a_max=None))
    plt.plot(df['Wavelength'], scattering_dBm, label=f'{label} (dBm)')

for laser_wavelength, color, alpha in zip([785, 1064], ['magenta', 'black'], [0.1, 0.1]):
    plt.axvline(x=laser_wavelength, color=color, linestyle='--', label=f'{laser_wavelength} nm laser')
    plt.axvspan(laser_wavelength, laser_wavelength + 197, color=color, alpha=alpha,
                label=f'Region {laser_wavelength}+197 nm')

plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative Scattering (dBm)")
plt.title("Scattering Spectra with Laser Zones (dBm Scale)")
plt.legend()
plt.grid(True)
plt.xlim(200, 1100)
plt.tight_layout()
plt.show()

# === 3. График производной ===
plt.figure(figsize=(12, 6))
for label in labels:
    plt.plot(df['Wavelength'], derivatives[label], label=f'd(R)/dλ {label}')
plt.xlabel("Wavelength (nm)")
plt.ylabel("First Derivative of Scattering")
plt.title("First Derivative of Scattering (Main Peak Region)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 4. Scattering vs Concentration (логарифмический график) ===
plt.figure(figsize=(8, 5))
plt.plot(peaks_summary["Concentration (%)"], peaks_summary["Scattering at Peak"], marker='o')
plt.xscale('log')
plt.xlabel("Concentration (%) [log scale]")
plt.ylabel("Scattering at Main Peak")
plt.title("Scattering vs Concentration (Main Peak)")
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# === 5–6–7. Линейная регрессия + R² + график ===
def linear_model(c, k, b):
    return k * c + b

concentration = peaks_summary["Concentration (%)"].values
scattering = peaks_summary["Scattering at Peak"].values
popt, _ = curve_fit(linear_model, concentration, scattering)
k_fit, b_fit = popt
scattering_pred = linear_model(concentration, k_fit, b_fit)

# === R² вручную ===
ss_res = np.sum((scattering - scattering_pred) ** 2)
ss_tot = np.sum((scattering - np.mean(scattering)) ** 2)
r_squared = 1 - ss_res / ss_tot

# === График линейной аппроксимации ===
c_fit = np.linspace(0.05, 110, 300)
s_fit = linear_model(c_fit, k_fit, b_fit)

plt.figure(figsize=(8, 5))
plt.plot(concentration, scattering, 'o', label='Measured Data')
plt.plot(c_fit, s_fit, '--', color='darkorange',
         label=f'Fit: S = {k_fit:.3f}·C + {b_fit:.3f}\n$R^2$ = {r_squared:.4f}')
plt.xlabel("Concentration (%)")
plt.ylabel("Scattering at Main Peak")
plt.title("Scattering vs Concentration (Linear Fit)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
