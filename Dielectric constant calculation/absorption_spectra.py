import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# === Загрузка данных ===
file_path = "09-06-25 Without sphere.csv"
df = pd.read_csv(file_path, skiprows=1, usecols=[
    'Wavelength (nm)', 'Abs', 'Abs.1', 'Abs.2', 'Abs.3', 'Abs.4', 'Abs.5'])

# === Переименование столбцов ===
df.columns = ['Wavelength', 'Oil', 'C01', 'C02', 'C05', 'C08', 'C2']
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# === Сглаживание и производные ===
window_length = 21
polyorder = 3
smooth_data = {}
derivatives = {}
for col in ['Oil', 'C01', 'C02', 'C05', 'C08', 'C2']:
    smooth = savgol_filter(df[col], window_length, polyorder)
    deriv = np.gradient(smooth, df['Wavelength'])
    smooth_data[col] = smooth
    derivatives[col] = deriv

# === Функция нахождения главного пика ===
def find_main_peak(wavelengths, absorbance, derivative, wl_min=240, wl_max=290):
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wl = np.array(wavelengths[mask])
    ab = np.array(absorbance[mask])
    der = derivative[mask]
    zero_crossings = np.where(np.diff(np.sign(der)))[0]
    if len(zero_crossings) == 0:
        return np.nan, np.nan
    peak_wavelengths = (wl[zero_crossings] + wl[zero_crossings + 1]) / 2
    peak_absorbance = [np.interp(w, wl, ab) for w in peak_wavelengths]
    max_index = np.argmax(peak_absorbance)
    return peak_wavelengths[max_index], peak_absorbance[max_index]

# === Расчёт пиков ===
concentrations = [0.0, 0.1, 0.2, 0.5, 0.8, 2.0]
labels = ['Oil', 'C01', 'C02', 'C05', 'C08', 'C2']
peak_wavelengths = []
peak_absorbances = []

for label in labels:
    wl, ab = find_main_peak(df['Wavelength'], df[label], derivatives[label])
    peak_wavelengths.append(wl)
    peak_absorbances.append(ab)

# === Сводная таблица ===
peaks_summary = pd.DataFrame({
    'Concentration (%)': concentrations,
    'Peak Wavelength (nm)': peak_wavelengths,
    'Absorbance at Peak': peak_absorbances
})

# === График поглощения (обычный) ===
plt.figure(figsize=(12, 6))
for label in labels:
    plt.plot(df['Wavelength'], df[label], label=label)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("Absorbance Spectra (Linear Scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === График поглощения в dBm с лазерными зонами ===
plt.figure(figsize=(14, 7))
for label in labels:
    absorbance_dBm = -10 * df[label]
    plt.plot(df['Wavelength'], absorbance_dBm, label=f'{label} (dBm)')

for laser_wavelength, color, alpha in zip([785, 1064], ['magenta', 'black'], [0.1, 0.1]):
    plt.axvline(x=laser_wavelength, color=color, linestyle='--', label=f'{laser_wavelength} nm laser')
    plt.axvspan(laser_wavelength, laser_wavelength + 197, color=color, alpha=alpha,
                label=f'Region {laser_wavelength}+197 nm')

plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative Absorbance (dBm)")
plt.title("Absorbance Spectra (dBm Scale) with Laser Markers and Shaded Regions")
plt.legend()
plt.grid(True)
plt.xlim(200, 1100)
plt.tight_layout()
plt.show()

# === График производной ===
plt.figure(figsize=(12, 6))
for label in labels:
    plt.plot(df['Wavelength'], derivatives[label], label=f'd(Abs)/dλ {label}')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Smoothed Derivative of Absorbance")
plt.title("First Derivative of Absorbance (Main Peak Zone)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Absorbance vs Concentration (лог шкала) ===
plt.figure(figsize=(8, 5))
plt.plot(peaks_summary["Concentration (%)"], peaks_summary["Absorbance at Peak"], marker='o')
plt.xscale('log')
plt.xlabel("Concentration (%) [log scale]")
plt.ylabel("Absorbance at Main Peak")
plt.title("Absorbance vs Concentration (Main Peak)")
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# === Линейная регрессия A = k*C + b ===
def linear_model(c, k, b):
    return k * c + b

concentration = peaks_summary["Concentration (%)"].values
absorbance = peaks_summary["Absorbance at Peak"].values
popt, _ = curve_fit(linear_model, concentration, absorbance)
k_fit, b_fit = popt
absorbance_pred = linear_model(concentration, k_fit, b_fit)

# === R² вручную ===
ss_res = np.sum((absorbance - absorbance_pred) ** 2)
ss_tot = np.sum((absorbance - np.mean(absorbance)) ** 2)
r_squared = 1 - ss_res / ss_tot

# === График линейной аппроксимации ===
c_fit = np.linspace(0, 2.2, 200)
a_fit = linear_model(c_fit, k_fit, b_fit)

plt.figure(figsize=(8, 5))
plt.plot(concentration, absorbance, 'o', label='Measured Data')
plt.plot(c_fit, a_fit, '--', color='darkorange',
         label=f'Fit: A = {k_fit:.3f}·C + {b_fit:.3f}\n$R^2$ = {r_squared:.4f}')
plt.xlabel("Concentration (%)")
plt.ylabel("Absorbance at Main Peak")
plt.title("Absorbance vs Concentration (Linear Fit)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
