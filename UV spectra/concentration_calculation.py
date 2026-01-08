import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# === Загрузка и подготовка данных ===
file_path = "4-06-25 all oil with abs.csv"
df = pd.read_csv(file_path)
df_clean = df.iloc[1:].copy()

# Преобразование в числовые значения
df_clean["λ_Oil10H"] = pd.to_numeric(df_clean['Convert to Abs("Oil10H"):Oil10H'], errors='coerce')
df_clean["Abs_Oil10H"] = pd.to_numeric(df_clean["Unnamed: 13"], errors='coerce')
df_clean["λ_Oil1H"] = pd.to_numeric(df_clean['Convert to Abs("Oil1H"):Oil1H'], errors='coerce')
df_clean["Abs_Oil1H"] = pd.to_numeric(df_clean["Unnamed: 11"], errors='coerce')
df_clean["λ_Oil0_1H"] = pd.to_numeric(df_clean['Convert to Abs("Oil0_1H"):Oil0_1H'], errors='coerce')
df_clean["Abs_Oil0_1H"] = pd.to_numeric(df_clean["Unnamed: 9"], errors='coerce')

# === График поглощения (обычный) ===
plt.figure(figsize=(12, 6))
plt.plot(df_clean["λ_Oil10H"], df_clean["Abs_Oil10H"], label="Oil 10% concentration")
plt.plot(df_clean["λ_Oil1H"], df_clean["Abs_Oil1H"], label="Oil 1% concentration")
plt.plot(df_clean["λ_Oil0_1H"], df_clean["Abs_Oil0_1H"], label="Oil 0.1% concentration")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("Absorbance Spectra (Linear Scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Конвертация в dBm ===
df_clean["Abs_Oil10H_dBm"] = -10 * df_clean["Abs_Oil10H"]
df_clean["Abs_Oil1H_dBm"] = -10 * df_clean["Abs_Oil1H"]
df_clean["Abs_Oil0_1H_dBm"] = -10 * df_clean["Abs_Oil0_1H"]

# === Сглаживание и производные ===
window_length = 21
polyorder = 3
Abs_Oil10H_smooth = savgol_filter(df_clean["Abs_Oil10H"], window_length, polyorder)
Abs_Oil1H_smooth = savgol_filter(df_clean["Abs_Oil1H"], window_length, polyorder)
Abs_Oil0_1H_smooth = savgol_filter(df_clean["Abs_Oil0_1H"], window_length, polyorder)

df_clean["dAbs_Oil10H"] = np.gradient(Abs_Oil10H_smooth, df_clean["λ_Oil10H"])
df_clean["dAbs_Oil1H"] = np.gradient(Abs_Oil1H_smooth, df_clean["λ_Oil1H"])
df_clean["dAbs_Oil0_1H"] = np.gradient(Abs_Oil0_1H_smooth, df_clean["λ_Oil0_1H"])

# === Функция нахождения главного пика ===
def find_main_peak_with_abs(wavelengths, absorbance, derivative, wl_min=240, wl_max=290):
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wl = wavelengths[mask].reset_index(drop=True)
    ab = absorbance[mask].reset_index(drop=True)
    der = derivative[mask].reset_index(drop=True)
    zero_crossings = np.where(np.diff(np.sign(der)))[0]
    if len(zero_crossings) == 0:
        return np.nan, np.nan
    peak_wavelengths = (wl.iloc[zero_crossings].values + wl.iloc[zero_crossings + 1].values) / 2
    peak_absorbance = [np.interp(w, wl, ab) for w in peak_wavelengths]
    max_index = np.argmax(peak_absorbance)
    return peak_wavelengths[max_index], peak_absorbance[max_index]

# === Расчёт пиков ===
wl_10H, abs_10H = find_main_peak_with_abs(df_clean["λ_Oil10H"], df_clean["Abs_Oil10H"], df_clean["dAbs_Oil10H"])
wl_1H, abs_1H = find_main_peak_with_abs(df_clean["λ_Oil1H"], df_clean["Abs_Oil1H"], df_clean["dAbs_Oil1H"])
wl_0_1H, abs_0_1H = find_main_peak_with_abs(df_clean["λ_Oil0_1H"], df_clean["Abs_Oil0_1H"], df_clean["dAbs_Oil0_1H"])

# === Сводная таблица ===
peaks_summary = pd.DataFrame({
    "Sample": ["Oil10H", "Oil1H", "Oil0_1H"],
    "Main Peak Wavelength (nm)": [wl_10H, wl_1H, wl_0_1H],
    "Absorbance at Peak": [abs_10H, abs_1H, abs_0_1H],
    "Concentration (ml)": [10.0, 1.0, 0.1]
})

print("Основные пики по производной:")
print(peaks_summary)



# === График поглощения (dBm) с лазерными зонами ===
plt.figure(figsize=(14, 7))
plt.plot(df_clean["λ_Oil10H"], df_clean["Abs_Oil10H_dBm"], label="Oil 10% concentration (dBm)")
plt.plot(df_clean["λ_Oil1H"], df_clean["Abs_Oil1H_dBm"], label="Oil 1% concentration (dBm)")
plt.plot(df_clean["λ_Oil0_1H"], df_clean["Abs_Oil0_1H_dBm"], label="Oil 0.1% concentration (dBm)")

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
plt.plot(df_clean["λ_Oil10H"], df_clean["dAbs_Oil10H"], label="d(Abs)/dλ Oil 10% concentration")
plt.plot(df_clean["λ_Oil1H"], df_clean["dAbs_Oil1H"], label="d(Abs)/dλ Oil 1% concentration")
plt.plot(df_clean["λ_Oil0_1H"], df_clean["dAbs_Oil0_1H"], label="d(Abs)/dλ Oil 0.1% concentration")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Smoothed Derivative of Absorbance")
plt.title("First Derivative of Absorbance (Main Peak Zone)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Absorbance vs Concentration лог-график ===
plt.figure(figsize=(8, 5))
plt.plot(peaks_summary["Concentration (ml)"], peaks_summary["Absorbance at Peak"], marker='o')
plt.xscale('log')
plt.xlabel("Concentration (ml) [log scale]")
plt.ylabel("Absorbance at Main Peak")
plt.title("Absorbance vs Concentration (Main Peak)")
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# === Линейная регрессия A = k*C + b ===
def linear_model(c, k, b):
    return k * c + b

concentration = peaks_summary["Concentration (ml)"].values
absorbance = peaks_summary["Absorbance at Peak"].values

popt, _ = curve_fit(linear_model, concentration, absorbance)
k_fit, b_fit = popt
absorbance_pred = linear_model(concentration, k_fit, b_fit)

# R² вручную
ss_res = np.sum((absorbance - absorbance_pred) ** 2)
ss_tot = np.sum((absorbance - np.mean(absorbance)) ** 2)
r_squared = 1 - ss_res / ss_tot

# === График линейной аппроксимации ===
c_fit = np.linspace(0, 12, 200)
a_fit = linear_model(c_fit, k_fit, b_fit)

plt.figure(figsize=(8, 5))
plt.plot(concentration, absorbance, 'o', label='Measured Data')
plt.plot(c_fit, a_fit, '--', color='darkorange',
         label=f'Fit: A = {k_fit:.3f}·C + {b_fit:.3f}\n$R^2$ = {r_squared:.4f}')
plt.xlabel("Concentration (ml)")
plt.ylabel("Absorbance at Main Peak")
plt.title("Absorbance vs Concentration (Linear Fit)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
