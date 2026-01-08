import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Загрузка CSV ===
file_path = "09-06-25 Without sphere.csv"
df_raw = pd.read_csv(file_path, skiprows=1)

# === Выводим имена колонок ===
print("Columns:", df_raw.columns.tolist())

# === Переименование по текущему формату ===
# Предполагаем, что структура: Wavelength, Abs, Abs.1, Abs.2 и т.д.
column_map = {
    'Wavelength (nm)': 'Wavelength',
    'Abs': 'Oil',
    'Abs.1': 'Oil_0.1',
    'Abs.2': 'Oil_0.2',
    'Abs.3': 'Oil_0.5',
    'Abs.4': 'Oil_0.8',
    'Abs.5': 'Oil_2.0'
}

df_renamed = df_raw.rename(columns=column_map)

# Удалим строки с NaN
df_clean = df_renamed[list(column_map.values())].dropna()

# Преобразуем все значения в числа
df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

# Переводим absorbance → transmission → восстановленная absorbance
for col in ['Oil', 'Oil_0.1', 'Oil_0.2', 'Oil_0.5', 'Oil_0.8', 'Oil_2.0']:
    df_clean[f"T_{col}"] = np.exp(-df_clean[col])  # Transmission
    df_clean.loc[df_clean[f"T_{col}"] > 1, f"T_{col}"] = 1.0  # Ограничение T <= 1
    df_clean[f"A_check_{col}"] = -np.log(df_clean[f"T_{col}"])  # Absorbance recovered

# === График Transmission ===
plt.figure(figsize=(12, 6))
for col, label in zip(
    ['T_Oil', 'T_Oil_0.1', 'T_Oil_0.2', 'T_Oil_0.5', 'T_Oil_0.8', 'T_Oil_2.0'],
    ['Oil (pure)', '0.1%', '0.2%', '0.5%', '0.8%', '2.0%']
):
    plt.plot(df_clean["Wavelength"], df_clean[col], label=label)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission (T = exp(-Abs))")
plt.title("Transmission Spectra of Oil Dilutions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === График восстановленной Absorbance ===
plt.figure(figsize=(12, 6))
for col, label in zip(
    ['A_check_Oil', 'A_check_Oil_0.1', 'A_check_Oil_0.2', 'A_check_Oil_0.5', 'A_check_Oil_0.8', 'A_check_Oil_2.0'],
    ['Oil (pure)', '0.1%', '0.2%', '0.5%', '0.8%', '2.0%']
):
    plt.plot(df_clean["Wavelength"], df_clean[col], label=label)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance (via -ln(T))")
plt.title("Absorbance Spectra of Oil Dilutions (Recomputed from Transmission)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Сохранение всех данных ===
output_file = "oil_absorption_transmission_analysis.csv"
df_clean.to_csv(output_file, index=False)
print(f"Данные сохранены в файл: {output_file}")

# === Сохранение только Transmission ===
trans_cols = ['Wavelength'] + [f"T_{c}" for c in ['Oil', 'Oil_0.1', 'Oil_0.2', 'Oil_0.5', 'Oil_0.8', 'Oil_2.0']]
df_clean[trans_cols].to_csv("oil_transmission_only.csv", index=False)
print("Transmission data saved to oil_transmission_only.csv")
