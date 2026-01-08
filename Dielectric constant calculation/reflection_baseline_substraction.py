import pandas as pd
import matplotlib.pyplot as plt

# === Параметры ===
file_path = "09-06-25 With sphere.csv"  # путь к исходному CSV
output_path = "corrected_reflectance_no_baseline.csv"  # итоговый файл
SAT = 0.300  # порог насыщения

# === Загрузка и предварительная очистка ===
df_raw = pd.read_csv(file_path, skiprows=1)
df_raw['Wavelength (nm)'] = pd.to_numeric(df_raw['Wavelength (nm)'], errors='coerce')

# Указываем какие колонки нас интересуют (без baseline)
abs_columns = ['Abs', 'Abs.1', 'Abs.2', 'Abs.3', 'Abs.4', 'Abs.5']
labels = ['Oil', 'Oil_0.1', 'Oil_0.2', 'Oil_0.5', 'Oil_0.8', 'Oil_2.0']

# Создаем чистую копию без NaN
df_clean = df_raw.dropna(subset=['Wavelength (nm)'] + abs_columns).copy()

# === Коррекция отражения по SAT ===
for col in abs_columns:
    df_clean[f"R_corr_{col}"] = (pd.to_numeric(df_clean[col], errors='coerce') / SAT).clip(lower=0, upper=1.0)

# === Формирование итогового DataFrame ===
columns_to_save = ['Wavelength (nm)'] + [f"R_corr_{col}" for col in abs_columns]
df_result = df_clean[columns_to_save].copy()
df_result.columns = ['Wavelength'] + labels  # переименование

# === Сохранение в CSV ===
df_result.to_csv(output_path, index=False)
print(f"Скорректированные данные сохранены в файл: {output_path}")

# === Построение графика ===
plt.figure(figsize=(10, 6))
for label in labels:
    plt.plot(df_result['Wavelength'], df_result[label], label=label)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (corrected)")
plt.title("Corrected Reflectance Spectra (SAT scaled)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
