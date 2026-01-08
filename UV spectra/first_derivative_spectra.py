import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

def fwhm(wavelengths, absorbance):
    absorbance = np.array(absorbance)
    wavelengths = np.array(wavelengths)
    half_max = np.max(absorbance) / 2.0
    indices = np.where(absorbance >= half_max)[0]
    if indices.size < 2:
        return np.nan, None, None
    λ1 = wavelengths[indices[0]]
    λ2 = wavelengths[indices[-1]]
    return abs(λ2 - λ1), λ1, λ2


# === Функция: интеграл под кривой поглощения ===
def integrate_absorbance_area(wavelengths, absorbance):
    """Calculate area under the absorbance curve using trapezoidal integration."""
    wavelengths = np.array(wavelengths)
    absorbance = np.array(absorbance)

    # Проверка и сортировка по возрастанию
    if wavelengths[0] > wavelengths[-1]:
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        absorbance = absorbance[sort_idx]

    return np.trapz(absorbance, wavelengths)

def absorption_centroid(wavelengths, absorbance):
    """
    Вычисляет центр тяжести (центроид) полосы поглощения.

    Parameters:
        wavelengths (array-like): длины волн, например, в нанометрах
        absorbance (array-like): соответствующие значения поглощения

    Returns:
        float: длина волны центра тяжести полосы поглощения
    """
    wavelengths = np.array(wavelengths)
    absorbance = np.array(absorbance)

    # Убедимся, что массивы отсортированы по длине волны
    if not np.all(np.diff(wavelengths) > 0):
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        absorbance = absorbance[sort_idx]

    # Вычисление центра тяжести
    numerator = np.trapz(wavelengths * absorbance, wavelengths)
    denominator = np.trapz(absorbance, wavelengths)

    if denominator == 0:
        return np.nan  # чтобы избежать деления на ноль

    return numerator / denominator



# === Безопасная фильтрация ===
def safe_savgol_filter(y, default_window=11, polyorder=3):
    y = np.asarray(y)
    if len(y) < polyorder + 2:
        return y  # слишком короткий вектор — возврат без фильтра
    # убедимся, что window <= len(y) и нечётный
    window = min(default_window, len(y) - (len(y) + 1) % 2)
    if window < polyorder + 2:
        window = polyorder + 2 + (polyorder + 2 + 1) % 2  # делаем нечётным
    if window > len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if window < polyorder + 2:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder)


# === Реализация tvregdiff ===
def tvregdiff(f, iter=200, alph=0.2):
    f = np.asarray(f).flatten()
    n = len(f)
    D = diags([-1, 1], [0, 1], shape=(n - 1, n)).tocsc()
    u = np.zeros_like(f)
    for _ in range(iter):
        Du = D @ u
        Du_norm = np.sqrt(Du**2 + 1e-8)
        W = diags(1.0 / Du_norm, 0)
        A = alph * (D.T @ W @ D) + eye(n)
        u = spsolve(A, f)
    d = D @ u
    return np.concatenate(([d[0]], d))

# === Поиск нулей производной с фильтрацией по расстоянию ===
def find_zero_crossings_and_absorbance(wavelengths, absorbance, derivative, min_spacing_nm=1.0, group_window_nm=10.0, min_abs=0.05):
    wavelengths = np.array(wavelengths)
    absorbance = np.array(absorbance)
    derivative = np.array(derivative)

    sign_changes = np.where(np.diff(np.sign(derivative)) != 0)[0]
    raw_points = []

    # Сначала находим все переходы через 0 производной
    for i in sign_changes:
        x0, x1 = wavelengths[i], wavelengths[i + 1]
        y0, y1 = derivative[i], derivative[i + 1]
        if (y1 - y0) != 0:
            x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
            abs_interp = interp1d(wavelengths, absorbance, kind='linear', bounds_error=False, fill_value="extrapolate")
            A_zero = float(abs_interp(x_zero))
            if A_zero > min_abs:
                raw_points.append((x_zero, A_zero))

    # Сортировка по длине волны
    raw_points = sorted(raw_points, key=lambda x: x[0])

    # Группировка по близости
    grouped = []
    current_group = []

    for x, a in raw_points:
        if not current_group:
            current_group.append((x, a))
        elif abs(x - current_group[-1][0]) <= group_window_nm:
            current_group.append((x, a))
        else:
            # Сохраняем максимум из группы
            best = max(current_group, key=lambda t: t[1])
            grouped.append(best)
            current_group = [(x, a)]

    if current_group:
        best = max(current_group, key=lambda t: t[1])
        grouped.append(best)

    return pd.DataFrame(grouped, columns=["Wavelength_zero_dAbs", "Absorbance_at_zero"])

def compute_peak_shifts_with_comments(df_peaks, df_centroids):
    """
    Вычисляет спектральные и интенсивностные сдвиги полос поглощения.

    Классификация:
    - ГИПСОХРОМНЫЙ СДВИГ: максимум или центр тяжести сместился влево (λ уменьшилась)
    - БАТОХРОМНЫЙ СДВИГ: максимум или центр тяжести сместился вправо (λ увеличилась)
    - ГИПЕРХРОМИЯ: поглощение увеличилось
    - ГИПОХРОМИЯ: поглощение уменьшилось

    Базовая точка сравнения — первый пик поглощения минимальной концентрации (Oil 0.1%)
    """

    # === Приведение названий к одному формату
    df_peaks = df_peaks.copy()
    df_peaks["Sample_clean"] = df_peaks["Sample"].str.lower().str.replace(" ", "").str.replace("(1st)", "").str.replace(
        "(2nd)", "")
    df_centroids = df_centroids.copy()
    df_centroids["Sample_clean"] = df_centroids["Sample"].str.lower().str.replace("spec_", "")

    # === Объединение таблиц
    df_merged = pd.merge(df_peaks, df_centroids, left_on="Sample_clean", right_on="Sample_clean", how="inner")

    # === Определение базовой точки: Oil 0.1%
    base_row = df_merged[df_merged["Sample_clean"].str.contains("0_1h")].iloc[0]
    λ_base_peak = base_row["Wavelength (nm)"]
    λ_base_centroid = base_row["Absorbance_Centroid"]
    A_base = base_row["Absorbance"]

    # === Расчёт сдвигов
    df_merged["Shift_Peak_nm"] = df_merged["Wavelength (nm)"] - λ_base_peak
    df_merged["Shift_Centroid_nm"] = df_merged["Absorbance_Centroid"] - λ_base_centroid
    df_merged["Delta_Absorbance"] = df_merged["Absorbance"] - A_base

    # === Классификация сдвигов
    def classify_shift(row):
        shift_peak = row["Shift_Peak_nm"]
        shift_centroid = row["Shift_Centroid_nm"]
        delta_abs = row["Delta_Absorbance"]

        # Сдвиги по длине волны
        if shift_peak < 0 or shift_centroid < 0:
            spectral = "ГИПСОХРОМНЫЙ СДВИГ (энергия перехода ↑)"
        elif shift_peak > 0 or shift_centroid > 0:
            spectral = "БАТОХРОМНЫЙ СДВИГ (энергия перехода ↓)"
        else:
            spectral = "Сдвиг по λ отсутствует"

        # Сдвиги по интенсивности
        if delta_abs > 0:
            intensity = "ГИПЕРХРОМИЯ (поглощение ↑)"
        elif delta_abs < 0:
            intensity = "ГИПОХРОМИЯ (поглощение ↓)"
        else:
            intensity = "Изменение поглощения отсутствует"

        return spectral + "; " + intensity

    df_merged["Interpretation"] = df_merged.apply(classify_shift, axis=1)

    return df_merged[[
        "Sample_x", "Wavelength (nm)", "Absorbance", "Absorbance_Centroid",
        "Shift_Peak_nm", "Shift_Centroid_nm", "Delta_Absorbance", "Interpretation"
    ]]


# === Загрузка данных ===
file_path = "4-06-25 all oil with abs.csv"
df = pd.read_csv(file_path)
df_clean = df.iloc[1:].copy()

# === Преобразование в числовой формат ===
df_clean["λ_Oil10H"] = pd.to_numeric(df_clean['Convert to Abs("Oil10H"):Oil10H'], errors='coerce')
df_clean["Abs_Oil10H"] = pd.to_numeric(df_clean["Unnamed: 13"], errors='coerce')
df_clean["λ_Oil1H"] = pd.to_numeric(df_clean['Convert to Abs("Oil1H"):Oil1H'], errors='coerce')
df_clean["Abs_Oil1H"] = pd.to_numeric(df_clean["Unnamed: 11"], errors='coerce')
df_clean["λ_Oil0_1H"] = pd.to_numeric(df_clean['Convert to Abs("Oil0_1H"):Oil0_1H'], errors='coerce')
df_clean["Abs_Oil0_1H"] = pd.to_numeric(df_clean["Unnamed: 9"], errors='coerce')

# === Удаление строк с NaN ===
df_clean = df_clean.dropna(subset=[
    "λ_Oil10H", "Abs_Oil10H",
    "λ_Oil1H", "Abs_Oil1H",
    "λ_Oil0_1H", "Abs_Oil0_1H"
])

# === Нахождение максимумов по 10% и 1% ===
λ_max_10H = df_clean["λ_Oil10H"][df_clean["Abs_Oil10H"].idxmax()]
Abs_max_10H = np.max(df_clean["Abs_Oil10H"])

λ_max_1H = df_clean["λ_Oil1H"][df_clean["Abs_Oil1H"].idxmax()]
Abs_max_1H = np.max(df_clean["Abs_Oil1H"])

# === Поиск двух пиков на Oil 0.1% ===
λ_0_1H = df_clean["λ_Oil0_1H"].values
Abs_0_1H = df_clean["Abs_Oil0_1H"].values
peaks_idx, _ = find_peaks(Abs_0_1H, distance=40, prominence=0.001)

# Сортировка по высоте и выбор двух наивысших
sorted_peaks = peaks_idx[np.argsort(Abs_0_1H[peaks_idx])[::-1]]
λ_max_0_1H_1 = λ_0_1H[sorted_peaks[0]] if len(sorted_peaks) >= 1 else None
Abs_max_0_1H_1 = Abs_0_1H[sorted_peaks[0]] if len(sorted_peaks) >= 1 else None
λ_max_0_1H_2 = λ_0_1H[sorted_peaks[1]] if len(sorted_peaks) >= 2 else None
Abs_max_0_1H_2 = Abs_0_1H[sorted_peaks[1]] if len(sorted_peaks) >= 2 else None

# === Словарь спектров для анализа ===
spectra = {
    "Oil10H": {
        "λ": df_clean["λ_Oil10H"].values,
        "Abs": df_clean["Abs_Oil10H"].values
    },
    "Oil1H": {
        "λ": df_clean["λ_Oil1H"].values,
        "Abs": df_clean["Abs_Oil1H"].values
    },
    "Oil0_1H": {
        "λ": df_clean["λ_Oil0_1H"].values,
        "Abs": df_clean["Abs_Oil0_1H"].values
    }
}

# === Добавим фильтрованное значение в spectra ===
for sample in spectra:
    Abs = spectra[sample]["Abs"]
    Abs_filt = safe_savgol_filter(Abs, default_window=11, polyorder=3)
    spectra[sample]["Abs_filt"] = Abs_filt


results = []
centroids_plot = []
fwhm_lines = []

for sample, data in spectra.items():
    λ = data["λ"]
    Abs = data["Abs_filt"]
    area = integrate_absorbance_area(λ, Abs)
    centroid = absorption_centroid(λ, Abs)
    fwhm_val, λ1, λ2 = fwhm(λ, Abs)

    results.append({
        "Sample": sample,
        "Absorbance_Area": area,
        "Absorbance_Centroid": centroid,
        "FWHM": fwhm_val
    })

    # Сохраним для графика
    centroids_plot.append((λ, Abs, centroid))
    if λ1 and λ2:
        fwhm_lines.append((λ1, λ2, Abs.max()))

# === График исходных данных + максимумы ===
plt.figure(figsize=(14, 6))
plt.plot(df_clean["λ_Oil10H"], df_clean["Abs_Oil10H"], label="Oil 10%")
plt.plot(df_clean["λ_Oil1H"], df_clean["Abs_Oil1H"], label="Oil 1%")
plt.plot(df_clean["λ_Oil0_1H"], df_clean["Abs_Oil0_1H"], label="Oil 0.1%")

# === Отметим максимумы красными точками и подпишем в легенде ===
plt.plot(λ_max_10H, Abs_max_10H, 'ro', label=f"Max 10%: {λ_max_10H:.1f} nm / {Abs_max_10H:.2f}")
plt.plot(λ_max_1H, Abs_max_1H, 'ro', label=f"Max 1%: {λ_max_1H:.1f} nm / {Abs_max_1H:.2f}")

if λ_max_0_1H_1 is not None:
    plt.plot(λ_max_0_1H_1, Abs_max_0_1H_1, 'ro', label=f"Max 0.1% (1st): {λ_max_0_1H_1:.1f} nm / {Abs_max_0_1H_1:.2f}")
if λ_max_0_1H_2 is not None:
    plt.plot(λ_max_0_1H_2, Abs_max_0_1H_2, 'ro', label=f"Max 0.1% (2nd): {λ_max_0_1H_2:.1f} nm / {Abs_max_0_1H_2:.2f}")

# Центры тяжести — пурпурные точки
for λ, Abs, c in centroids_plot:
    A_interp = interp1d(λ, Abs, kind='linear', bounds_error=False, fill_value="extrapolate")
    plt.plot(c, A_interp(c), 'o', color='purple', label="Centroid" if 'Centroid' not in plt.gca().get_legend_handles_labels()[1] else "")

# FWHM — жёлтые вертикальные линии
for λ1, λ2, Amax in fwhm_lines:
    plt.axvline(x=λ1, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=λ2, color='orange', linestyle='--', alpha=0.7)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("Original Absorbance Spectra with Peak Maxima")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Сохранение таблицы максимумов ===
df_maxima = pd.DataFrame({
    "Sample": ["Oil 10%", "Oil 1%", "Oil 0.1% (1st)", "Oil 0.1% (2nd)"],
    "Wavelength (nm)": [λ_max_10H, λ_max_1H, λ_max_0_1H_1, λ_max_0_1H_2],
    "Absorbance": [Abs_max_10H, Abs_max_1H, Abs_max_0_1H_1, Abs_max_0_1H_2]
})

df_maxima.to_csv("peak_maxima_summary.csv", index=False)
print("Файл с максимумами сохранён: peak_maxima_summary.csv")

# === Фильтрация спектров ===
df_clean["Abs_Oil10H_filt"] = safe_savgol_filter(df_clean["Abs_Oil10H"])
df_clean["Abs_Oil1H_filt"]  = safe_savgol_filter(df_clean["Abs_Oil1H"])
df_clean["Abs_Oil0_1H_filt"] = safe_savgol_filter(df_clean["Abs_Oil0_1H"])

# === Производные (на основе отфильтрованных данных) ===
df_clean["dAbs_Oil10H"]   = tvregdiff(df_clean["Abs_Oil10H_filt"].values, iter=200, alph=0.02)
df_clean["dAbs_Oil1H"]    = tvregdiff(df_clean["Abs_Oil1H_filt"].values,  iter=200, alph=0.02)
df_clean["dAbs_Oil0_1H"]  = tvregdiff(df_clean["Abs_Oil0_1H_filt"].values, iter=200, alph=0.02)

# === Поиск нулей производной ===
results_10H  = find_zero_crossings_and_absorbance(
    df_clean["λ_Oil10H"], df_clean["Abs_Oil10H_filt"], df_clean["dAbs_Oil10H"], min_spacing_nm=1.5)
results_1H   = find_zero_crossings_and_absorbance(
    df_clean["λ_Oil1H"],  df_clean["Abs_Oil1H_filt"],  df_clean["dAbs_Oil1H"],  min_spacing_nm=1.5)
results_0_1H = find_zero_crossings_and_absorbance(
    df_clean["λ_Oil0_1H"], df_clean["Abs_Oil0_1H_filt"], df_clean["dAbs_Oil0_1H"], min_spacing_nm=1.5)

# === График производных ===
plt.figure(figsize=(14, 6))
plt.grid(True)
plt.legend()
plt.plot(df_clean["λ_Oil10H"], df_clean["dAbs_Oil10H"], label="dAbs Oil 10%")
plt.plot(df_clean["λ_Oil1H"],  df_clean["dAbs_Oil1H"],  label="dAbs Oil 1%")
plt.plot(df_clean["λ_Oil0_1H"], df_clean["dAbs_Oil0_1H"], label="dAbs Oil 0.1%")
plt.show()



# === Собираем спектры в словарь для интеграции ===
spectra = {
    "Spec_Oil10H": {
        "λ": df_clean["λ_Oil10H"].values,
        "Abs_filt": df_clean["Abs_Oil10H_filt"].values
    },
    "Spec_Oil1H": {
        "λ": df_clean["λ_Oil1H"].values,
        "Abs_filt": df_clean["Abs_Oil1H_filt"].values
    },
    "Spec_Oil0_1H": {
        "λ": df_clean["λ_Oil0_1H"].values,
        "Abs_filt": df_clean["Abs_Oil0_1H_filt"].values
    }
}


results = []
colors = {"Oil10H": "blue", "Oil1H": "purple", "Oil0_1H": "gray"}
plt.figure(figsize=(14, 6))
for sample, data in spectra.items():
    λ = data["λ"]
    Abs = data["Abs_filt"]
    if len(λ) == 0 or len(Abs) == 0:
        continue
    area = integrate_absorbance_area(λ, Abs)
    centroid = absorption_centroid(λ, Abs)
    band_fwhm = fwhm(λ, Abs)
    results.append({
        "Sample": sample,
        "Absorbance_Area": area,
        "Absorbance_Centroid": centroid,
        "FWHM": band_fwhm
    })

df_results = pd.DataFrame(results)
df_results.to_csv("absorbance_centroids_corrected.csv", index=False)

df_maxima_clean = pd.DataFrame({
    "Sample": ["Oil 10%", "Oil 1%", "Oil 0.1% (1st)"],
    "Wavelength (nm)": [λ_max_10H, λ_max_1H, λ_max_0_1H_1],
    "Absorbance": [Abs_max_10H, Abs_max_1H, Abs_max_0_1H_1],
    "Sample_clean": ["oil10h", "oil1h", "oil0_1h"]
})


# === Загрузка центроидов из файла absorbance_centroids_corrected.csv ===
df_centroids = pd.read_csv("absorbance_centroids_corrected.csv")

# Приведение имён образцов к единому формату без префикса "Spec_" и в нижнем регистре
df_centroids["Sample_clean"] = df_centroids["Sample"].str.lower().str.replace("spec_", "", regex=False)

# === Приведение Sample_clean в df_maxima_clean к тому же формату
df_maxima_clean["Sample_clean"] = df_maxima_clean["Sample_clean"].str.lower()

# === Выполняем анализ смещения и интерпретации ===
df_shift_analysis = compute_peak_shifts_with_comments(df_maxima_clean, df_centroids)

# Сохраняем результат
df_shift_analysis.to_csv("spectral_shift_interpretation.csv", index=False)
print("Файл анализа сдвигов сохранён: spectral_shift_interpretation.csv")
