import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === Load input data ===
file_path = "12_06_25_Oil.xlsx"
df = pd.read_excel(file_path)

# === Extract data ===
wavenumbers_cm_inv = df['XDP'].values                   # ν [cm⁻¹]
T = df['T_0.5 mm Oil'].values / 100.0                   # % → fraction
R = df['Y'].values / 100.0

# Convert ν [cm⁻¹] → λ [μm]
wavelengths_um = 1e4 / wavenumbers_cm_inv

# === Constants ===
PI = np.pi
y0 = 2.4      # incident medium (e.g., ZnSe)
y2 = 2.4      # substrate
t = 3.2 * 2 * 1000  # thickness in μm

# === Cost function based on Murman model ===
def cost_function(X, y, y0, y2, t, r1, r2):
    n, k = X
    if n <= 0 or k < 0:
        return 1e6
    k = max(k, 1e-6)
    try:
        q = np.exp(np.clip(4 * PI * t * k / y, -700, 700))
        q0 = np.exp(np.clip(-4 * PI * t * k / y, -700, 700))
        p = np.cos(4 * PI * t * n / y)
        p0 = np.sin(4 * PI * t * n / y)
        mod2 = n**2 + k**2

        A = ((n - y0)**2 + k**2) * ((n + y2)**2 + k**2)
        B = ((n + y0)**2 + k**2)**2 * ((n - y2)**2 + k**2)
        E = ((n + y0)**2 + k**2)**2 * ((n + y2)**2 + k**2)
        F = ((n - y0)**2 + k**2) * ((n - y2)**2 + k**2)
        C = mod2 * (y0**2 + y2**2) - mod2**2 - y0**2 * y2**2 - 4 * y0 * y2 * k**2
        G = mod2 * (y0**2 + y2**2) - mod2**2 - y0**2 * y2**2 + 4 * y0 * y2 * k**2
        D = k * (y2 - y0) * (mod2 + y0 * y2)
        H = k * (y2 + y0) * (mod2 - y0 * y2)

        ZR = A * q + B * q0 + 2 * C * p + 4 * D * p0
        ZT = 16 * y0 * y2 * mod2
        Z = E * q + F * q0 + 2 * G * p + 4 * H * p0
        if Z == 0:
            return 1e6
        R_calc = ZR / Z
        T_calc = ZT / Z
        return (R_calc - r1)**2 + (T_calc - r2)**2
    except:
        return 1e6

# === Optimization routine ===
def compute_optical_constants_stable(wavelengths_um, R, T, y0, y2, t):
    n_array = np.zeros_like(wavelengths_um)
    k_array = np.zeros_like(wavelengths_um)

    for i, y in enumerate(wavelengths_um):
        r1, r2 = R[i], T[i]

        if r1 + r2 >= 1 or r1 < 0 or r2 < 0:
            n_array[i], k_array[i] = np.nan, np.nan
            continue

        bounds = [(0.1, 5.0), (1e-6, 5.0)]
        res = minimize(cost_function, [y2, 0.01], args=(y, y0, y2, t, r1, r2),
                       method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-9, 'maxiter': 1000})

        if res.success:
            n_array[i], k_array[i] = res.x
        else:
            print(f"NO CONVERGENCE at ν = {wavenumbers_cm_inv[i]:.2f} cm⁻¹")
            n_array[i], k_array[i] = np.nan, np.nan

    return n_array, k_array

# === Run computation ===
n_arr, k_arr = compute_optical_constants_stable(wavelengths_um, R, T, y0, y2, t)

# === Save results ===
results_df = pd.DataFrame({
    'Wavenumber (cm⁻¹)': wavenumbers_cm_inv,
    'Wavelength (µm)': wavelengths_um,
    'n': n_arr,
    'k': k_arr,
    '2nk': 2 * n_arr * k_arr,
    'n² - k²': n_arr**2 - k_arr**2
})
results_df.to_csv("optical_constants_refined_stable.csv", index=False)

# === Plot ===
plt.figure()
plt.plot(wavenumbers_cm_inv, n_arr, label='n (real part)')
plt.plot(wavenumbers_cm_inv, k_arr, label='k (imaginary part)')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Refractive Index')
plt.title('Spectral Refractive Index (Stable Murman Model)')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig("optical_constants_plot_refined_stable.png")
plt.show()

