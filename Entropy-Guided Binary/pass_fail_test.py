import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

# ============================================================
# IEEE STYLE SETTINGS
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.1,
})

# ============================================================
# FILES
# ============================================================

REFERENCE_FILE = "Raman_Oil_test.csv"

FILES = {
    "0.08% BHT": "Oil_BHT_0_08_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "0.5% BHT":  "Oil_BHT_0_5_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "0.8% BHT":  "Oil_BHT_0_8_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "1.0% BHT":  "Oil_BHT_1%_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
}

# ============================================================
# PARAMETERS
# ============================================================

TOTAL_BITS = 4096
K = 64
M = TOTAL_BITS // K
R_MIN, R_MAX = 1.0, 64.0

FIG_RAMAN = "fig1_raman_raw_spectra.pdf"
FIG_BER   = "fig2_ber_vs_sample.pdf"

# ============================================================
# ROBUST RAMAN FILE LOADER
# ============================================================

def load_raman_file(filename):
    """
    Ultra-robust loader for Raman .txt / .csv files with headers.
    Extracts first two numeric columns from file, ignoring text headers.
    """

    data = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # replace common separators
            line = line.replace(",", " ").replace(";", " ").replace("\t", " ")
            parts = line.strip().split()

            if len(parts) < 1:
                continue

            # try parsing numbers
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass

            # accept line if it has at least one numeric value
            if len(nums) >= 1:
                data.append(nums)

    if len(data) == 0:
        raise ValueError(f"{filename}: no numeric Raman data found")

    data = np.array(data, dtype=float)

    if data.shape[1] == 1:
        y = data[:, 0]
        nu = np.arange(len(y), dtype=float)
        print(f"[WARN] {filename}: single-column numeric data → synthetic Raman axis")

    else:
        nu = data[:, 0]
        y  = data[:, 1]

    # ensure increasing axis
    if nu[0] > nu[-1]:
        nu = nu[::-1]
        y  = y[::-1]

    return nu, y

# ============================================================
# PREPROCESSING
# ============================================================

def remove_cosmic_rays(y, kernel=5, thresh=6.0):
    if kernel % 2 == 0:
        kernel += 1
    med = medfilt(y, kernel_size=kernel)
    diff = np.abs(y - med)
    sigma = np.median(diff) * 1.4826 + 1e-12
    y2 = y.copy()
    y2[diff > thresh * sigma] = med[diff > thresh * sigma]
    return y2

def baseline_als(y, lam=1e6, p=0.01, niter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2, axis=0)
    w = np.ones(L)
    for _ in range(niter):
        Z = np.diag(w) + lam * (D.T @ D)
        z = np.linalg.solve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def normalize_positive(x):
    x = np.maximum(x, 0)
    return x / (x.sum() + 1e-12)

# ============================================================
# ENTROPY → BINARY CODE
# ============================================================

def local_structure_energy(p, r, dnu):
    p_r = gaussian_filter1d(p, sigma=r / dnu, mode="reflect")
    logp = np.log(p_r + 1e-12)
    grad = np.gradient(logp, dnu)
    return p_r * grad**2

def spectrum_to_bits(nu, y):
    dnu = np.mean(np.diff(nu))
    y = remove_cosmic_rays(y)
    y = y - baseline_als(y)
    p = normalize_positive(y)

    r_list = np.logspace(np.log10(R_MIN), np.log10(R_MAX), K)
    segments = np.array_split(np.arange(len(p)), M)

    E = np.zeros((K, M))
    for k, r in enumerate(r_list):
        g = local_structure_energy(p, r, dnu)
        g /= (g.sum() + 1e-12)
        for m, idx in enumerate(segments):
            E[k, m] = g[idx].sum()

    thr = np.median(E)
    return (E >= thr).astype(np.uint8).reshape(-1)

def ber(a, b):
    return np.mean(a != b)

# ============================================================
# REFERENCE MASK
# ============================================================

nu_ref, y_ref = load_raman_file(REFERENCE_FILE)
reference_bits = spectrum_to_bits(nu_ref, y_ref)

print("\nReference binary mask")
print("Length:", len(reference_bits))
print("Prefix (128 bits):")
print("".join(str(b) for b in reference_bits[:128]))

# ============================================================
# RAW RAMAN PLOT
# ============================================================

fig1, ax1 = plt.subplots(figsize=(3.5, 2.4))
spectra_bits = {}

for label, fname in FILES.items():
    nu, y = load_raman_file(fname)
    spectra_bits[label] = spectrum_to_bits(nu, y)
    ax1.plot(nu, y, label=label)

ax1.set_xlabel("Raman shift (cm$^{-1}$)")
ax1.set_ylabel("Intensity (a.u.)")
ax1.legend(fontsize=7)
fig1.tight_layout(pad=0.3)
plt.savefig(FIG_RAMAN, format="pdf", bbox_inches="tight")
plt.close(fig1)

# ============================================================
# BER + PASS / FAIL
# ============================================================

labels, ber_vals = [], []

for label, bits in spectra_bits.items():
    labels.append(label)
    ber_vals.append(ber(bits, reference_bits))

ber_vals = np.array(ber_vals)
threshold = np.median(ber_vals) + 3 * np.std(ber_vals)

results = []
for label, b in zip(labels, ber_vals):
    results.append({
        "Sample": label,
        "BER vs reference": b,
        "Decision": "PASS" if b <= threshold else "FAIL"
    })

df_results = pd.DataFrame(results)

print("\nPASS / FAIL TABLE")
print(df_results)

# ============================================================
# BER FIGURE
# ============================================================

fig2, ax2 = plt.subplots(figsize=(3.5, 2.4))
ax2.plot(labels, ber_vals, "o-", color="black")
ax2.axhline(threshold, linestyle="--", linewidth=0.8, color="black")

ax2.set_xlabel("Sample")
ax2.set_ylabel("BER vs reference")
ax2.set_xticklabels(labels, rotation=45, ha="right")

fig2.tight_layout(pad=0.3)
plt.savefig(FIG_BER, format="pdf", bbox_inches="tight")
plt.close(fig2)

print("\nSaved figures:")
print(FIG_RAMAN)
print(FIG_BER)
