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
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.2,
})

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "Raman_Oil_test.csv"
FIGURE_PDF = "raman_0p8_BHT.pdf"

TOTAL_BITS = 4096
K = 64
M = TOTAL_BITS // K
R_MIN = 1.0
R_MAX = 64.0

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
        W = np.diag(w)
        Z = W + lam * (D.T @ D)
        z = np.linalg.solve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def normalize_positive(x):
    x = np.maximum(x, 0.0)
    s = x.sum()
    return x / s if s > 0 else np.zeros_like(x)

# ============================================================
# ENTROPY / FISHER STRUCTURE
# ============================================================

def local_structure_energy(p, r, dnu):
    p_r = gaussian_filter1d(p, sigma=r / dnu, mode="reflect")
    logp = np.log(p_r + 1e-12)

    grad = np.zeros_like(logp)
    grad[1:-1] = (logp[2:] - logp[:-2]) / (2 * dnu)
    grad[0] = (logp[1] - logp[0]) / dnu
    grad[-1] = (logp[-1] - logp[-2]) / dnu

    return p_r * grad**2

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)
nu = df.iloc[:, 0].values
y = df.iloc[:, 1].values

if nu[0] > nu[-1]:
    nu = nu[::-1]
    y = y[::-1]

dnu = np.mean(np.diff(nu))

y = remove_cosmic_rays(y)
baseline = baseline_als(y)
p = normalize_positive(y - baseline)

# ============================================================
# BUILD REFERENCE BINARY MASK
# ============================================================

r_list = np.logspace(np.log10(R_MIN), np.log10(R_MAX), K)
segments = np.array_split(np.arange(len(p)), M)

E = np.zeros((K, M))

for k, r in enumerate(r_list):
    g = local_structure_energy(p, r, dnu)
    g /= (g.sum() + 1e-12)
    for m, idx in enumerate(segments):
        E[k, m] = g[idx].sum()

threshold = np.median(E)
reference_binary_mask = (E >= threshold).astype(np.uint8).reshape(-1)

bitstring = "".join(str(b) for b in reference_binary_mask.tolist())

# ============================================================
# IEEE-CORRECT RAMAN FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(3.35, 2.2))  # IEEE one-column

ax.plot(nu, y, color="black")

ax.set_xlabel("Raman shift (cm$^{-1}$)")
ax.set_ylabel("Intensity (a.u.)")

ax.tick_params(which="both", direction="in", top=True, right=True)

fig.tight_layout(pad=0.3)
plt.savefig(FIGURE_PDF, format="pdf", bbox_inches="tight")
plt.show()

# ============================================================
# TERMINAL OUTPUT (NOT IN FIGURE)
# ============================================================

print("Reference binary mask (0.8% BHT)")
print("Length:", len(reference_binary_mask))
print("Number of '1' bits:", int(reference_binary_mask.sum()))
print("First 128 bits:")
print(bitstring[:128])
