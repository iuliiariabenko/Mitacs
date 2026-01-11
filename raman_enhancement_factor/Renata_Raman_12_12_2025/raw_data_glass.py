import pandas as pd
import matplotlib.pyplot as plt

files = {
    "0.08% BHT": "Oil_BHT_0_08_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "0.5% BHT":  "Oil_BHT_0_5_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "0.8% BHT":  "Oil_BHT_0_8_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",
    "1.0% BHT":  "Oil_BHT_1%_50Xlens_785_1200grating_25%_acc3_acq3s_AE50000_1.txt",

}

RAMAN_MIN = 700
RAMAN_MAX = 1800

plt.figure(figsize=(10, 6))

for label, filename in files.items():
    df = pd.read_csv(
        filename,
        sep="\t",
        header=None,
        names=["RamanShift", "Intensity"]
    )

    # 🔹 Фильтрация диапазона
    mask = (df["RamanShift"] >= RAMAN_MIN) & (df["RamanShift"] <= RAMAN_MAX)
    df_roi = df.loc[mask]

    plt.plot(
        df_roi["RamanShift"],
        df_roi["Intensity"],
        label=label,
        linewidth=1.2
    )

plt.xlabel("Raman shift (cm$^{-1}$)")
plt.ylabel("Intensity (a.u.)")
plt.title("Raw Raman spectra (2500–3000 cm$^{-1}$)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
