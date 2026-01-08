import numpy as np

def factorial_plan(temps, press, oxy, hefa, duration_s=300):
    """
    temps, press, oxy, hefa: iterables (e.g. [200, 250, 300], [1,2], [0,50], [0.2,0.5,0.8])
    Returns list of steps dicts
    """
    steps = []
    idx = 0
    for T in temps:
        for P in press:
            for O in oxy:
                for h in hefa:
                    steps.append(dict(
                        step_index=idx,
                        duration_s=duration_s,
                        temperature_C=float(T),
                        pressure_bar=float(P),
                        oxygen_ppm=float(O),
                        hefa_fraction=float(h),
                        note=None
                    ))
                    idx += 1
    return steps

def ramp_plan(T_start, T_end, n, P, O, h, duration_s=300):
    Ts = np.linspace(T_start, T_end, n)
    return [dict(step_index=i, duration_s=duration_s,
                 temperature_C=float(Ts[i]), pressure_bar=float(P),
                 oxygen_ppm=float(O), hefa_fraction=float(h), note="ramp")
            for i in range(n)]
