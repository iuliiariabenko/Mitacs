import os, requests
import streamlit as st
import pandas as pd
import plotly.express as px

API = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Fuel EDM", layout="wide")
st.title("Fuel EDM — Spectra / Model / Experiment Planner")

# ---- Sidebar: campaign & experiment creation ----
st.sidebar.header("Create")
with st.sidebar.expander("New Campaign", expanded=True):
    name = st.text_input("Campaign name", "Gastops SAF Study")
    customer = st.text_input("Customer", "Gastops")
    objective = st.text_area("Objective", "Relate Raman/UV features to composition and coking propensity.")
    if st.button("Create Campaign"):
        r = requests.post(f"{API}/campaign", json={"name": name, "customer": customer, "objective": objective})
        st.sidebar.success(f"Created: {r.json()['id']}")

with st.sidebar.expander("New Experiment / Run", expanded=False):
    campaign_id = st.number_input("Campaign ID", min_value=1, value=1)
    run_name = st.text_input("Run name", "Run_001")
    operator = st.text_input("Operator", "iuliia")
    if st.button("Create Experiment"):
        r = requests.post(f"{API}/experiment", json={"campaign_id": int(campaign_id), "run_name": run_name, "operator": operator})
        st.sidebar.success(f"Created: {r.json()['id']}")

# ---- Main: planning ----
st.header("Experiment planning")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Create Plan (factorial / ramp)")
    experiment_id = st.number_input("Experiment ID", min_value=1, value=1)
    plan_name = st.text_input("Plan name", "DOE_v1")

    plan_mode = st.selectbox("Mode", ["Factorial", "Ramp"])
    if st.button("Create Plan"):
        r = requests.post(f"{API}/plan", json={"experiment_id": int(experiment_id), "name": plan_name})
        st.session_state["plan_id"] = r.json()["id"]
        st.success(f"Plan created: {st.session_state['plan_id']}")

    plan_id = st.session_state.get("plan_id", 1)
    st.info(f"Active plan_id = {plan_id}")

    if plan_mode == "Factorial":
        temps = st.text_input("Temperatures (C)", "200,250,300")
        press = st.text_input("Pressure (bar)", "1,2")
        oxy   = st.text_input("O2 (ppm)", "0,50")
        hefa  = st.text_input("HEFA fraction", "0.2,0.5,0.8")
        duration = st.number_input("Step duration (s)", min_value=10, value=300)

        if st.button("Generate & Upload steps"):
            T = [float(x) for x in temps.split(",")]
            P = [float(x) for x in press.split(",")]
            O = [float(x) for x in oxy.split(",")]
            H = [float(x) for x in hefa.split(",")]

            steps = []
            idx = 0
            for t in T:
                for p in P:
                    for o in O:
                        for h in H:
                            steps.append({
                                "step_index": idx,
                                "duration_s": int(duration),
                                "temperature_C": t,
                                "pressure_bar": p,
                                "oxygen_ppm": o,
                                "hefa_fraction": h,
                                "note": "factorial"
                            })
                            idx += 1

            requests.post(f"{API}/plan/{plan_id}/steps", json=steps)
            st.success(f"Uploaded {len(steps)} steps")

    else:
        T_start = st.number_input("T start (C)", value=200.0)
        T_end   = st.number_input("T end (C)", value=350.0)
        n       = st.number_input("N steps", min_value=2, value=8)
        Pbar    = st.number_input("Pressure (bar)", value=2.0)
        Oppm    = st.number_input("O2 (ppm)", value=50.0)
        hfrac   = st.number_input("HEFA fraction", min_value=0.0, max_value=1.0, value=0.5)
        duration = st.number_input("Step duration (s)", min_value=10, value=300)

        if st.button("Generate Ramp & Upload steps"):
            Ts = list(pd.Series(pd.np.linspace(T_start, T_end, int(n))).values)  # simple
            steps = []
            for i, t in enumerate(Ts):
                steps.append({
                    "step_index": i,
                    "duration_s": int(duration),
                    "temperature_C": float(t),
                    "pressure_bar": float(Pbar),
                    "oxygen_ppm": float(Oppm),
                    "hefa_fraction": float(hfrac),
                    "note": "ramp"
                })
            requests.post(f"{API}/plan/{plan_id}/steps", json=steps)
            st.success(f"Uploaded {len(steps)} steps")

with col2:
    st.subheader("Plan viewer + schedule (Gantt-like)")
    plan_id = st.session_state.get("plan_id", 1)
    steps = requests.get(f"{API}/plan/{plan_id}/steps").json()
    df = pd.DataFrame(steps) if steps else pd.DataFrame(columns=[
        "step_index","duration_s","temperature_C","pressure_bar","oxygen_ppm","hefa_fraction","note"
    ])

    st.dataframe(df, use_container_width=True)

    if len(df) > 0:
        df_sorted = df.sort_values("step_index").copy()
        df_sorted["t_start_s"] = df_sorted["duration_s"].cumsum().shift(fill_value=0)
        df_sorted["t_end_s"] = df_sorted["t_start_s"] + df_sorted["duration_s"]
        fig = px.timeline(
            df_sorted,
            x_start="t_start_s", x_end="t_end_s", y="step_index",
            color="hefa_fraction",
            hover_data=["temperature_C","pressure_bar","oxygen_ppm","note"]
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download plan CSV",
            data=df_sorted.to_csv(index=False).encode("utf-8"),
            file_name=f"plan_{plan_id}.csv",
            mime="text/csv"
        )

st.header("Upload spectra (presigned S3)")
mp_id = st.number_input("MeasurementPoint ID (mp_id)", min_value=1, value=1)
sensor_type = st.selectbox("Sensor type", ["Raman", "UV", "SERS"])
if st.button("Get upload URL"):
    pres = requests.post(f"{API}/presign", params={"sensor_type": sensor_type, "mp_id": int(mp_id)}).json()
    st.code(pres["upload_url"])
    st.write("Save this URI to DB automatically via your acquisition script:", pres["s3_uri"])
