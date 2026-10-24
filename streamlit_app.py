"""
Colombo Dengue Early-Warning - Streamlit front end.
Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud -> main file path = streamlit_app.py
"""
import json, pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
from dengue_features import RAW_FIELDS          # also makes the pickle importable

st.set_page_config(page_title="Colombo Dengue Early Warning", page_icon="🦟", layout="centered")


@st.cache_resource
def load():
    with open(ROOT / "models" / "dengue_model.pkl", "rb") as f:
        model = pickle.load(f)
    meta = json.load(open(ROOT / "models" / "model_meta.json"))
    return model, meta


MODEL, META = load()
THRESH, HORIZON, RANGES = META["high_week_threshold_cases"], META["horizon_weeks"], META["field_ranges"]

# name, label, unit, step
FIELDS = [
    ("cases_t", "Dengue cases - this week", "cases", 1.0),
    ("cases_t1", "Dengue cases - 1 week ago", "cases", 1.0),
    ("cases_t2", "Dengue cases - 2 weeks ago", "cases", 1.0),
    ("cases_t3", "Dengue cases - 3 weeks ago", "cases", 1.0),
    ("epi_week", "Epidemiological week number", "1-53", 1.0),
    ("rain_4wk", "Rainfall, 4-week average", "mm/week", 0.5),
    ("rain_8wk", "Rainfall, 8-week average", "mm/week", 0.5),
    ("humidity_4wk", "Relative humidity, 4-week average", "%", 0.5),
    ("temp_4wk", "Temperature, 4-week average", "°C", 0.1),
    ("wind_4wk", "Wind speed, 4-week average", "km/h", 0.1),
    ("visib_4wk", "Visibility, 4-week average", "km", 0.1),
]
assert [f[0] for f in FIELDS] == RAW_FIELDS


def fill(inputs):                       # callback for the example buttons
    for k, v in inputs.items():
        st.session_state[k] = float(v)


# initial values = median of the training data
for name, *_ in FIELDS:
    st.session_state.setdefault(name, float(RANGES[name]["median"]))

st.title("🦟 Colombo Dengue Early Warning")
st.markdown(f"Probability that **{HORIZON} weeks from now** Colombo district records a **high-dengue week** "
            f"(more than **{THRESH:.0f}** notified cases).")

with st.sidebar:
    st.header("Try a real week")
    for i, ex in enumerate(META["examples"]):
        st.button(ex["label"], key=f"ex{i}", on_click=fill, args=(ex["inputs"],), help=ex["outcome"], use_container_width=True)
    st.divider()
    st.caption(f"Model: **{META['model']}**  \nTest ROC-AUC {META['test_auc']} "
               f"(95% CI {META['test_auc_ci'][0]}–{META['test_auc_ci'][1]})  \n"
               f"Trained on {META['data_range'][0]} to {META['data_range'][1]}  \n"
               "Source: Epidemiology Unit, Ministry of Health, Sri Lanka")

st.subheader("Enter the latest weekly figures")
cols = st.columns(2)
values = {}
for i, (name, label, unit, step) in enumerate(FIELDS):
    values[name] = cols[i % 2].number_input(f"{label} ({unit})", min_value=0.0, step=step, key=name)

if st.button("Predict", type="primary"):
    errs = []
    if not 1 <= values["epi_week"] <= 53:
        errs.append("Epidemiological week must be between 1 and 53.")
    if errs:
        for e in errs:
            st.error(e)
        st.stop()

    X = pd.DataFrame([values])[RAW_FIELDS].astype(float)
    p = float(MODEL.predict_proba(X)[0, 1])
    level, msg, fn = (("HIGH", "Likely high-dengue week ahead: consider stepping up vector control and hospital readiness.", st.error) if p >= 0.5 else
                      ("ELEVATED", "Some risk of a high week: keep monitoring closely.", st.warning) if p >= 0.25 else
                      ("LOW", "A high-dengue week is unlikely on current indicators.", st.success))
    c1, c2 = st.columns([1, 2])
    c1.metric("Risk of a high week", f"{p:.0%}")
    with c2:
        fn(f"**{level}**: {msg}")
        st.progress(min(max(p, 0.0), 1.0))

    for k in RAW_FIELDS:
        if not RANGES[k]["min"] <= values[k] <= RANGES[k]["max"]:
            st.warning(f"{k} = {values[k]:g} is outside the training range "
                       f"({RANGES[k]['min']:.1f} to {RANGES[k]['max']:.1f}); treat the result with extra caution.")

    # --- why? native XGBoost contributions (log-odds); no extra libraries needed ---
    try:
        import xgboost as xgb
        Xf = MODEL.named_steps["features"].transform(X)
        booster = MODEL.named_steps["model"].get_booster()
        contrib = booster.predict(xgb.DMatrix(Xf, feature_names=list(Xf.columns)), pred_contribs=True)[0][:-1]
        s = pd.Series(contrib, index=Xf.columns).sort_values(key=abs, ascending=False).head(8)[::-1]
        st.subheader("What pushed this prediction?")
        st.bar_chart(s, horizontal=True)
        st.caption("Positive bars raise the risk, negative bars lower it (XGBoost contributions in log-odds).")
    except Exception:
        pass

st.divider()
st.caption("Coursework decision-support prototype built on ~270 weekly records from one district; not validated for public-health use. "
           "The model anticipates sustained high periods better than sudden explosive surges.")
