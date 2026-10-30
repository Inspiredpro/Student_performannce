import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")

# The exact feature names and order the model was trained with (from the notebook)
FEATURES = ["studytime", "failures", "absences", "age"]
STUDY_LABELS = {
    1: "1 - under 2 hours/week",
    2: "2 - 2 to 5 hours/week",
    3: "3 - 5 to 10 hours/week",
    4: "4 - over 10 hours/week",
}


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


def predict(model, row: pd.DataFrame):
    """Return (label, pass_probability)."""
    pass_idx = list(model.classes_).index(1)
    p_pass = float(model.predict_proba(row)[0][pass_idx])
    return p_pass


st.title("🎓 Student Success Predictor")
st.caption("Predicts whether a student is likely to pass (final grade ≥ 10) "
           "using early information, before any grades are known.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model.joblib: {e}")
    st.stop()

# Safety check: make sure the saved model expects the features this app collects
expected = list(getattr(model, "feature_names_in_", FEATURES))
if expected != FEATURES:
    st.error(
        "This app expects a model trained on "
        f"{FEATURES}, but model.joblib was trained on {expected}. "
        "Update FEATURES and the input widgets in app.py to match."
    )
    st.stop()

tab_single, tab_batch = st.tabs(["Single student", "Batch (CSV)"])

# ---------- Single student ----------
with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        studytime = st.selectbox("Weekly study time", options=list(STUDY_LABELS),
                                 format_func=STUDY_LABELS.get, index=1)
        failures = st.slider("Past class failures", 0, 3, 0)
    with c2:
        absences = st.number_input("Absences", min_value=0, max_value=100, value=4, step=1)
        age = st.slider("Age", 15, 22, 17)

    threshold = st.slider("Pass threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Predict PASS when the pass probability is at or above this value.")

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([[studytime, failures, absences, age]], columns=FEATURES)
        p_pass = predict(model, row)
        if p_pass >= threshold:
            st.success(f"PASS ✅  (pass probability {p_pass:.0%})")
        else:
            st.error(f"AT RISK of failing ❌  (pass probability {p_pass:.0%})")
        st.progress(p_pass, text=f"Probability of passing: {p_pass:.1%}")
        st.caption("This is a statistical estimate from four early indicators. "
                   "Use it to offer support, not to make decisions about a student.")

# ---------- Batch ----------
with tab_batch:
    st.write(f"Upload a CSV with these columns: `{', '.join(FEATURES)}`")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = data[FEATURES]
            pass_idx = list(model.classes_).index(1)
            data["PassProbability%"] = (model.predict_proba(X)[:, pass_idx] * 100).round(1)
            data["Prediction"] = (data["PassProbability%"] >= 50).map({True: "PASS", False: "AT RISK"})
            data = data.sort_values("PassProbability%")  # most at-risk first
            st.dataframe(data, use_container_width=True)
            st.download_button("Download results", data.to_csv(index=False),
                               "student_predictions.csv", "text/csv")
