import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

WINDOW = 7
CONF_THRESHOLD = 0.55

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    values = data.iloc[:, 0].dropna().tolist()
    values = [int(x)-1 for x in values]
    return values

# =========================
# CREATE DATASET
# =========================
def create_dataset(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot (Light Version)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    values = load_data(uploaded_file)
    st.success(f"Loaded {len(values)} data")

    X, y = create_dataset(values, WINDOW)

    if st.button("Train Model"):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        st.session_state.model = model
        st.success("Model trained ✅")

# =========================
# PREDICT
# =========================
if "model" in st.session_state and uploaded_file:

    st.subheader("🔮 Dự đoán")

    values = load_data(uploaded_file)
    input_seq = np.array(values[-WINDOW:]).reshape(1, -1)

    pred = st.session_state.model.predict_proba(input_seq)[0]

    choice = np.argmax(pred)
    confidence = np.max(pred)

    st.metric("Dự đoán", choice + 1)
    st.metric("Confidence", f"{confidence*100:.2f}%")

    st.write("### Xác suất:")
    for i, p in enumerate(pred):
        st.write(f"{i+1}: {p*100:.2f}%")

    if confidence > CONF_THRESHOLD:
        st.success("✅ NÊN CHƠI")
    else:
        st.warning("❌ BỎ QUA")
