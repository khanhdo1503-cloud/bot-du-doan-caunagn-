import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

WINDOW = 7
CONF_THRESHOLD = 0.55

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5-pPONvbU7PR7FteVtEBvN6EuudQ2rgbV3sHX-Ngy1PALF4nvyTBidXOXXE325_TLKKDJwZB7xFgH/pub?output=csv"

# =========================
# LOAD DATA
# =========================
def load_data():
    try:
        df = pd.read_csv(CSV_URL)
        col = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        return col.dropna().astype(int).tolist()
    except:
        return []

# =========================
# INIT
# =========================
if "values" not in st.session_state:
    st.session_state.values = []

if "model" not in st.session_state:
    st.session_state.model = None

# =========================
# HEADER
# =========================
st.title("🧠 Fantan Bot UI")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Google Sheet"):
        st.session_state.values = load_data()
        st.success("Loaded data")

with col2:
    if st.button("♻️ Refresh Data"):
        st.session_state.values = []
        st.success("Reset data")

values = st.session_state.values

# =========================
# DATA INPUT
# =========================
st.subheader("📥 DATA INPUT")

data_input = st.text_area(
    "Dán data vào đây (ví dụ: 123412341234...)",
    height=200
)

# =========================
# PROCESS INPUT
# =========================
def parse_input(text):
    result = []
    for char in text:
        if char in ["1", "2", "3", "4"]:
            result.append(int(char))
    return result

# =========================
# DATASET
# =========================
def create_dataset(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window] - 1)
    return np.array(X), np.array(y)

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    new_values = parse_input(data_input)

    if len(new_values) == 0:
        st.error("❌ Không đọc được dữ liệu")
        st.stop()

    # update data
    st.session_state.values.extend(new_values)
    values = st.session_state.values

    st.success(f"Đã thêm {len(new_values)} giá trị")

    # TRAIN + PREDICT
    if len(values) > WINDOW:

        X, y = create_dataset(values, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        st.session_state.model = model

        seq = np.array(values[-WINDOW:]).reshape(1, -1)
        pred = model.predict_proba(seq)[0]

        choice = int(np.argmax(pred))
        conf = float(np.max(pred))

        st.subheader("🔮 RESULT")

        st.metric("Dự đoán", choice + 1)
        st.metric("Confidence", f"{conf*100:.2f}%")

        if conf > CONF_THRESHOLD:
            st.success("✅ NÊN CHƠI")
        else:
            st.warning("❌ BỎ QUA")

    else:
        st.warning("Chưa đủ data")

# =========================
# INFO
# =========================
st.divider()
st.write(f"📊 Tổng data: {len(st.session_state.values)}")
