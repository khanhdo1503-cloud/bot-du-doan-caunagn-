import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================
WINDOW = 7
CONF_THRESHOLD = 0.55

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5-pPONvbU7PR7FteVtEBvN6EuudQ2rgbV3sHX-Ngy1PALF4nvyTBidXOXXE325_TLKKDJwZB7xFgH/pub?output=csv"

# =========================
# LOAD DATA (SAFE)
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_URL)

        if df is None or df.empty:
            return []

        values = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        values = values.dropna().astype(int).tolist()

        return values if isinstance(values, list) else []

    except:
        return []

# =========================
# INIT SESSION (ANTI CRASH)
# =========================
if "values" not in st.session_state:
    st.session_state.values = load_data()

if "model" not in st.session_state:
    st.session_state.model = None

# ép values luôn là list
if not isinstance(st.session_state.values, list):
    st.session_state.values = []

values = st.session_state.values

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot Pro")

# reload
if st.button("🔄 Reload Data"):
    st.session_state.values = load_data()
    st.success("Đã reload data")

# check data an toàn
if not values:
    st.warning("⚠️ Chưa có data hoặc load lỗi")
    st.stop()

# =========================
# HIỂN THỊ DATA
# =========================
st.subheader("📊 Data")

st.write(f"🔢 Tổng số data: {len(values)}")

st.dataframe(pd.DataFrame(values, columns=["Kết quả"]), height=300)

# =========================
# CREATE DATASET
# =========================
def create_dataset(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window] - 1)  # 1-4 → 0-3
    return np.array(X), np.array(y)

# =========================
# TRAIN MODEL
# =========================
if len(values) > WINDOW:

    if st.button("🧠 Train Model"):
        X, y = create_dataset(values, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        st.session_state.model = model
        st.success("Model trained ✅")

# =========================
# ADD NEW DATA
# =========================
st.subheader("➕ Nhập data mới")

new_value = st.number_input("Nhập (1-4)", min_value=1, max_value=4, step=1)

if st.button("➕ Thêm"):
    st.session_state.values.append(int(new_value))
    st.success(f"Đã thêm {new_value}")

# =========================
# PREDICT
# =========================
if st.session_state.model is not None:

    st.subheader("🔮 Dự đoán")

    if len(st.session_state.values) >= WINDOW:

        input_seq = st.session_state.values[-WINDOW:]
        input_seq = np.array(input_seq).reshape(1, -1)

        pred = st.session_state.model.predict_proba(input_seq)[0]

        choice = np.argmax(pred)
        confidence = np.max(pred)

        st.metric("🎯 Kết quả", choice + 1)
        st.metric("🔥 Confidence", f"{confidence*100:.2f}%")

        st.write("### 📊 Xác suất")
        for i, p in enumerate(pred):
            st.write(f"{i+1}: {p*100:.2f}%")

        if confidence > CONF_THRESHOLD:
            st.success("✅ NÊN CHƠI")
        else:
            st.warning("❌ BỎ QUA")

    else:
        st.warning("⚠️ Chưa đủ data để dự đoán")
