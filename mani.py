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
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(CSV_URL)
    values = df.iloc[:, 0].dropna().astype(int).tolist()
    return values

# =========================
# CREATE DATASET
# =========================
def create_dataset(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window] - 1)  # map về 0-3
    return np.array(X), np.array(y)

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot Pro")

# Load data button
if st.button("🔄 Load Data từ Google Sheet"):
    st.session_state.values = load_data()

# Nếu chưa load
if "values" not in st.session_state:
    st.warning("👉 Bấm Load Data trước")
    st.stop()

values = st.session_state.values

# =========================
# HIỂN THỊ DATA
# =========================
st.subheader("📊 Data hiện tại")

st.write(f"🔢 Tổng số data: {len(values)}")

df_show = pd.DataFrame(values, columns=["Kết quả"])
st.dataframe(df_show, height=300)

# =========================
# TRAIN MODEL
# =========================
if len(values) > WINDOW:
    X, y = create_dataset(values, WINDOW)

    if st.button("🧠 Train Model"):
        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)
        st.session_state.model = model
        st.success("Model đã train xong ✅")

# =========================
# NHẬP DATA MỚI
# =========================
st.subheader("➕ Nhập kết quả mới")

new_value = st.number_input("Nhập (1-4)", min_value=1, max_value=4, step=1)

if st.button("➕ Thêm vào data"):
    st.session_state.values.append(int(new_value))
    st.success(f"Đã thêm: {new_value}")

# =========================
# PREDICT
# =========================
if "model" in st.session_state and len(st.session_state.values) >= WINDOW:

    st.subheader("🔮 Dự đoán ván tiếp theo")

    input_seq = st.session_state.values[-WINDOW:]
    input_seq = np.array(input_seq).reshape(1, -1)

    pred = st.session_state.model.predict_proba(input_seq)[0]

    choice = np.argmax(pred)
    confidence = np.max(pred)

    st.metric("🎯 Dự đoán", choice + 1)
    st.metric("🔥 Confidence", f"{confidence*100:.2f}%")

    st.write("### 📊 Xác suất chi tiết")
    for i, p in enumerate(pred):
        st.write(f"{i+1}: {p*100:.2f}%")

    if confidence > CONF_THRESHOLD:
        st.success("✅ NÊN CHƠI")
    else:
        st.warning("❌ BỎ QUA")
