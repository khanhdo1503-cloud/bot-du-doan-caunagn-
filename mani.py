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
        values = col.dropna().astype(int).tolist()
        return "".join(str(x) for x in values)  # convert thành chuỗi
    except:
        return ""

# =========================
# PARSE DATA
# =========================
def parse_data(text):
    return [int(c) for c in text if c in "1234"]

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
# INIT
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot - Text Mode")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Google Sheet"):
        st.session_state.data_text = load_data()
        st.success("Đã load data")

with col2:
    if st.button("♻️ Reset"):
        st.session_state.data_text = ""
        st.success("Đã reset")

# =========================
# DATA INPUT (CHÍNH)
# =========================
st.subheader("📥 DATA (sửa trực tiếp tại đây)")

data_text = st.text_area(
    "Chuỗi data (ví dụ: 123412341234...)",
    value=st.session_state.data_text,
    height=250
)

st.session_state.data_text = data_text

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    values = parse_data(data_text)

    if len(values) == 0:
        st.error("❌ Data không hợp lệ")
        st.stop()

    st.success(f"Data hợp lệ: {len(values)} số")

    if len(values) > WINDOW:

        X, y = create_dataset(values, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        seq = np.array(values[-WINDOW:]).reshape(1, -1)
        pred = model.predict_proba(seq)[0]

        choice = int(np.argmax(pred))
        conf = float(np.max(pred))

        st.subheader("🔮 RESULT")

        st.metric("Dự đoán", choice + 1)
        st.metric("Confidence", f"{conf*100:.2f}%")

        if conf > CONF_THRESHOLD:
            st.success("✅ CHƠI")
        else:
            st.warning("❌ BỎ")

    else:
        st.warning("⚠️ Chưa đủ data")

# =========================
# INFO
# =========================
values = parse_data(data_text)
st.write(f"📊 Tổng data: {len(values)}")
