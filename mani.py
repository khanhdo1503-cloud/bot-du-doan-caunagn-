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
        return "".join(str(x) for x in values)
    except:
        return ""

# =========================
# PARSE
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

if "result" not in st.session_state:
    st.session_state.result = None

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot (RUN là ra kết quả)")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Google Sheet"):
        st.session_state.data_text = load_data()

with col2:
    if st.button("♻️ Reset"):
        st.session_state.data_text = ""
        st.session_state.result = None

# =========================
# INPUT
# =========================
st.subheader("📥 DATA")

data_text = st.text_area(
    "Chuỗi data",
    value=st.session_state.data_text,
    height=200
)

st.session_state.data_text = data_text

values = parse_data(data_text)

st.write(f"📊 Tổng data: {len(values)}")

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW:
        st.session_state.result = "NOT_ENOUGH"
    else:
        X, y = create_dataset(values, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        seq = np.array(values[-WINDOW:]).reshape(1, -1)
        pred = model.predict_proba(seq)[0]

        choice = int(np.argmax(pred))
        conf = float(np.max(pred))

        st.session_state.result = (choice, conf, pred)

# =========================
# HIỂN THỊ RESULT (QUAN TRỌNG)
# =========================
if st.session_state.result is not None:

    st.subheader("🔮 RESULT")

    if st.session_state.result == "NOT_ENOUGH":
        st.warning("⚠️ Chưa đủ data")
    else:
        choice, conf, pred = st.session_state.result

        st.metric("Dự đoán", choice + 1)
        st.metric("Confidence", f"{conf*100:.2f}%")

        st.write("### 📊 Xác suất chi tiết")
        for i, p in enumerate(pred):
            st.write(f"{i+1}: {p*100:.2f}%")

        if conf > CONF_THRESHOLD:
            st.success("✅ CHƠI")
        else:
            st.warning("❌ BỎ")
