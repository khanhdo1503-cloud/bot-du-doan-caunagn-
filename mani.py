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
    try:
        df = pd.read_csv(CSV_URL)
        values = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        values = values.dropna().astype(int).tolist()
        return values
    except:
        return []

# =========================
# INIT SESSION (CỨNG)
# =========================
if "values" not in st.session_state:
    st.session_state.values = []

if "model" not in st.session_state:
    st.session_state.model = None

# luôn đảm bảo values là list
if type(st.session_state.values) != list:
    st.session_state.values = []

values = st.session_state.values

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot (Stable V2)")

# =========================
# LOAD DATA
# =========================
if st.button("🔄 Load Data"):
    data = load_data()

    if isinstance(data, list):
        st.session_state.values = data
        st.success(f"Loaded {len(data)} data")
    else:
        st.error("Load data thất bại")

values = st.session_state.values

# =========================
# CHECK DATA
# =========================
if not values:
    st.warning("👉 Bấm Load Data trước")
    st.stop()

# =========================
# HIỂN THỊ
# =========================
st.subheader("📊 Data")

st.write(f"Tổng: {len(values)}")

st.dataframe(pd.DataFrame(values, columns=["Kết quả"]), height=300)

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
# TRAIN
# =========================
if len(values) > WINDOW:
    if st.button("🧠 Train"):
        X, y = create_dataset(values, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        st.session_state.model = model
        st.success("Train xong")

# =========================
# ADD DATA (SIÊU AN TOÀN)
# =========================
st.subheader("➕ Nhập data")

new_value = st.number_input("1-4", 1, 4)

if st.button("➕ Thêm"):

    if type(st.session_state.values) != list:
        st.session_state.values = []

    st.session_state.values = st.session_state.values + [int(new_value)]

    st.success(f"Đã thêm {new_value}")

# =========================
# PREDICT
# =========================
if st.session_state.model is not None:

    st.subheader("🔮 Dự đoán")

    if len(st.session_state.values) >= WINDOW:

        seq = st.session_state.values[-WINDOW:]
        seq = np.array(seq).reshape(1, -1)

        pred = st.session_state.model.predict_proba(seq)[0]

        choice = int(np.argmax(pred))
        conf = float(np.max(pred))

        st.metric("Kết quả", choice + 1)
        st.metric("Confidence", f"{conf*100:.2f}%")

        if conf > CONF_THRESHOLD:
            st.success("✅ CHƠI")
        else:
            st.warning("❌ BỎ")

    else:
        st.warning("Chưa đủ data")
