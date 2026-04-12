import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

WINDOW = 7
CONF_THRESHOLD = 0.55

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5-pPONvbU7PR7FteVtEBvN6EuudQ2rgbV3sHX-Ngy1PALF4nvyTBidXOXXE325_TLKKDJwZB7xFgH/pub?output=csv"

# =========================
# SAFE DATA LAYER (QUAN TRỌNG NHẤT)
# =========================
def get_values():
    raw = st.session_state.get("values", [])

    if not isinstance(raw, list):
        return []

    clean = []
    for x in raw:
        try:
            x = int(x)
            if x in [1,2,3,4]:
                clean.append(x)
        except:
            pass

    return clean

def set_values(v):
    if isinstance(v, list):
        st.session_state.values = v
    else:
        st.session_state.values = []

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
# UI
# =========================
st.title("🧠 Fantan Bot UI Stable")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Google Sheet"):
        set_values(load_data())
        st.success("Loaded data")

with col2:
    if st.button("♻️ Reset Data"):
        set_values([])
        st.success("Reset xong")

values = get_values()

# =========================
# INPUT
# =========================
st.subheader("📥 DATA INPUT")

data_input = st.text_area(
    "Dán chuỗi số (ví dụ: 123412341234...)",
    height=200
)

# =========================
# PARSE
# =========================
def parse_input(text):
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
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    new_values = parse_input(data_input)

    if not new_values:
        st.error("❌ Data không hợp lệ")
        st.stop()

    v = get_values()
    v.extend(new_values)
    set_values(v)

    st.success(f"Đã thêm {len(new_values)} giá trị")

    # TRAIN + PREDICT
    if len(v) > WINDOW:

        X, y = create_dataset(v, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        st.session_state.model = model

        seq = np.array(v[-WINDOW:]).reshape(1, -1)
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
        st.warning("Chưa đủ data")

# =========================
# INFO (KHÔNG BAO GIỜ CRASH)
# =========================
st.divider()

safe_values = get_values()
st.write(f"📊 Tổng data: {len(safe_values)}")
