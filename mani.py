import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

WINDOW = 7
CONF_THRESHOLD = 0.55

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5-pPONvbU7PR7FteVtEBvN6EuudQ2rgbV3sHX-Ngy1PALF4nvyTBidXOXXE325_TLKKDJwZB7xFgH/pub?output=csv"

# =========================
# LOAD DATA (NO CACHE)
# =========================
def load_data():
    try:
        df = pd.read_csv(CSV_URL)

        values = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        values = values.dropna().astype(int).tolist()

        return values if isinstance(values, list) else []

    except Exception as e:
        st.error("❌ Lỗi load Google Sheet")
        st.write(e)
        return []

# =========================
# INIT SESSION
# =========================
if "values" not in st.session_state:
    st.session_state.values = []

if "model" not in st.session_state:
    st.session_state.model = None

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot Cloud")

# LOAD BUTTON (QUAN TRỌNG)
if st.button("🔄 Load Data"):
    st.session_state.values = load_data()

values = st.session_state.get("values", [])

# CHECK
if not isinstance(values, list):
    values = []

# =========================
# SHOW DATA
# =========================
if values:
    st.success(f"Loaded {len(values)} data")

    st.dataframe(pd.DataFrame(values, columns=["Kết quả"]), height=300)

else:
    st.warning("👉 Bấm 'Load Data' để lấy dữ liệu")
    st.stop()

# =========================
# CREATE DATASET
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

    if st.button("🧠 Train Model"):
        X, y = create_dataset(values, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        st.session_state.model = model
        st.success("Model trained ✅")

# =========================
# ADD DATA
# =========================
st.subheader("➕ Nhập kết quả mới")

new_value = st.number_input("Nhập (1-4)", 1, 4)

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

        if confidence > CONF_THRESHOLD:
            st.success("✅ NÊN CHƠI")
        else:
            st.warning("❌ BỎ QUA")

    else:
        st.warning("⚠️ Chưa đủ data")
