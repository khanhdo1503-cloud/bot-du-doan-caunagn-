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
# SAFE GET VALUES (QUAN TRỌNG NHẤT)
# =========================
def get_values():
    v = st.session_state.get("values", [])

    # ép sạch tuyệt đối
    if not isinstance(v, list):
        return []

    # loại bỏ phần tử lỗi nếu có
    clean = []
    for x in v:
        try:
            clean.append(int(x))
        except:
            pass

    return clean

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
st.title("🧠 Fantan Bot FINAL")

# LOAD
if st.button("🔄 Load Data"):
    data = load_data()
    st.session_state.values = data if isinstance(data, list) else []
    st.success(f"Loaded {len(st.session_state.values)} data")

# LUÔN dùng get_values
values = get_values()

# =========================
# CHECK
# =========================
if len(values) == 0:
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
# ADD DATA
# =========================
st.subheader("➕ Nhập data")

new_value = st.number_input("1-4", 1, 4)

if st.button("➕ Thêm"):
    v = get_values()
    v.append(int(new_value))
    st.session_state.values = v
    st.success(f"Đã thêm {new_value}")

# =========================
# PREDICT
# =========================
if st.session_state.model is not None:

    st.subheader("🔮 Dự đoán")

    v = get_values()

    if len(v) >= WINDOW:

        seq = np.array(v[-WINDOW:]).reshape(1, -1)
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
