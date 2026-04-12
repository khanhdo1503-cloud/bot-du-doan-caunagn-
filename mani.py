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
# SAFE VALUES
# =========================
def get_values():
    v = st.session_state.get("values", [])
    if not isinstance(v, list):
        return []
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
st.title("🧠 Fantan Bot AUTO")

# LOAD
if st.button("🔄 Load Data"):
    st.session_state.values = load_data()
    st.success(f"Loaded {len(st.session_state.values)} data")

values = get_values()

if len(values) == 0:
    st.warning("👉 Bấm Load Data trước")
    st.stop()

# =========================
# HIỂN THỊ
# =========================
st.subheader("📊 Data")

st.write(f"Tổng: {len(values)}")
st.dataframe(pd.DataFrame(values, columns=["Kết quả"]), height=250)

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
# NHẬP NHIỀU DATA
# =========================
st.subheader("➕ Nhập nhiều kết quả (cách nhau bằng dấu cách)")

multi_input = st.text_input("Ví dụ: 1 2 4 3 1 1 4")

if st.button("🚀 Cập nhật & Dự đoán"):

    # tách chuỗi
    try:
        new_values = [int(x) for x in multi_input.strip().split()]
        new_values = [x for x in new_values if x in [1,2,3,4]]
    except:
        new_values = []

    if len(new_values) == 0:
        st.error("❌ Input không hợp lệ")
        st.stop()

    # cập nhật data
    v = get_values()
    v.extend(new_values)
    st.session_state.values = v

    st.success(f"Đã thêm {len(new_values)} giá trị")

    # =========================
    # AUTO TRAIN
    # =========================
    if len(v) > WINDOW:
        X, y = create_dataset(v, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        st.session_state.model = model

        # =========================
        # AUTO PREDICT
        # =========================
        seq = np.array(v[-WINDOW:]).reshape(1, -1)
        pred = model.predict_proba(seq)[0]

        choice = int(np.argmax(pred))
        conf = float(np.max(pred))

        st.subheader("🔮 Dự đoán ngay")

        st.metric("Kết quả", choice + 1)
        st.metric("Confidence", f"{conf*100:.2f}%")

        if conf > CONF_THRESHOLD:
            st.success("✅ CHƠI")
        else:
            st.warning("❌ BỎ")

    else:
        st.warning("Chưa đủ data để train")
