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
# UI HEADER
# =========================
st.title("🧠 Fantan Bot - Editable Table")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Google Sheet"):
        st.session_state.values = load_data()
        st.success("Loaded data")

with col2:
    if st.button("♻️ Reset"):
        st.session_state.values = []
        st.success("Reset xong")

# =========================
# DATA TABLE (EDIT TRỰC TIẾP)
# =========================
st.subheader("📋 DATA (Có thể sửa trực tiếp)")

values = st.session_state.values

# đảm bảo luôn là list
if not isinstance(values, list):
    values = []

df = pd.DataFrame(values, columns=["Kết quả"])

edited_df = st.data_editor(
    df,
    num_rows="dynamic",  # cho phép thêm dòng
    use_container_width=True
)

# =========================
# LẤY DATA SAU KHI EDIT
# =========================
def get_clean_values(df):
    result = []
    for x in df["Kết quả"]:
        try:
            x = int(x)
            if x in [1,2,3,4]:
                result.append(x)
        except:
            pass
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

    clean_values = get_clean_values(edited_df)

    if len(clean_values) == 0:
        st.error("❌ Data không hợp lệ")
        st.stop()

    st.session_state.values = clean_values

    st.success(f"Data hợp lệ: {len(clean_values)} dòng")

    if len(clean_values) > WINDOW:

        X, y = create_dataset(clean_values, WINDOW)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        st.session_state.model = model

        seq = np.array(clean_values[-WINDOW:]).reshape(1, -1)
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
# THỐNG KÊ
# =========================
st.divider()
st.write(f"📊 Tổng data: {len(st.session_state.values)}")
