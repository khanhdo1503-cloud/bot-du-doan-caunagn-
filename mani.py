import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

WINDOW = 10

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5-pPONvbU7PR7FteVtEBvN6EuudQ2rgbV3sHX-Ngy1PALF4nvyTBidXOXXE325_TLKKDJwZB7xFgH/pub?output=csv"

# =========================
# LOAD
# =========================
def load_data():
    try:
        df = pd.read_csv(CSV_URL)
        col = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        return col.dropna().astype(int).tolist()
    except:
        return []

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

if "probs" not in st.session_state:
    st.session_state.probs = None

if "history" not in st.session_state:
    st.session_state.history = []  # lưu dự đoán

if "last_len" not in st.session_state:
    st.session_state.last_len = 0

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot + Tracking")

if st.button("☁️ Load Data"):
    data = load_data()
    st.session_state.data_text = "".join(str(x) for x in data)

data_text = st.text_area("📥 DATA", value=st.session_state.data_text, height=150)
st.session_state.data_text = data_text

values = parse_data(data_text)
cur_len = len(values)

st.write(f"📊 Tổng data: {cur_len}")

# =========================
# HANDLE DELETE DATA
# =========================
if cur_len < st.session_state.last_len:
    # xóa history thừa
    st.session_state.history = [
        h for h in st.session_state.history if h["len"] <= cur_len
    ]

st.session_state.last_len = cur_len

# =========================
# UI 20 VÁN
# =========================
st.subheader("📋 20 VÁN GẦN NHẤT")

color_map = {1:"#ff4b4b",2:"#4b7bff",3:"#2ecc71",4:"#f1c40f"}

boxes = "".join([
    f"<div style='width:35px;height:35px;background:{color_map[v]};color:white;display:flex;align-items:center;justify-content:center;border-radius:6px;font-weight:bold'>{v}</div>"
    for v in values[-20:]
])

st.markdown(f"<div style='display:flex;gap:6px'>{boxes}</div>", unsafe_allow_html=True)

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW:
        st.warning("❌ Chưa đủ data")
        st.stop()

    X, y = create_dataset(values, WINDOW)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    seq = np.array(values[-WINDOW:]).reshape(1, -1)
    probs = model.predict_proba(seq)[0]

    st.session_state.probs = probs

    # lưu prediction
    top2 = np.argsort(probs)[-2:][::-1]
    st.session_state.history.append({
        "len": cur_len,
        "pick": [top2[0]+1, top2[1]+1]
    })

# =========================
# RESULT
# =========================
if st.session_state.probs is not None:

    probs = st.session_state.probs

    st.subheader("📊 XÁC SUẤT")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1", f"{probs[0]*100:.1f}%")
    col2.metric("2", f"{probs[1]*100:.1f}%")
    col3.metric("3", f"{probs[2]*100:.1f}%")
    col4.metric("4", f"{probs[3]*100:.1f}%")

    top2 = np.argsort(probs)[-2:][::-1]
    st.success(f"👉 {top2[0]+1} + {top2[1]+1}")

# =========================
# TÍNH WIN / LOSS
# =========================
win = 0
loss = 0

for h in st.session_state.history:
    if h["len"] < len(values):
        actual = values[h["len"]]  # ván tiếp theo
        if actual in h["pick"]:
            win += 1
        else:
            loss += 1

total = win + loss
rate = (win / total * 100) if total > 0 else 0

# =========================
# UI BỘ ĐẾM
# =========================
st.markdown("---")
st.subheader("📊 BỘ ĐẾM")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Tổng ván", total)
c2.metric("Thắng", win)
c3.metric("Thua", loss)
c4.metric("Winrate", f"{rate:.1f}%")
