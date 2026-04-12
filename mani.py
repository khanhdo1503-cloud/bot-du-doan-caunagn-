import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

WINDOW = 10
RETRAIN_THRESHOLD = 30

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
# STATS
# =========================
def calc_frequency(values):
    freq = [0,0,0,0]
    for v in values[-50:]:
        freq[v-1] += 1
    total = sum(freq)
    return [f/total if total>0 else 0 for f in freq]

def calc_streak(values):
    last_seen = [0,0,0,0]
    for i, v in enumerate(reversed(values)):
        if last_seen[v-1] == 0:
            last_seen[v-1] = i+1
    max_gap = max(last_seen) if max(last_seen)>0 else 1
    return [x/max_gap for x in last_seen]

# =========================
# INIT
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "model" not in st.session_state:
    st.session_state.model = None

if "new_count" not in st.session_state:
    st.session_state.new_count = 0

if "probs" not in st.session_state:
    st.session_state.probs = None

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot (Clean UI)")

if st.button("☁️ Load Data"):
    data = load_data()
    st.session_state.data_text = "".join(str(x) for x in data)

data_text = st.text_area("📥 DATA", value=st.session_state.data_text, height=150)
st.session_state.data_text = data_text

values = parse_data(data_text)

st.write(f"📊 Tổng: {len(values)} | 🔄 Chưa học: {st.session_state.new_count}")

# =========================
# HIỂN THỊ 20 VÁN NGANG
# =========================
st.subheader("📋 20 VÁN GẦN NHẤT")

last20 = values[-20:]

color_map = {
    1: "#ff4b4b",
    2: "#4b7bff",
    3: "#2ecc71",
    4: "#f1c40f"
}

html = "<div style='display:flex;gap:6px;flex-wrap:wrap'>"

for v in last20:
    color = color_map.get(v, "#ccc")
    html += f"""
    <div style='
        width:35px;
        height:35px;
        background:{color};
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:6px;
        font-weight:bold;
    '>{v}</div>
    """

html += "</div>"

st.markdown(html, unsafe_allow_html=True)

# =========================
# TRAIN
# =========================
def train_model(values):
    X, y = create_dataset(values, WINDOW)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    return model

# =========================
# RUN
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW:
        st.warning("Chưa đủ data")
    else:
        if st.session_state.model is None or st.session_state.new_count >= RETRAIN_THRESHOLD:
            st.session_state.model = train_model(values)
            st.session_state.new_count = 0
            st.success("🧠 Đã học lại")

        seq = np.array(values[-WINDOW:]).reshape(1, -1)
        ml_pred = st.session_state.model.predict_proba(seq)[0]

        freq = calc_frequency(values)
        streak = calc_streak(values)

        final = []
        for i in range(4):
            score = 0.5*ml_pred[i] + 0.3*freq[i] + 0.2*streak[i]
            final.append(score)

        final = np.array(final)
        final = final / np.sum(final)

        st.session_state.probs = final

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

    st.subheader("🔮 GỢI Ý")
    st.success(f"👉 {top2[0]+1} + {top2[1]+1}")

# =========================
# ADD
# =========================
st.subheader("➕ THÊM NHANH")

new_input = st.text_input("Nhập: 1234")

if st.button("➕ ADD"):
    new_vals = parse_data(new_input)
    if new_vals:
        st.session_state.data_text += "".join(str(x) for x in new_vals)
        st.session_state.new_count += len(new_vals)
        st.success(f"+{len(new_vals)}")
    else:
        st.error("Sai input")
