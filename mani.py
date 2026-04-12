import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================
WINDOW = 15
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
# NÃO PHỤ
# =========================
def calc_frequency(values, n=50):
    recent = values[-n:]
    freq = [recent.count(i) for i in [1,2,3,4]]
    total = sum(freq)
    return [f/total if total > 0 else 0 for f in freq]

def calc_streak(values):
    last_seen = [0,0,0,0]
    for i,v in enumerate(reversed(values)):
        if last_seen[v-1] == 0:
            last_seen[v-1] = i+1
    m = max(last_seen) if max(last_seen)>0 else 1
    return [x/m for x in last_seen]

def calc_prior(values):
    freq = [values.count(i) for i in [1,2,3,4]]
    total = sum(freq)
    return [f/total if total > 0 else 0 for f in freq]

# =========================
# INIT SESSION
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "probs" not in st.session_state:
    st.session_state.probs = None

if "history" not in st.session_state:
    st.session_state.history = []

if "last_len" not in st.session_state:
    st.session_state.last_len = 0

# =========================
# FIX HISTORY CŨ
# =========================
for h in st.session_state.history:
    if "result" not in h:
        h["result"] = None

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot FINAL")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Data từ Sheet"):
        data = load_data()
        if len(data) > 0:
            st.session_state.data_text = "".join(str(x) for x in data)
            st.rerun()

with col2:
    if st.button("🗑 Reset Win/Loss"):
        st.session_state.history = []
        st.success("Đã reset")

# =========================
# INPUT (KHÔNG RESET)
# =========================
with st.form("form"):
    st.text_area("📥 DATA (1-4)", key="data_text", height=150)
    st.form_submit_button("💾 Cập nhật")

values = parse_data(st.session_state.data_text)
cur_len = len(values)

st.write(f"📊 Tổng data: {cur_len}")

# =========================
# HANDLE DELETE DATA
# =========================
if cur_len < st.session_state.last_len:
    st.session_state.history = [
        h for h in st.session_state.history if h["len"] <= cur_len
    ]

st.session_state.last_len = cur_len

# =========================
# HIỂN THỊ 20 VÁN
# =========================
st.subheader("📋 20 VÁN GẦN NHẤT")

color_map = {
    1:"#ff4b4b",
    2:"#4b7bff",
    3:"#2ecc71",
    4:"#f1c40f"
}

boxes = "".join([
    f"<div style='width:35px;height:35px;background:{color_map[v]};color:white;display:flex;align-items:center;justify-content:center;border-radius:6px;font-weight:bold'>{v}</div>"
    for v in values[-20:]
])

st.markdown(f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{boxes}</div>", unsafe_allow_html=True)

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW:
        st.warning("❌ Chưa đủ data")
        st.stop()

    # 🔥 CHECK prediction cũ
    if len(st.session_state.history) > 0:
        last = st.session_state.history[-1]

        if last["len"] < len(values):
            actual = values[last["len"]]
            last["result"] = actual

    # 🔥 TRAIN MODEL
    X, y = create_dataset(values, WINDOW)
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X, y)

    seq = np.array(values[-WINDOW:]).reshape(1, -1)
    ml = model.predict_proba(seq)[0]

    # 🔥 NÃO PHỤ
    freq = calc_frequency(values)
    streak = calc_streak(values)
    prior = calc_prior(values)

    # 🔥 TRỘN NÃO
    final = []
    for i in range(4):
        score = (
            0.4 * ml[i] +
            0.25 * freq[i] +
            0.2 * streak[i] +
            0.15 * prior[i]
        )
        final.append(score)

    final = np.array(final)
    final = final / np.sum(final)

    st.session_state.probs = final

    top2 = np.argsort(final)[-2:][::-1]

    # 🔥 SAVE prediction
    st.session_state.history.append({
        "len": len(values),
        "pick": [top2[0]+1, top2[1]+1],
        "result": None
    })

# =========================
# HIỂN THỊ KẾT QUẢ
# =========================
if st.session_state.probs is not None:

    probs = st.session_state.probs

    st.subheader("📊 XÁC SUẤT")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(str(i+1), f"{probs[i]*100:.1f}%")

    top2 = np.argsort(probs)[-2:][::-1]
    st.success(f"👉 ĐÁNH: {top2[0]+1} + {top2[1]+1}")

# =========================
# WIN / LOSS
# =========================
win = 0
loss = 0

for h in st.session_state.history:
    result = h.get("result")
    if result is not None:
        if result in h["pick"]:
            win += 1
        else:
            loss += 1

total = win + loss
rate = (win / total * 100) if total > 0 else 0

# =========================
# UI HIỆU SUẤT
# =========================
st.markdown("---")
st.subheader("📊 HIỆU SUẤT")

if total == 0:
    st.info("Chưa có ván nào (RUN → thêm data → RUN)")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng", total)
    c2.metric("Win", win)
    c3.metric("Loss", loss)
    c4.metric("Winrate", f"{rate:.1f}%")
