import streamlit as st
import pandas as pd
import numpy as np
import re

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

def parse_data(text):
    return list(map(int, re.findall(r"[1-4]", text)))

# =========================
# FEATURES (MARKOV + STATS)
# =========================
def calc_frequency(values):
    recent = values[-50:]
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

def calc_markov(values):
    matrix = np.zeros((4,4))
    for i in range(len(values)-1):
        matrix[values[i]-1][values[i+1]-1] += 1

    probs = []
    last = values[-1] - 1

    row = matrix[last]
    total = np.sum(row)

    if total == 0:
        return [0.25]*4

    probs = row / total
    return probs.tolist()

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
# SESSION STATE
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "model" not in st.session_state:
    st.session_state.model = None

if "probs" not in st.session_state:
    st.session_state.probs = None

if "history" not in st.session_state:
    st.session_state.history = []

if "last_len" not in st.session_state:
    st.session_state.last_len = 0

# =========================
# FIX HISTORY
# =========================
for h in st.session_state.history:
    if "result" not in h:
        h["result"] = None

# =========================
# UI
# =========================
st.title("🧠 Fantan BOT Stable (No TensorFlow)")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Data"):
        data = load_data()
        if data:
            st.session_state.data_text = "".join(map(str, data))
            st.rerun()

with col2:
    if st.button("🗑 Reset"):
        st.session_state.history = []
        st.session_state.model = None
        st.session_state.probs = None
        st.success("Reset OK")

# INPUT
with st.form("form"):
    st.text_area("DATA (1-4)", key="data_text", height=150)
    st.form_submit_button("Update")

values = parse_data(st.session_state.data_text)
cur_len = len(values)

st.write(f"📊 Data: {cur_len}")

# =========================
# LAST 20 DISPLAY
# =========================
st.subheader("📋 20 ván gần nhất")

color_map = {
    1:"#ff4b4b",
    2:"#4b7bff",
    3:"#2ecc71",
    4:"#f1c40f"
}

st.markdown(
    "<div style='display:flex;gap:6px;flex-wrap:wrap'>" +
    "".join([
        f"<div style='width:35px;height:35px;background:{color_map[v]};color:white;display:flex;align-items:center;justify-content:center;border-radius:6px'>{v}</div>"
        for v in values[-20:]
    ]) +
    "</div>",
    unsafe_allow_html=True
)

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW + 5:
        st.warning("Chưa đủ data")
        st.stop()

    # dataset
    X, y = create_dataset(values, WINDOW)

    # model init
    if st.session_state.model is None:
        st.session_state.model = RandomForestClassifier(
            n_estimators=250,
            random_state=42
        )

    model = st.session_state.model
    model.fit(X, y)

    # prediction
    seq = np.array(values[-WINDOW:]).reshape(1, -1)
    ml_probs = model.predict_proba(seq)[0]

    # features
    freq = calc_frequency(values)
    streak = calc_streak(values)
    markov = calc_markov(values)

    # ensemble (NON-ML SAFE)
    final = []
    for i in range(4):
        score = (
            0.45 * ml_probs[i] +
            0.25 * freq[i] +
            0.15 * streak[i] +
            0.15 * markov[i]
        )
        final.append(score)

    final = np.array(final)
    final = final / np.sum(final)

    st.session_state.probs = final

    top2 = np.argsort(final)[-2:][::-1]

    st.session_state.history.append({
        "len": len(values),
        "pick": [top2[0]+1, top2[1]+1],
        "result": None
    })

# =========================
# DISPLAY PROBS
# =========================
if st.session_state.probs is not None:

    st.subheader("📊 XÁC SUẤT")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(str(i+1), f"{st.session_state.probs[i]*100:.1f}%")

    top2 = np.argsort(st.session_state.probs)[-2:][::-1]
    st.success(f"👉 ĐÁNH: {top2[0]+1} + {top2[1]+1}")

# =========================
# WIN LOSS
# =========================
win = 0
loss = 0

for h in st.session_state.history:
    if h["result"] is not None:
        if h["result"] in h["pick"]:
            win += 1
        else:
            loss += 1

total = win + loss
rate = (win / total * 100) if total > 0 else 0

st.markdown("---")
st.subheader("📊 HIỆU SUẤT")

if total == 0:
    st.info("Chưa có dữ liệu win/loss")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Win", win)
    c3.metric("Loss", loss)
    c4.metric("Winrate", f"{rate:.1f}%")
