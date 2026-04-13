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
        col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        return col.dropna().astype(int).tolist()
    except:
        return []

def parse_data(text):
    if not text:
        return []
    return list(map(int, re.findall(r"[1-4]", str(text))))

# =========================
# FEATURES
# =========================
def freq(values):
    recent = values[-50:]
    f = [recent.count(i) for i in [1,2,3,4]]
    s = sum(f)
    return [x/s if s else 0 for x in f]

def streak(values):
    last = [0,0,0,0]
    for i,v in enumerate(reversed(values)):
        if last[v-1] == 0:
            last[v-1] = i+1
    m = max(last) if max(last) else 1
    return [x/m for x in last]

def markov(values):
    m = np.zeros((4,4))
    for i in range(len(values)-1):
        m[values[i]-1][values[i+1]-1] += 1

    last = values[-1]-1
    row = m[last]

    if row.sum() == 0:
        return [0.25]*4

    return (row/row.sum()).tolist()

# =========================
# SESSION INIT
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "model" not in st.session_state:
    st.session_state.model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

if "probs" not in st.session_state:
    st.session_state.probs = None

if "history" not in st.session_state:
    st.session_state.history = []

# 🔥 FIX HISTORY CŨ (QUAN TRỌNG)
for h in st.session_state.history:
    h.setdefault("len", 0)
    h.setdefault("result", None)
    h.setdefault("pick", [])

# =========================
# UI
# =========================
st.title("🧠 Fantan BOT FINAL (No Crash Version)")

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
# UPDATE RESULT AUTO (KHÔNG CRASH)
# =========================
if len(st.session_state.history) > 0:
    last = st.session_state.history[-1]

    if last["result"] is None and cur_len > last["len"]:
        idx = last["len"]
        if idx < len(values):
            last["result"] = values[idx]

# =========================
# LAST 20
# =========================
st.subheader("📋 20 ván gần nhất")

color = {1:"#ff4b4b",2:"#4b7bff",3:"#2ecc71",4:"#f1c40f"}

st.markdown(
    "<div style='display:flex;gap:6px;flex-wrap:wrap'>" +
    "".join([
        f"<div style='width:35px;height:35px;background:{color[v]};color:white;display:flex;align-items:center;justify-content:center;border-radius:6px'>{v}</div>"
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
    X, y = [], []
    for i in range(len(values)-WINDOW):
        X.append(values[i:i+WINDOW])
        y.append(values[i+WINDOW]-1)

    X = np.array(X)
    y = np.array(y)

    model = st.session_state.model
    model.fit(X, y)

    seq = np.array(values[-WINDOW:]).reshape(1,-1)
    ml = model.predict_proba(seq)[0]

    f1 = freq(values)
    f2 = streak(values)
    f3 = markov(values)

    final = []
    for i in range(4):
        final.append(
            0.5*ml[i] +
            0.2*f1[i] +
            0.15*f2[i] +
            0.15*f3[i]
        )

    final = np.array(final)
    final = final / final.sum()

    st.session_state.probs = final

    top2 = np.argsort(final)[-2:][::-1]

    st.session_state.history.append({
        "len": len(values),
        "pick": [top2[0]+1, top2[1]+1],
        "result": None
    })

# =========================
# OUTPUT
# =========================
if st.session_state.probs is not None:

    st.subheader("📊 XÁC SUẤT")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(str(i+1), f"{st.session_state.probs[i]*100:.1f}%")

    t = np.argsort(st.session_state.probs)[-2:][::-1]
    st.success(f"👉 ĐÁNH: {t[0]+1} + {t[1]+1}")

# =========================
# HIỆU SUẤT
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
    st.info("Chưa có dữ liệu (RUN → thêm data → RUN)")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Win", win)
    c3.metric("Loss", loss)
    c4.metric("Winrate", f"{rate:.1f}%")
