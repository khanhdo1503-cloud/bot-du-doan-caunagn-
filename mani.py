import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
# DATASET (LSTM)
# =========================
def create_dataset(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window] - 1)
    return np.array(X), np.array(y)

def reshape_lstm(X):
    return X.reshape((X.shape[0], X.shape[1], 1))

# =========================
# MODEL
# =========================
def build_model(window):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(4, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# SESSION INIT
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "model" not in st.session_state:
    st.session_state.model = None

if "history" not in st.session_state:
    st.session_state.history = []

if "probs" not in st.session_state:
    st.session_state.probs = None

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
st.title("🧠 Fantan LSTM BOT v2")

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
        st.success("Reset OK")

# INPUT
with st.form("form"):
    st.text_area("DATA (1-4)", key="data_text", height=150)
    st.form_submit_button("Update")

values = parse_data(st.session_state.data_text)
cur_len = len(values)

st.write(f"📊 Data: {cur_len}")

# =========================
# HISTORY CHECK
# =========================
if cur_len < st.session_state.last_len:
    st.session_state.history = [h for h in st.session_state.history if h["len"] <= cur_len]

st.session_state.last_len = cur_len

# =========================
# SHOW LAST 20
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
# RUN LSTM BOT
# =========================
if st.button("🚀 RUN BOT LSTM"):

    if len(values) < WINDOW + 5:
        st.warning("Chưa đủ data cho LSTM")
        st.stop()

    # dataset
    X, y = create_dataset(values, WINDOW)
    X = reshape_lstm(X)

    # build / train model
    model = build_model(WINDOW)
    model.fit(X, y, epochs=15, batch_size=16, verbose=0)

    st.session_state.model = model

    # prediction input
    seq = np.array(values[-WINDOW:]).reshape(1, WINDOW, 1)

    probs = model.predict(seq, verbose=0)[0]
    st.session_state.probs = probs

    top2 = np.argsort(probs)[-2:][::-1]

    st.session_state.history.append({
        "len": len(values),
        "pick": [top2[0]+1, top2[1]+1],
        "result": None
    })

# =========================
# DISPLAY PROBS
# =========================
if st.session_state.probs is not None:
    st.subheader("📊 XÁC SUẤT LSTM")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(str(i+1), f"{st.session_state.probs[i]*100:.1f}%")

    top2 = np.argsort(st.session_state.probs)[-2:][::-1]
    st.success(f"👉 ĐÁNH: {top2[0]+1} + {top2[1]+1}")

# =========================
# WIN / LOSS
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
