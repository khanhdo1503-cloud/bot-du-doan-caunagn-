import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

    row = m[values[-1]-1]
    if row.sum() == 0:
        return [0.25]*4
    return (row/row.sum()).tolist()

# =========================
# SESSION
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "rf_model" not in st.session_state:
    st.session_state.rf_model = RandomForestClassifier(n_estimators=150)

if "meta_model" not in st.session_state:
    st.session_state.meta_model = LogisticRegression(max_iter=200)

if "history" not in st.session_state:
    st.session_state.history = []

if "probs" not in st.session_state:
    st.session_state.probs = None

# =========================
# UI
# =========================
st.title("🧠 Fantan META AI BOT")

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
        st.success("Reset OK")

with st.form("form"):
    st.text_area("DATA (1-4)", key="data_text", height=150)
    st.form_submit_button("Update")

values = parse_data(st.session_state.data_text)

st.write(f"📊 Data: {len(values)}")

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW + 10:
        st.warning("Chưa đủ data")
        st.stop()

    rf = st.session_state.rf_model
    meta = st.session_state.meta_model

    meta_X = []
    meta_y = []

    # =========================
    # BUILD META DATASET
    # =========================
    for i in range(WINDOW, len(values)-1):

        seq = values[i-WINDOW:i]

        X_tmp, y_tmp = [], []
        for j in range(i-WINDOW):
            X_tmp.append(values[j:j+WINDOW])
            y_tmp.append(values[j+WINDOW]-1)

        if len(X_tmp) < 10:
            continue

        rf.fit(X_tmp, y_tmp)
        ml = rf.predict_proba(np.array(seq).reshape(1,-1))[0]

        f1 = freq(values[:i])
        f2 = streak(values[:i])
        f3 = markov(values[:i])

        features = list(ml) + f1 + f2 + f3

        meta_X.append(features)
        meta_y.append(values[i]-1)

    # =========================
    # TRAIN META MODEL
    # =========================
    if len(meta_X) < 30:
        st.warning("Data chưa đủ để train AI")
        st.stop()

    meta.fit(meta_X, meta_y)

    # =========================
    # PREDICT
    # =========================
    seq = values[-WINDOW:]

    X_tmp, y_tmp = [], []
    for j in range(len(values)-WINDOW):
        X_tmp.append(values[j:j+WINDOW])
        y_tmp.append(values[j+WINDOW]-1)

    rf.fit(X_tmp, y_tmp)

    ml = rf.predict_proba(np.array(seq).reshape(1,-1))[0]
    f1 = freq(values)
    f2 = streak(values)
    f3 = markov(values)

    features = list(ml) + f1 + f2 + f3

    final = meta.predict_proba([features])[0]

    st.session_state.probs = final

# =========================
# OUTPUT
# =========================
if st.session_state.probs is not None:

    probs = st.session_state.probs

    st.subheader("📊 XÁC SUẤT (META AI)")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(str(i+1), f"{probs[i]*100:.1f}%")

    top2 = np.argsort(probs)[-2:][::-1]
    st.success(f"👉 ĐÁNH: {top2[0]+1} + {top2[1]+1}")

    # CONFIDENCE
    conf = sorted(probs)[-1] - sorted(probs)[-2]
    st.write(f"🔥 Confidence: {conf:.3f}")

    if conf < 0.05:
        st.warning("Kèo yếu - nghỉ")
    elif conf < 0.1:
        st.info("Kèo trung bình")
    else:
        st.success("Kèo mạnh")
