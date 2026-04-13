import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
WINDOW = 20
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5-pPONvbU7PR7FteVtEBvN6EuudQ2rgbV3sHX-Ngy1PALF4nvyTBidXOXXE325_TLKKDJwZB7xFgH/pub?output=csv"

# =========================
# LOAD
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
# FEATURE ENGINE
# =========================
def get_features(values):

    # frequency
    freq = [values[-50:].count(i)/50 for i in [1,2,3,4]]

    # streak
    streak = [0]*4
    for i,v in enumerate(reversed(values)):
        if streak[v-1] == 0:
            streak[v-1] = i+1
    streak = [x/max(streak) for x in streak]

    # markov
    m = np.zeros((4,4))
    for i in range(len(values)-1):
        m[values[i]-1][values[i+1]-1] += 1

    row = m[values[-1]-1]
    markov = row/row.sum() if row.sum() else np.ones(4)/4

    # recent pattern (last 5)
    recent = values[-5:]

    return np.concatenate([freq, streak, markov, recent])

# =========================
# SESSION
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "rf" not in st.session_state:
    st.session_state.rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

if "meta" not in st.session_state:
    st.session_state.meta = LogisticRegression(max_iter=300)

if "scaler" not in st.session_state:
    st.session_state.scaler = StandardScaler()

if "probs" not in st.session_state:
    st.session_state.probs = None

# =========================
# UI
# =========================
st.title("🧠 Fantan Smart AI v2 (20k Data Ready)")

col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Data"):
        data = load_data()
        if data:
            st.session_state.data_text = "".join(map(str, data))
            st.rerun()

with col2:
    if st.button("🗑 Reset"):
        st.session_state.probs = None

with st.form("form"):
    st.text_area("DATA (1-4)", key="data_text", height=150)
    st.form_submit_button("Update")

values = parse_data(st.session_state.data_text)

st.write(f"📊 Data: {len(values)}")

# =========================
# RUN
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW + 100:
        st.warning("Cần nhiều data hơn để AI hoạt động tốt")
        st.stop()

    rf = st.session_state.rf
    meta = st.session_state.meta
    scaler = st.session_state.scaler

    X_meta = []
    y_meta = []

    # =========================
    # BUILD DATASET
    # =========================
    for i in range(WINDOW, len(values)-1):

        seq = values[i-WINDOW:i]

        X_rf = []
        y_rf = []

        for j in range(i-WINDOW):
            X_rf.append(values[j:j+WINDOW])
            y_rf.append(values[j+WINDOW]-1)

        if len(X_rf) < 50:
            continue

        rf.fit(X_rf, y_rf)

        ml = rf.predict_proba([seq])[0]
        feat = get_features(values[:i])

        combined = np.concatenate([ml, feat])

        X_meta.append(combined)
        y_meta.append(values[i]-1)

    # =========================
    # TRAIN META
    # =========================
    X_meta = scaler.fit_transform(X_meta)
    meta.fit(X_meta, y_meta)

    # =========================
    # PREDICT
    # =========================
    seq = values[-WINDOW:]

    X_rf = []
    y_rf = []

    for j in range(len(values)-WINDOW):
        X_rf.append(values[j:j+WINDOW])
        y_rf.append(values[j+WINDOW]-1)

    rf.fit(X_rf, y_rf)

    ml = rf.predict_proba([seq])[0]
    feat = get_features(values)

    combined = np.concatenate([ml, feat])
    combined = scaler.transform([combined])

    probs = meta.predict_proba(combined)[0]

    st.session_state.probs = probs

# =========================
# OUTPUT
# =========================
if st.session_state.probs is not None:

    probs = st.session_state.probs

    st.subheader("📊 XÁC SUẤT (SMART AI)")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(str(i+1), f"{probs[i]*100:.1f}%")

    top2 = np.argsort(probs)[-2:][::-1]
    st.success(f"👉 ĐÁNH: {top2[0]+1} + {top2[1]+1}")

    conf = sorted(probs)[-1] - sorted(probs)[-2]

    st.write(f"🔥 Confidence: {conf:.3f}")

    if conf < 0.05:
        st.warning("Kèo yếu - nghỉ")
    elif conf < 0.1:
        st.info("Kèo trung bình")
    else:
        st.success("Kèo mạnh")
