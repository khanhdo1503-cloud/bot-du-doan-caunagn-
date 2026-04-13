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
def get_features(values):
    freq = [values[-50:].count(i)/50 for i in [1,2,3,4]]

    streak = [0]*4
    for i,v in enumerate(reversed(values)):
        if streak[v-1] == 0:
            streak[v-1] = i+1
    streak = [x/max(streak) for x in streak]

    m = np.zeros((4,4))
    for i in range(len(values)-1):
        m[values[i]-1][values[i+1]-1] += 1

    row = m[values[-1]-1]
    markov = row/row.sum() if row.sum() else np.ones(4)/4

    recent = values[-5:]

    return np.concatenate([freq, streak, markov, recent])

# =========================
# SESSION
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "trained" not in st.session_state:
    st.session_state.trained = False

if "rf" not in st.session_state:
    st.session_state.rf = None

if "meta" not in st.session_state:
    st.session_state.meta = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "probs" not in st.session_state:
    st.session_state.probs = None

if "ml_probs" not in st.session_state:
    st.session_state.ml_probs = None

# =========================
# AUTO LOAD DATA
# =========================
if st.session_state.data_text == "":
    data = load_data()
    if data:
        st.session_state.data_text = "".join(map(str, data))

# =========================
# UI
# =========================
st.title("🧠 Fantan AI PRO (FAST MODE)")

# ===== SETTINGS =====
st.subheader("⚙️ Cài đặt")

WINDOW = st.slider("Window Size", 5, 50, 20)

# ===== BUTTONS =====
col1, col2 = st.columns(2)

with col1:
    if st.button("☁️ Load Data"):
        data = load_data()
        if data:
            st.session_state.data_text = "".join(map(str, data))
            st.success("Loaded!")
            st.rerun()

with col2:
    if st.button("🗑 Reset"):
        st.session_state.data_text = ""
        st.session_state.trained = False
        st.success("Reset!")

# ===== INPUT =====
with st.form("form"):
    st.text_area("DATA (1-4)", key="data_text", height=150)
    st.form_submit_button("Update")

values = parse_data(st.session_state.data_text)
st.write(f"📊 Data: {len(values)}")

# =========================
# TRAIN AI
# =========================
if st.button("🧠 TRAIN AI (thủ công)"):

    if len(values) < WINDOW + 100:
        st.warning("Thiếu data để train")
        st.stop()

    rf = RandomForestClassifier(n_estimators=150, max_depth=10)
    meta = LogisticRegression(max_iter=300)
    scaler = StandardScaler()

    X_meta = []
    y_meta = []

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

        X_meta.append(np.concatenate([ml, feat]))
        y_meta.append(values[i]-1)

    X_meta = scaler.fit_transform(X_meta)
    meta.fit(X_meta, y_meta)

    st.session_state.rf = rf
    st.session_state.meta = meta
    st.session_state.scaler = scaler
    st.session_state.trained = True

    st.success("✅ TRAIN XONG")

# =========================
# RUN AI (FAST)
# =========================
if st.button("🚀 RUN AI"):

    if not st.session_state.trained:
        st.warning("❌ Chưa train AI")
        st.stop()

    rf = st.session_state.rf
    meta = st.session_state.meta
    scaler = st.session_state.scaler

    seq = values[-WINDOW:]

    X_rf = []
    y_rf = []

    for j in range(len(values)-WINDOW):
        X_rf.append(values[j:j+WINDOW])
        y_rf.append(values[j+WINDOW]-1)

    rf.fit(X_rf, y_rf)

    ml = rf.predict_proba([seq])[0]
    feat = get_features(values)

    final = meta.predict_proba(
        scaler.transform([np.concatenate([ml, feat])])
    )[0]

    st.session_state.probs = final
    st.session_state.ml_probs = ml

# =========================
# OUTPUT
# =========================
if st.session_state.probs is not None:

    probs = st.session_state.probs
    ml = st.session_state.ml_probs

    st.subheader("📊 Xác suất")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(f"{i+1}", f"{probs[i]*100:.1f}%")

    top2 = np.argsort(probs)[-2:][::-1]
    st.success(f"🎯 ĐÁNH: {top2[0]+1} + {top2[1]+1}")

    conf = sorted(probs)[-1] - sorted(probs)[-2]

    st.subheader("🔥 Confidence")
    st.write(f"{conf:.3f}")

    if conf < 0.05:
        st.warning("⚠️ Kèo yếu")
    elif conf < 0.1:
        st.info("🤔 Kèo trung")
    else:
        st.success("💰 Kèo mạnh")

    st.subheader("🧠 AI Insight")

    df = pd.DataFrame({
        "ML": ml,
        "META": probs
    }, index=[1,2,3,4])

    st.bar_chart(df)
