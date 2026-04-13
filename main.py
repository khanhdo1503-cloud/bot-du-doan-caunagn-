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

# =========================
# PARSE
# =========================
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

if "rf" not in st.session_state:
    st.session_state.rf = RandomForestClassifier(n_estimators=300, max_depth=10)

if "meta" not in st.session_state:
    st.session_state.meta = LogisticRegression(max_iter=300)

if "scaler" not in st.session_state:
    st.session_state.scaler = StandardScaler()

if "probs" not in st.session_state:
    st.session_state.probs = None

if "ml_probs" not in st.session_state:
    st.session_state.ml_probs = None

# =========================
# UI
# =========================
st.title("🧠 Fantan AI PRO DASHBOARD")

with st.form("form"):
    st.text_area("DATA (1-4)", key="data_text", height=150)
    st.form_submit_button("Update")

values = parse_data(st.session_state.data_text)
st.write(f"📊 Data: {len(values)}")

# =========================
# RUN
# =========================
if st.button("🚀 RUN AI"):

    if len(values) < WINDOW + 100:
        st.warning("Cần thêm data")
        st.stop()

    rf = st.session_state.rf
    meta = st.session_state.meta
    scaler = st.session_state.scaler

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

    # predict
    seq = values[-WINDOW:]

    X_rf = []
    y_rf = []

    for j in range(len(values)-WINDOW):
        X_rf.append(values[j:j+WINDOW])
        y_rf.append(values[j+WINDOW]-1)

    rf.fit(X_rf, y_rf)

    ml = rf.predict_proba([seq])[0]
    feat = get_features(values)

    final = meta.predict_proba(scaler.transform([np.concatenate([ml, feat])]))[0]

    st.session_state.probs = final
    st.session_state.ml_probs = ml

# =========================
# OUTPUT
# =========================
if st.session_state.probs is not None:

    probs = st.session_state.probs
    ml = st.session_state.ml_probs

    st.subheader("📊 XÁC SUẤT")

    cols = st.columns(4)
    for i in range(4):
        cols[i].metric(f"Số {i+1}", f"{probs[i]*100:.1f}%")

    top2 = np.argsort(probs)[-2:][::-1]
    st.success(f"🎯 ĐÁNH: {top2[0]+1} + {top2[1]+1}")

    # =========================
    # CONFIDENCE
    # =========================
    conf = sorted(probs)[-1] - sorted(probs)[-2]

    st.subheader("🔥 Confidence")
    st.write(f"{conf:.3f}")

    if conf < 0.05:
        st.warning("⚠️ KÈO YẾU → NGHỈ")
    elif conf < 0.1:
        st.info("🤔 KÈO TRUNG BÌNH")
    else:
        st.success("💰 KÈO MẠNH")

    # =========================
    # AI INSIGHT
    # =========================
    st.subheader("🧠 AI Insight")

    df_compare = pd.DataFrame({
        "ML": ml,
        "META": probs
    }, index=[1,2,3,4])

    st.bar_chart(df_compare)

    # =========================
    # DECISION TABLE
    # =========================
    st.subheader("📋 Chi tiết")

    for i in range(4):
        st.write(f"Số {i+1}: ML={ml[i]:.3f} → META={probs[i]:.3f}")
