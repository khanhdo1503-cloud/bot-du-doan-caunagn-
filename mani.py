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
# STATS
# =========================
def calc_frequency(values):
    freq = [0,0,0,0]
    for v in values[-50:]:
        freq[v-1] += 1
    total = sum(freq)
    return [f/total for f in freq]

def calc_streak(values):
    streak = [0,0,0,0]
    last_seen = {1:0,2:0,3:0,4:0}

    for i, v in enumerate(reversed(values)):
        if last_seen[v] == 0:
            last_seen[v] = i+1

    max_gap = max(last_seen.values())
    return [last_seen[i+1]/max_gap for i in range(4)]

# =========================
# INIT
# =========================
if "data_text" not in st.session_state:
    st.session_state.data_text = ""

if "result" not in st.session_state:
    st.session_state.result = None

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot LV4")

if st.button("☁️ Load Data"):
    data = load_data()
    st.session_state.data_text = "".join(str(x) for x in data)

data_text = st.text_area("DATA", value=st.session_state.data_text, height=200)
st.session_state.data_text = data_text

values = parse_data(data_text)

st.write(f"📊 Tổng data: {len(values)}")

# =========================
# RUN
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW:
        st.warning("Chưa đủ data")
    else:
        # ML
        X, y = create_dataset(values, WINDOW)
        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        seq = np.array(values[-WINDOW:]).reshape(1, -1)
        ml_pred = model.predict_proba(seq)[0]

        # stats
        freq = calc_frequency(values)
        streak = calc_streak(values)

        # hybrid
        final = []
        for i in range(4):
            score = 0.5*ml_pred[i] + 0.3*freq[i] + 0.2*streak[i]
            final.append(score)

        final = np.array(final)
        final = final / np.sum(final)

        # lấy top 2
        top2 = np.argsort(final)[-2:][::-1]

        st.subheader("🔮 KẾT QUẢ")

        for idx in top2:
            st.write(f"{idx+1} → {final[idx]*100:.2f}%")

        st.success(f"👉 Nên đánh: {top2[0]+1} + {top2[1]+1}")
