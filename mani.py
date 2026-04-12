import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

WINDOW = 10
RETRAIN_THRESHOLD = 30

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
st.title("🧠 Fantan Bot Auto Learning")

if st.button("☁️ Load Data"):
    data = load_data()
    st.session_state.data_text = "".join(str(x) for x in data)

data_text = st.text_area("📥 DATA", value=st.session_state.data_text, height=200)
st.session_state.data_text = data_text

values = parse_data(data_text)

st.write(f"📊 Tổng data: {len(values)}")
st.write(f"🔄 Số ván mới chưa học: {st.session_state.new_count}")

# =========================
# HIỂN THỊ 20 VÁN GẦN NHẤT
# =========================
st.subheader("📋 20 VÁN GẦN NHẤT")
st.write(values[-20:])

# =========================
# TRAIN (AUTO)
# =========================
def train_model(values):
    X, y = create_dataset(values, WINDOW)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    return model

# =========================
# RUN BOT
# =========================
if st.button("🚀 RUN BOT"):

    if len(values) < WINDOW:
        st.warning("Chưa đủ data")
    else:
        # kiểm tra có cần train lại không
        if st.session_state.model is None or st.session_state.new_count >= RETRAIN_THRESHOLD:
            st.session_state.model = train_model(values)
            st.session_state.new_count = 0
            st.success("🧠 Đã tự học lại")

        # predict
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
# HIỂN THỊ KẾT QUẢ
# =========================
if st.session_state.probs is not None:

    probs = st.session_state.probs

    st.subheader("📊 XÁC SUẤT 4 SỐ")

    for i in range(4):
        st.write(f"{i+1}: {probs[i]*100:.2f}%")

    top2 = np.argsort(probs)[-2:][::-1]

    st.subheader("🔮 GỢI Ý")
    st.success(f"👉 Đánh: {top2[0]+1} + {top2[1]+1}")

# =========================
# THÊM DATA MỚI
# =========================
st.subheader("➕ THÊM DATA MỚI")

new_input = st.text_input("Nhập nhanh (ví dụ: 1234)")

if st.button("➕ ADD"):

    new_vals = parse_data(new_input)

    if new_vals:
        st.session_state.data_text += "".join(str(x) for x in new_vals)
        st.session_state.new_count += len(new_vals)
        st.success(f"Đã thêm {len(new_vals)} số")
    else:
        st.error("Input lỗi")
