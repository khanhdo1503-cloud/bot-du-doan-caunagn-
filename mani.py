import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# CONFIG
# =========================
WINDOW = 7
CONF_THRESHOLD = 0.55

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    values = data.iloc[:, 0].dropna().tolist()
    values = [int(x)-1 for x in values]  # map 1-4 -> 0-3
    return values

# =========================
# CREATE DATASET
# =========================
def create_dataset(values, window):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)

# =========================
# BUILD MODEL
# =========================
def build_model(window):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# =========================
# UI
# =========================
st.title("🧠 Fantan Bot AI")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    values = load_data(uploaded_file)

    st.success(f"Loaded {len(values)} data points")

    X, y = create_dataset(values, WINDOW)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    if st.button("Train Model"):
        model = build_model(WINDOW)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        st.session_state.model = model
        st.success("Model trained xong ✅")

# =========================
# PREDICT
# =========================
if "model" in st.session_state:

    st.subheader("🔮 Dự đoán ván tiếp theo")

    values = load_data(uploaded_file)

    input_seq = values[-WINDOW:]
    input_seq = np.array(input_seq).reshape((1, WINDOW, 1))

    pred = st.session_state.model.predict(input_seq, verbose=0)[0]

    choice = np.argmax(pred)
    confidence = np.max(pred)

    st.metric("Dự đoán", choice + 1)
    st.metric("Confidence", f"{confidence*100:.2f}%")

    # hiển thị xác suất chi tiết
    st.write("### Chi tiết xác suất:")
    for i, p in enumerate(pred):
        st.write(f"{i+1}: {p*100:.2f}%")

    # decision
    if confidence > CONF_THRESHOLD:
        st.success("✅ NÊN CHƠI")
    else:
        st.warning("❌ BỎ QUA")
