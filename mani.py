import streamlit as st
import pandas as pd
import requests
import re
from collections import Counter
import plotly.graph_objects as go

st.set_page_config(page_title="V42 VISUAL BOT", layout="wide")

# ------------------ LOAD DATA ------------------

def fetch_sheets_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5-pPONvbU7PR7FteVtEBvN6EuudQ2rgbV3sHX-Ngy1PALF4nvyTBidXOXXE325_TLKKDJwZB7xFgH/pub?output=csv"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return "".join(re.findall(r'[1-4]', res.text))
    except:
        return ""
    return ""

# ------------------ CONVERT ------------------

def to_cl(seq):
    return ["C" if x % 2 == 0 else "L" for x in seq]

def to_tn(seq):
    return ["T" if x > 2 else "X" for x in seq]

# ------------------ GENE ------------------

def get_gene(seq):
    if not seq:
        return []

    gene = []
    count = 1
    current = seq[0]

    for i in range(1, len(seq)):
        if seq[i] == current:
            count += 1
        else:
            gene.append((current, count))
            current = seq[i]
            count = 1

    gene.append((current, count))
    return gene

# ------------------ ENGINE ------------------

def analyze_streak(seq, gene, target_streak, min_len=5):
    n = len(gene)
    results = {}

    if n == 0 or gene[-1][1] != target_streak:
        return {}

    for L in range(min_len, n):
        pattern = gene[-L:]

        outcomes = []
        detail_stop = []
        detail_mid = []
        detail_long = []

        for i in range(n - L - 1):
            if gene[i:i+L] == pattern:

                if gene[i+L-1][1] != target_streak:
                    continue

                next_len = gene[i+L][1]
                pos = sum(g[1] for g in gene[:i+L])

                if pos >= len(seq):
                    continue

                if target_streak == 2:
                    if next_len <= 2:
                        detail_stop.append(seq[pos])
                        outcomes.append("STOP2")
                    elif next_len == 3:
                        detail_mid.append(seq[pos])
                        outcomes.append("STOP3")
                    else:
                        detail_long.append(seq[pos])
                        outcomes.append("TO4")

                elif target_streak == 3:
                    if next_len == 3:
                        detail_stop.append(seq[pos])
                        outcomes.append("STOP3")
                    else:
                        detail_long.append(seq[pos])
                        outcomes.append("TO4")

        if outcomes:
            results[L] = {
                "stop": detail_stop,
                "mid": detail_mid,
                "long": detail_long
            }

    return results

# ------------------ ANALYSIS ------------------

def summarize(results):
    rows = []
    total_stop = total_mid = total_long = 0

    for L, data in results.items():
        s = len(data["stop"])
        m = len(data["mid"])
        l = len(data["long"])

        total_stop += s
        total_mid += m
        total_long += l

        rows.append({"L": L, "Stop": s, "Mid": m, "Long": l})

    return pd.DataFrame(rows), total_stop, total_mid, total_long

def breakdown_numbers(arr):
    return dict(Counter(arr))

# ------------------ VISUAL ------------------

def plot_gene(gene):
    colors = {"C": "blue", "L": "red", "T": "green", "X": "orange"}

    x = []
    y = []
    c = []

    index = 0
    for g in gene:
        for _ in range(g[1]):
            x.append(index)
            y.append(g[0])
            c.append(colors.get(g[0], "gray"))
            index += 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(color=c, size=6)
    ))

    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def highlight_streak(gene):
    last = gene[-1]
    return f"🔥 STREAK HIỆN TẠI: {last[0]} x {last[1]}"

# ------------------ UI ------------------

st.title("🧠 V42 VISUAL BOT")

if st.button("Load Data"):
    data = fetch_sheets_data()
    st.session_state["data_input"] = data

raw = st.text_area("Data", value=st.session_state.get("data_input", ""))

if st.button("Phân tích"):
    data = [int(x) for x in raw if x in "1234"]

    st.write("Tổng data:", len(data))

    for name, func in [("CHẴN/LẺ", to_cl), ("TO/NHỎ", to_tn)]:
        st.subheader(name)

        seq = func(data)
        gene = get_gene(seq)

        if not gene:
            continue

        # 🔥 highlight
        st.markdown(highlight_streak(gene))

        # 🎯 gene đẹp
        st.write("Gene:", " | ".join([f"{g[0]}{g[1]}" for g in gene[-10:]]))

        # 📊 timeline
        st.plotly_chart(plot_gene(gene[-100:]))

        if gene[-1][1] == 2:
            st.success("ĐANG STREAK 2")

            res = analyze_streak(seq, gene, 2)
            df, s2, s3, s4 = summarize(res)

            col1, col2, col3 = st.columns(3)
            col1.metric("STOP2", s2)
            col2.metric("LÊN 3", s3)
            col3.metric("LÊN ≥4", s4)

            st.dataframe(df)

            all_stop = []
            all_mid = []
            all_long = []

            for r in res.values():
                all_stop += r["stop"]
                all_mid += r["mid"]
                all_long += r["long"]

            st.write("STOP2:", breakdown_numbers(all_stop))
            st.write("STOP3:", breakdown_numbers(all_mid))
            st.write("TO4:", breakdown_numbers(all_long))

        elif gene[-1][1] == 3:
            st.warning("ĐANG STREAK 3")

            res = analyze_streak(seq, gene, 3)
            df, s3, _, s4 = summarize(res)

            col1, col2 = st.columns(2)
            col1.metric("STOP3", s3)
            col2.metric("LÊN ≥4", s4)

            st.dataframe(df)

            all_stop = []
            all_long = []

            for r in res.values():
                all_stop += r["stop"]
                all_long += r["long"]

            st.write("STOP3:", breakdown_numbers(all_stop))
            st.write("TO4:", breakdown_numbers(all_long))
