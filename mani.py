import streamlit as st
import pandas as pd
import requests
import re
from collections import Counter

st.set_page_config(page_title="V41 FIXED BOT", layout="centered")

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

# ------------------ CORE ENGINE ------------------

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

                g = gene[i+L-1]
                if g[1] != target_streak:
                    continue

                next_gene = gene[i+L]
                next_len = next_gene[1]

                # ===== FIX POS =====
                pos = sum(g[1] for g in gene[:i+L])  # start của next streak

                if pos >= len(seq):
                    continue

                # ===== STREAK 2 =====
                if target_streak == 2:
                    if next_len <= 2:
                        detail_stop.append(seq[pos])
                        outcomes.append("STOP2")

                    elif next_len == 3:
                        detail_mid.append(seq[pos])
                        outcomes.append("STOP3")

                    elif next_len >= 4:
                        detail_long.append(seq[pos])
                        outcomes.append("TO4")

                # ===== STREAK 3 =====
                elif target_streak == 3:
                    if next_len == 3:
                        detail_stop.append(seq[pos])
                        outcomes.append("STOP3")

                    elif next_len >= 4:
                        detail_long.append(seq[pos])
                        outcomes.append("TO4")

        if outcomes:
            results[L] = {
                "outcomes": outcomes,
                "stop": detail_stop,
                "mid": detail_mid,
                "long": detail_long
            }

    return results

# ------------------ ANALYSIS ------------------

def summarize(results):
    rows = []

    total_stop = 0
    total_mid = 0
    total_long = 0

    for L, data in results.items():
        s = len(data["stop"])
        m = len(data["mid"])
        l = len(data["long"])

        total_stop += s
        total_mid += m
        total_long += l

        rows.append({
            "L": L,
            "Count": s + m + l,
            "Stop": s,
            "Mid": m,
            "Long": l
        })

    return pd.DataFrame(rows), total_stop, total_mid, total_long

def breakdown_numbers(arr):
    return dict(Counter(arr))

# ------------------ UI ------------------

st.title("🧠 V41 FIXED BOT")

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
            st.warning("Không có dữ liệu")
            continue

        st.write("Gene hiện tại:", gene[-10:])

        # ===== STREAK 2 =====
        if gene[-1][1] == 2:
            st.success("ĐANG STREAK 2")

            res = analyze_streak(seq, gene, 2)
            df, s2, s3, s4 = summarize(res)

            st.write("Dừng 2:", s2)
            st.write("Lên 3:", s3)
            st.write("Lên ≥4:", s4)

            st.dataframe(df)

            all_stop = []
            all_mid = []
            all_long = []

            for r in res.values():
                all_stop += r["stop"]
                all_mid += r["mid"]
                all_long += r["long"]

            st.write("Chi tiết STOP2:", breakdown_numbers(all_stop))
            st.write("Chi tiết STOP3:", breakdown_numbers(all_mid))
            st.write("Chi tiết TO4:", breakdown_numbers(all_long))

        # ===== STREAK 3 =====
        elif gene[-1][1] == 3:
            st.warning("ĐANG STREAK 3")

            res = analyze_streak(seq, gene, 3)
            df, s3, _, s4 = summarize(res)

            st.write("Dừng 3:", s3)
            st.write("Lên ≥4:", s4)

            st.dataframe(df)

            all_stop = []
            all_long = []

            for r in res.values():
                all_stop += r["stop"]
                all_long += r["long"]

            st.write("Chi tiết STOP3:", breakdown_numbers(all_stop))
            st.write("Chi tiết TO4:", breakdown_numbers(all_long))
