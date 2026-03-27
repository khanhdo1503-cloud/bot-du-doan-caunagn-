import streamlit as st
import pandas as pd
import requests
import re
from collections import defaultdict

st.set_page_config(page_title="V43 ALL STREAK BOT", layout="centered")

# ------------------ LOAD ------------------

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

# ------------------ CORE V43 ------------------

def analyze_all_streaks(data, gene, min_len=5):
    n = len(gene)
    results = defaultdict(lambda: defaultdict(int))  # number -> weighted count

    current_streak = gene[-1][1]

    for L in range(min_len, n):
        pattern = gene[-L:]
        weight = L * L

        for i in range(n - L - 1):
            if gene[i:i+L] == pattern:

                # chỉ lấy đúng streak giống hiện tại
                if gene[i+L-1][1] != current_streak:
                    continue

                pos = sum(g[1] for g in gene[:i+L])

                if pos >= len(data):
                    continue

                next_number = data[pos]
                results[L][next_number] += weight

    return results

# ------------------ PROCESS ------------------

def summarize_numbers(results):
    total_counts = defaultdict(int)

    for L in results:
        for num, val in results[L].items():
            total_counts[num] += val

    total = sum(total_counts.values())

    if total == 0:
        return None

    probs = {k: round(v/total*100, 2) for k, v in total_counts.items()}

    # sort theo xác suất
    sorted_probs = dict(sorted(probs.items(), key=lambda x: -x[1]))

    return total_counts, sorted_probs

# ------------------ UI ------------------

st.title("🧠 V43 ALL STREAK BOT")

if st.button("☁️ Load Data"):
    st.session_state["data_input"] = fetch_sheets_data()

raw = st.text_area("Data", value=st.session_state.get("data_input", ""))

if st.button("Phân tích"):
    data = [int(x) for x in raw if x in "1234"]

    if len(data) < 200:
        st.warning("Cần ít nhất 200 data")
        st.stop()

    st.write("📊 Tổng data:", len(data))

    for name, func in [("CHẴN/LẺ", to_cl), ("TO/NHỎ", to_tn)]:
        st.subheader(name)

        seq = func(data)
        gene = get_gene(seq)

        if not gene:
            continue

        current_streak = gene[-1][1]
        st.write(f"🔥 Streak hiện tại: {current_streak}")
        st.write("Gene:", " ".join([f"{x}{y}" for x,y in gene[-10:]]))

        results = analyze_all_streaks(data, gene)

        summary = summarize_numbers(results)

        if not summary:
            st.warning("Không đủ dữ liệu match")
            continue

        counts, probs = summary

        st.write("📌 SỐ LẦN XUẤT HIỆN (CÓ TRỌNG SỐ):")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            st.write(f"Số {k}: {v} điểm")

        st.write("📊 XÁC SUẤT (%):")
        for k, v in probs.items():
            st.write(f"Số {k}: {v}%")

        best = max(probs, key=probs.get)

        st.success(f"🎯 NÊN ĐÁNH SỐ: {best} ({probs[best]}%)")
