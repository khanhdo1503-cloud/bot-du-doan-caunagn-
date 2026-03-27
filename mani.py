import streamlit as st
import pandas as pd
import requests
import re
from collections import Counter

st.set_page_config(page_title="V44 FINAL FIX", layout="centered")

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

# ------------------ MATCH ------------------

def find_matches(gene, data, target_streak, min_len=5):
    n = len(gene)

    stop2_nums = []
    stop3_nums = []
    to4_nums = []

    if gene[-1][1] != target_streak:
        return stop2_nums, stop3_nums, to4_nums

    for L in range(min_len, n):
        pattern = gene[-L:]

        found = False

        for i in range(n - L - 1):
            if gene[i:i+L] == pattern:

                if gene[i+L-1][1] != target_streak:
                    continue

                next_len = gene[i+L][1]

                pos = sum(g[1] for g in gene[:i+L])
                if pos >= len(data):
                    continue

                next_number = data[pos]
                found = True

                # ===== LOGIC =====
                if target_streak == 2:
                    if next_len <= 2:
                        stop2_nums.append(next_number)
                    elif next_len == 3:
                        stop3_nums.append(next_number)
                    elif next_len >= 4:
                        to4_nums.append(next_number)

                elif target_streak == 3:
                    if next_len == 3:
                        stop3_nums.append(next_number)
                    elif next_len >= 4:
                        to4_nums.append(next_number)

        if not found:
            break

    return stop2_nums, stop3_nums, to4_nums

# ------------------ UI ------------------

st.title("🧠 V44 FINAL FIX BOT")

if st.button("☁️ Load Data"):
    data = fetch_sheets_data()
    if data:
        st.session_state["data_input"] = data

raw = st.text_area("Nhập data", value=st.session_state.get("data_input", ""))

if st.button("Phân tích"):
    data = [int(x) for x in raw if x in "1234"]

    if len(data) < 200:
        st.warning("Cần ít nhất 200 data")
        st.stop()

    st.write("📊 Tổng:", len(data))

    for name, func in [("CHẴN/LẺ", to_cl), ("TO/NHỎ", to_tn)]:
        st.subheader(name)

        seq = func(data)
        gene = get_gene(seq)

        st.write("Gene:", " ".join([f"{x}{y}" for x,y in gene[-10:]]))

        current = gene[-1][1]

        if current in [2, 3]:
            st.success(f"🚨 STREAK = {current}")

            stop2, stop3, to4 = find_matches(gene, data, current)

            # ===== STREAK 2 =====
            if current == 2:
                st.write(f"🟢 STOP2 ({len(stop2)} lần):")
                st.write(dict(Counter(stop2)))

                st.write(f"⚖️ LÊN 3 ({len(stop3)} lần):")
                st.write(dict(Counter(stop3)))

                st.write(f"💀 LÊN 4+ ({len(to4)} lần):")
                st.write(dict(Counter(to4)))

            # ===== STREAK 3 =====
            if current == 3:
                st.write(f"⚖️ STOP3 ({len(stop3)} lần):")
                st.write(dict(Counter(stop3)))

                st.write(f"💀 LÊN 4+ ({len(to4)} lần):")
                st.write(dict(Counter(to4)))

        else:
            st.info("⏳ Không phải streak 2 hoặc 3")
