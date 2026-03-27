import streamlit as st
import pandas as pd
import requests
import re
from collections import defaultdict

st.set_page_config(page_title="V41 UPGRADE BOT", layout="centered")

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

# ------------------ MATCH (CHUNG CHO STREAK 2 & 3) ------------------

def find_matches(gene, data, target_streak, min_len=5):
    results = {}
    number_counts = defaultdict(int)
    n = len(gene)

    if gene[-1][1] != target_streak:
        return {}, {}

    for L in range(min_len, n):
        pattern = gene[-L:]
        weight = L * L

        outcomes = []

        for i in range(n - L - 1):
            if gene[i:i+L] == pattern:

                if gene[i+L-1][1] != target_streak:
                    continue

                next_len = gene[i+L][1]

                pos = sum(g[1] for g in gene[:i+L])
                if pos >= len(data):
                    continue

                next_number = data[pos]

                # ===== LOGIC V40 =====
                if next_len <= 2:
                    outcomes.append("STOP_2")
                elif next_len == 3:
                    outcomes.append("STOP_3")
                else:
                    outcomes.append("TO_4+")

                # ===== NEW: ĐẾM SỐ =====
                number_counts[next_number] += weight

        if outcomes:
            results[L] = outcomes
        else:
            break

    return results, number_counts

# ------------------ ANALYSIS ------------------

def analyze(results):
    total_weight = 0
    stop2 = stop3 = to4 = 0
    score_win = score_lose = 0
    rows = []

    for L, outcomes in results.items():
        weight = L * L

        s2 = outcomes.count("STOP_2")
        s3 = outcomes.count("STOP_3")
        s4 = outcomes.count("TO_4+")

        stop2 += s2
        stop3 += s3
        to4 += s4

        score_win += s2 * weight
        score_lose += s4 * weight
        total_weight += (s2 + s3 + s4) * weight

        rows.append({
            "Gene Length": L,
            "Count": len(outcomes),
            "Stop@2": s2,
            "Stop@3": s3,
            "To≥4": s4,
            "Weight": weight
        })

    if total_weight == 0:
        return None

    p_win = score_win / total_weight
    p_lose = score_lose / total_weight
    EV = p_win*1 + p_lose*(-2)

    return {
        "stop2": stop2,
        "stop3": stop3,
        "to4": to4,
        "p_win": round(p_win*100,1),
        "p_lose": round(p_lose*100,1),
        "EV": round(EV,3),
        "table": pd.DataFrame(rows)
    }

# ------------------ UI ------------------

st.title("🧠 V41 UPGRADE BOT")

if st.button("☁️ Load từ Google Sheets"):
    data_from_sheets = fetch_sheets_data()
    if data_from_sheets:
        st.session_state["data_input"] = data_from_sheets

raw_input = st.text_area("Nhập dữ liệu", value=st.session_state.get("data_input", ""))

if st.button("Phân tích"):
    data = [int(x) for x in raw_input if x in "1234"]

    if len(data) < 200:
        st.warning("Cần ít nhất 200 data")
        st.stop()

    st.write("📊 Tổng data:", len(data))

    for name, func in [("CHẴN/LẺ", to_cl), ("TO/NHỎ", to_tn)]:
        st.subheader(name)

        seq = func(data)
        gene = get_gene(seq)

        st.write("Gene:", " ".join([f"{x}{y}" for x,y in gene[-10:]]))

        current_streak = gene[-1][1]

        if current_streak in [2, 3]:
            st.success(f"🚨 STREAK = {current_streak} → PHÂN TÍCH")

            matches, number_counts = find_matches(gene, data, current_streak)
            result = analyze(matches)

            if result:
                # ===== V40 CORE =====
                st.write("🟢 Stop2:", result["stop2"])
                st.write("⚖️ Stop3:", result["stop3"])
                st.write("💀 To4+:", result["to4"])

                st.metric("Win %", result["p_win"])
                st.metric("Lose %", result["p_lose"])
                st.metric("EV", result["EV"])

                if result["EV"] > 0 and result["p_lose"] < 35:
                    st.success("🟢 NÊN ĐÁNH")
                elif result["p_lose"] > 40:
                    st.error("🔴 NÉ GẤP")
                else:
                    st.warning("⚠️ KHÔNG RÕ")

                st.dataframe(result["table"])

                # ===== NEW: BREAKDOWN SỐ =====
                st.write("🎯 PHÂN TÍCH SỐ:")

                total = sum(number_counts.values())
                if total > 0:
                    probs = {k: round(v/total*100,2) for k,v in number_counts.items()}
                    probs = dict(sorted(probs.items(), key=lambda x: -x[1]))

                    for num, count in sorted(number_counts.items(), key=lambda x: -x[1]):
                        st.write(f"Số {num}: {count} điểm")

                    st.write("📊 XÁC SUẤT:")
                    for num, p in probs.items():
                        st.write(f"Số {num}: {p}%")

                    best = max(probs, key=probs.get)
                    st.success(f"👉 ƯU TIÊN SỐ: {best}")
                else:
                    st.warning("Không đủ dữ liệu số")

        else:
            st.info("⏳ Chưa vào vùng streak mạnh (2 hoặc 3)")
