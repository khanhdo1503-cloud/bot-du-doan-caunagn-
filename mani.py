
import streamlit as st
import pandas as pd
import requests
import re
from collections import Counter

st.set_page_config(page_title="V43 FINAL BOT", layout="centered")

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

# ------------------ MATCH (FIXED CORE) ------------------

def find_matches(gene, data, target_streak, min_len=5):
    results = {}
    n = len(gene)

    if gene[-1][1] != target_streak:
        return {}

    for L in range(min_len, n):
        pattern = gene[-L:]

        outcomes = []
        stop2_nums = []
        stop3_nums = []
        to4_nums = []

        for i in range(n - L - 1):
            if gene[i:i+L] == pattern:

                if gene[i+L-1][1] != target_streak:
                    continue

                next_len = gene[i+L][1]

                pos = sum(g[1] for g in gene[:i+L])
                if pos >= len(data):
                    continue

                # ✅ FIX QUAN TRỌNG: chỉ lấy 1 số duy nhất
                next_number = data[pos]

                # ===== LOGIC =====
                if target_streak == 2:
                    if next_len <= 2:
                        outcomes.append("STOP_2")
                        stop2_nums.append(next_number)

                    elif next_len == 3:
                        outcomes.append("STOP_3")
                        stop3_nums.append(next_number)

                    else:
                        outcomes.append("TO_4+")
                        to4_nums.append(next_number)

                elif target_streak == 3:
                    if next_len == 3:
                        outcomes.append("STOP_3")
                        stop3_nums.append(next_number)
                    else:
                        outcomes.append("TO_4+")
                        to4_nums.append(next_number)

        if outcomes:
            results[L] = {
                "outcomes": outcomes,
                "stop2": stop2_nums,
                "stop3": stop3_nums,
                "to4": to4_nums
            }
        else:
            break

    return results

# ------------------ ANALYSIS ------------------

def analyze(results):
    total_weight = 0
    stop2 = stop3 = to4 = 0
    score_win = score_lose = 0
    rows = []

    for L, data in results.items():
        outcomes = data["outcomes"]
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

st.title("🧠 V43 FINAL BOT")

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

            matches = find_matches(gene, data, current_streak)
            result = analyze(matches)

            if result:
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

                # ===== CHI TIẾT SỐ (CHUẨN FIX) =====
                st.write("🎯 CHI TIẾT SỐ:")

                all_stop2 = []
                all_stop3 = []
                all_to4 = []

                for r in matches.values():
                    all_stop2 += r["stop2"]
                    all_stop3 += r["stop3"]
                    all_to4 += r["to4"]

                if all_stop2:
                    st.write("🟢 STOP2:", dict(Counter(all_stop2)))

                if all_stop3:
                    st.write("⚖️ STOP3:", dict(Counter(all_stop3)))

                if all_to4:
                    st.write("💀 TO4+:", dict(Counter(all_to4)))

        else:
            st.info("⏳ Chưa vào vùng streak mạnh (2 hoặc 3)")
