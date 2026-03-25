import streamlit as st
import pandas as pd
import requests
import re

st.set_page_config(page_title="V40 REALTIME STREAK-2 BOT", layout="centered")

# ------------------ GOOGLE SHEETS ------------------

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

# ------------------ CORE LOGIC ------------------

def find_streak2_matches(gene, min_len=5):
    results = {}
    n = len(gene)

    # chỉ chạy khi hiện tại là streak = 2
    if gene[-1][1] != 2:
        return {}

    for L in range(min_len, n):
        pattern = gene[-L:]
        outcomes = []

        for i in range(n - L - 1):
            if gene[i:i+L] == pattern:

                # gene hiện tại trong quá khứ
                current_gene = gene[i+L-1]

                # chỉ xét đúng streak = 2
                if current_gene[1] != 2:
                    continue

                next_gene = gene[i+L]
                next_len = next_gene[1]

                # ===== LOGIC CHUẨN =====
                if next_len == 1 or next_len == 2:
                    outcomes.append("STOP_2")   # dừng ở 2 → WIN
                elif next_len == 3:
                    outcomes.append("STOP_3")   # → DRAW
                elif next_len >= 4:
                    outcomes.append("TO_4+")    # → LOSE

        if len(outcomes) > 0:
            results[L] = outcomes
        else:
            break

    return results

# ------------------ ANALYSIS ------------------

def analyze(results):
    total_weight = 0

    stop2 = 0
    stop3 = 0
    to4 = 0

    score_win = 0
    score_lose = 0

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

st.title("🧠 V40 REALTIME STREAK-2 BOT")

if st.button("☁️ Load từ Google Sheets"):
    data_from_sheets = fetch_sheets_data()
    if data_from_sheets:
        st.session_state["data_input"] = data_from_sheets
        st.success(f"Đã tải {len(data_from_sheets)} dữ liệu!")

raw_input = st.text_area(
    "Nhập dữ liệu (1 2 3 4)",
    value=st.session_state.get("data_input", "")
)

if st.button("Phân tích"):
    data = [int(x) for x in raw_input if x in "1234"]

    if len(data) < 200:
        st.warning("Cần ít nhất 200 data")
        st.stop()

    st.write(f"📊 Tổng data: {len(data)}")

    # ================= CHẴN LẺ =================
    st.subheader("CHẴN / LẺ")

    cl_seq = to_cl(data)
    cl_gene = get_gene(cl_seq)

    st.write("Gene hiện tại:", " ".join([f"{x}{y}" for x,y in cl_gene[-10:]]))

    if cl_gene[-1][1] == 2:
        st.success("🚨 ĐANG Ở STREAK = 2 → PHÂN TÍCH")

        matches = find_streak2_matches(cl_gene)
        result = analyze(matches)

        if result:
            st.write("🟢 Dừng tại 2 (WIN):", result["stop2"])
            st.write("⚖️ Lên 3 (DRAW):", result["stop3"])
            st.write("💀 Lên ≥4 (LOSE):", result["to4"])

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
        else:
            st.warning("Không có match đủ mạnh")
    else:
        st.info("⏳ Chưa phải streak = 2 → chưa vào kèo")

    # ================= TO NHỎ =================
    st.subheader("TO / NHỎ")

    tn_seq = to_tn(data)
    tn_gene = get_gene(tn_seq)

    st.write("Gene hiện tại:", " ".join([f"{x}{y}" for x,y in tn_gene[-10:]]))

    if tn_gene[-1][1] == 2:
        st.success("🚨 ĐANG Ở STREAK = 2 → PHÂN TÍCH")

        matches = find_streak2_matches(tn_gene)
        result = analyze(matches)

        if result:
            st.write("🟢 Dừng tại 2 (WIN):", result["stop2"])
            st.write("⚖️ Lên 3 (DRAW):", result["stop3"])
            st.write("💀 Lên ≥4 (LOSE):", result["to4"])

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
        else:
            st.warning("Không có match đủ mạnh")
    else:
        st.info("⏳ Chưa phải streak = 2 → chưa vào kèo")
