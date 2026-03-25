import streamlit as st
import pandas as pd
import requests
import re

st.set_page_config(page_title="V39 GENE MATCH BOT", layout="centered")

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

# ------------------ GENE CORE ------------------

def get_gene(seq):
    gene = []
    count = 1
    current = seq[0]

    for i in range(1, len(seq)):
        if seq[i] == current:
            count += 1
        else:
            gene.append(f"{current}{count}")
            current = seq[i]
            count = 1

    gene.append(f"{current}{count}")
    return gene

# ------------------ MATCH ENGINE ------------------

def find_gene_matches(gene, min_len=5):
    results = {}
    n = len(gene)

    for L in range(min_len, n):
        pattern = gene[-L:]
        outcomes = []

        for i in range(n - L - 1):
            if gene[i:i+L] == pattern:
                next_gene = gene[i+L]
                s = int(next_gene[1:])

                # ===== LOGIC CHUẨN CỦA M =====
                if s == 2:
                    outcomes.append("WIN")
                elif s == 3:
                    outcomes.append("DRAW")
                elif s >= 4:
                    outcomes.append("LOSE")

        if len(outcomes) > 0:
            results[L] = outcomes
        else:
            break

    return results

# ------------------ ANALYSIS ------------------

def analyze_matches(results):
    total_weight = 0
    win_score = 0
    lose_score = 0

    total_win = 0
    total_draw = 0
    total_lose = 0

    rows = []

    for L, outcomes in results.items():
        weight = L * L

        w = outcomes.count("WIN")
        d = outcomes.count("DRAW")
        l = outcomes.count("LOSE")

        total_win += w
        total_draw += d
        total_lose += l

        win_score += w * weight
        lose_score += l * weight
        total_weight += (w + d + l) * weight

        rows.append({
            "Gene Length": L,
            "Count": len(outcomes),
            "Win": w,
            "Draw": d,
            "Lose": l,
            "Weight": weight
        })

    if total_weight == 0:
        return None

    p_win = win_score / total_weight
    p_lose = lose_score / total_weight
    EV = p_win*1 + p_lose*(-2)

    return {
        "p_win": round(p_win*100,1),
        "p_lose": round(p_lose*100,1),
        "EV": round(EV,3),
        "total_win": total_win,
        "total_draw": total_draw,
        "total_lose": total_lose,
        "table": pd.DataFrame(rows)
    }

# ------------------ UI ------------------

st.title("🧠 V39 GENE MATCH ANTI-STREAK")

# LOAD DATA
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

    matches = find_gene_matches(cl_gene)
    result = analyze_matches(matches)

    if result:
        st.write("🔥 WIN (streak=2):", result["total_win"])
        st.write("⚖️ DRAW (streak=3):", result["total_draw"])
        st.write("💀 LOSE (streak≥4):", result["total_lose"])

        st.metric("Win %", result["p_win"])
        st.metric("Lose %", result["p_lose"])
        st.metric("EV", result["EV"])

        if result["EV"] > 0 and result["p_lose"] < 35:
            st.success("🟢 VÀO")
        elif result["p_lose"] > 40:
            st.error("🔴 NÉ")
        else:
            st.warning("⚠️ CHỜ")

        st.dataframe(result["table"])
        st.write("Gene hiện tại:", " ".join(cl_gene[-10:]))

    else:
        st.warning("Không có match gene")

    # ================= TO NHỎ =================
    st.subheader("TO / NHỎ")

    tn_seq = to_tn(data)
    tn_gene = get_gene(tn_seq)

    matches = find_gene_matches(tn_gene)
    result = analyze_matches(matches)

    if result:
        st.write("🔥 WIN (streak=2):", result["total_win"])
        st.write("⚖️ DRAW (streak=3):", result["total_draw"])
        st.write("💀 LOSE (streak≥4):", result["total_lose"])

        st.metric("Win %", result["p_win"])
        st.metric("Lose %", result["p_lose"])
        st.metric("EV", result["EV"])

        if result["EV"] > 0 and result["p_lose"] < 35:
            st.success("🟢 VÀO")
        elif result["p_lose"] > 40:
            st.error("🔴 NÉ")
        else:
            st.warning("⚠️ CHỜ")

        st.dataframe(result["table"])
        st.write("Gene hiện tại:", " ".join(tn_gene[-10:]))

    else:
        st.warning("Không có match gene")
