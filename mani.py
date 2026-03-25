import streamlit as st
import pandas as pd

st.set_page_config(page_title="V38 EXACT MATCH BOT", layout="centered")

# ------------------ UTILS ------------------

def get_streaks(seq):
    streaks = []
    count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            count += 1
        else:
            streaks.append(count)
            count = 1
    streaks.append(count)
    return streaks

def to_cl(seq):
    return [x % 2 == 0 for x in seq]

def to_tn(seq):
    return [x > 2 for x in seq]

# ------------------ EXACT MATCH ENGINE ------------------

def find_exact_matches(seq, min_len=15):
    results = {}

    n = len(seq)

    for L in range(min_len, n):
        pattern = seq[-L:]
        count = 0
        outcomes = []

        for i in range(n - L - 5):
            if seq[i:i+L] == pattern:
                count += 1

                future = seq[i+L:i+L+5]
                if len(future) < 3:
                    continue

                fs = get_streaks(future)

                if fs[0] == 1:
                    outcomes.append("WIN")   # gãy ngay
                elif fs[0] == 2:
                    outcomes.append("DRAW")
                elif fs[0] >= 3:
                    outcomes.append("LOSE")

        if count > 0:
            results[L] = outcomes
        else:
            break  # stop khi không còn match

    return results

# ------------------ ANALYSIS ------------------

def analyze_matches(results):
    total_weight = 0
    win_score = 0
    draw_score = 0
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
        draw_score += d * weight
        lose_score += l * weight

        total_weight += (w + d + l) * weight

        rows.append({
            "Match Length": L,
            "Count": len(outcomes),
            "Win": w,
            "Draw": d,
            "Lose": l,
            "Weight": weight
        })

    if total_weight == 0:
        return None, None

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

st.title("🧠 V38 EXACT MATCH ANTI-STREAK")

raw_input = st.text_area("Nhập dữ liệu (1 2 3 4)", "")

if st.button("Phân tích"):
    data = [int(x) for x in raw_input if x in "1234"]

    if len(data) < 200:
        st.warning("Cần ít nhất 200 data")
        st.stop()

    # ===== CHẴN LẺ =====
    st.subheader("CHẴN / LẺ")

    cl_seq = to_cl(data)
    cl_seq = [1 if x else 0 for x in cl_seq]

    matches = find_exact_matches(cl_seq)

    result = analyze_matches(matches)

    if result:
        st.write("🔥 Tổng số lần streak = 2 (WIN):", result["total_win"])
        st.write("⚖️ Tổng số lần streak = 3 (DRAW):", result["total_draw"])
        st.write("💀 Tổng số lần streak ≥4 (LOSE):", result["total_lose"])

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

    else:
        st.warning("Không tìm thấy pattern ≥15")

    # ===== TO NHỎ =====
    st.subheader("TO / NHỎ")

    tn_seq = to_tn(data)
    tn_seq = [1 if x else 0 for x in tn_seq]

    matches = find_exact_matches(tn_seq)

    result = analyze_matches(matches)

    if result:
        st.write("🔥 Tổng số lần streak = 2 (WIN):", result["total_win"])
        st.write("⚖️ Tổng số lần streak = 3 (DRAW):", result["total_draw"])
        st.write("💀 Tổng số lần streak ≥4 (LOSE):", result["total_lose"])

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

    else:
        st.warning("Không tìm thấy pattern ≥15")
