# =============================================================================
# app.py — MLB Model Dashboard (Streamlit)
#
# Run locally:  streamlit run app.py
# Deployed at:  Streamlit Cloud (free)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Model Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; }

.stMetric {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
}

.stMetric label { color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
.stMetric [data-testid="metric-container"] > div:nth-child(2) { font-family: 'Bebas Neue'; font-size: 2rem; }

.bet-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
    border-left: 3px solid #238636;
}

.bet-card.strong { border-left-color: #3fb950; }
.bet-card.good   { border-left-color: #d29922; }
.bet-card.lean   { border-left-color: #58a6ff; }

.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 4px;
}
.tag-ml  { background: #1f4d2e; color: #3fb950; }
.tag-rl  { background: #3d2a00; color: #d29922; }
.tag-tot { background: #1a2d4a; color: #58a6ff; }
.tag-win { background: #1f4d2e; color: #3fb950; }
.tag-loss{ background: #4a1a1a; color: #f85149; }

.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.08em;
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 4px;
    margin: 24px 0 16px 0;
}

.mono { font-family: 'DM Mono', monospace; }

div[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}

.stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

footer { display: none; }
#MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

RESULTS_DIR = "results"

@st.cache_data(ttl=300)
def load_predictions(target_date: str = None) -> pd.DataFrame:
    if target_date:
        path = os.path.join(RESULTS_DIR, f"predictions_{target_date}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    # Find most recent
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "predictions_*.csv")), reverse=True)
    if files:
        return pd.read_csv(files[0])
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_value_bets(target_date: str = None) -> pd.DataFrame:
    if target_date:
        path = os.path.join(RESULTS_DIR, f"value_bets_{target_date}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "value_bets_*.csv")), reverse=True)
    if files:
        return pd.read_csv(files[0])
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_backtest_bets() -> pd.DataFrame:
    files = glob.glob(os.path.join(RESULTS_DIR, "backtest_bets_*.csv"))
    # Only load single-day backtest files (daily scoring)
    single_day = [f for f in files if f.count("_") >= 4]
    if not single_day:
        # Fall back to any backtest bets file
        single_day = files
    dfs = []
    for f in single_day:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "game", "bet"]) if "date" in combined.columns else combined
        return combined
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_all_game_results() -> pd.DataFrame:
    files = glob.glob(os.path.join(RESULTS_DIR, "backtest_*.csv"))
    # Exclude bet files
    game_files = [f for f in files if "bets" not in f]
    dfs = []
    for f in game_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        if "date" in combined.columns and "home" in combined.columns:
            combined = combined.drop_duplicates(subset=["date", "home", "away"])
        return combined
    return pd.DataFrame()


# ── Helper functions ──────────────────────────────────────────────────────────

def rating_color(rating: str) -> str:
    r = str(rating).lower()
    if "strong" in r: return "strong"
    if "good"   in r: return "good"
    return "lean"


def bet_type_tag(bet_type: str) -> str:
    t = str(bet_type).lower()
    if "moneyline" in t: return '<span class="tag tag-ml">ML</span>'
    if "run"       in t: return '<span class="tag tag-rl">RL</span>'
    return '<span class="tag tag-tot">TOT</span>'


def result_tag(won) -> str:
    if won is True or str(won).upper() == "TRUE":
        return '<span class="tag tag-win">WIN</span>'
    if won is False or str(won).upper() == "FALSE":
        return '<span class="tag tag-loss">LOSS</span>'
    return ""


def fmt_odds(odds) -> str:
    try:
        o = int(float(odds))
        return f"+{o}" if o > 0 else str(o)
    except Exception:
        return str(odds)


def fmt_pct(val) -> str:
    try:
        return f"{float(val):.1f}%"
    except Exception:
        return str(val)


def fmt_edge(val) -> str:
    try:
        v = float(val)
        if v < 1:
            v = v * 100
        return f"{v:.1f}%"
    except Exception:
        return str(val)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚾ MLB MODEL")
    st.markdown("---")

    # Date selector
    available_dates = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "predictions_*.csv")), reverse=True):
        d = os.path.basename(f).replace("predictions_", "").replace(".csv", "")
        available_dates.append(d)

    if available_dates:
        selected_date = st.selectbox("📅 Date", available_dates, index=0)
    else:
        selected_date = str(date.today())
        st.info("No prediction files found yet. Run daily.py first.")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "📊 Today's Picks",
        "📈 Performance",
        "📋 Bet History",
        "🎯 Accuracy",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#8b949e;text-align:center'>"
        "Model updates daily at 10 AM ET<br>"
        "For entertainment purposes only"
        "</div>",
        unsafe_allow_html=True
    )


# ── Load data ─────────────────────────────────────────────────────────────────

preds    = load_predictions(selected_date)
bets     = load_value_bets(selected_date)
all_bets = load_all_backtest_bets()
all_games = load_all_game_results()


# ── Page: Today's Picks ───────────────────────────────────────────────────────

if page == "📊 Today's Picks":

    st.markdown(f"# TODAY'S PICKS — {selected_date}")

    if bets.empty and preds.empty:
        st.warning("No data found for this date. Run daily.py to generate predictions.")
        st.stop()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Games Today", len(preds) if not preds.empty else 0)
    with col2:
        st.metric("Value Bets", len(bets) if not bets.empty else 0)
    with col3:
        if not bets.empty and "Type" in bets.columns:
            ml_bets = len(bets[bets["Type"] == "Moneyline"])
            st.metric("ML Bets", ml_bets)
        else:
            st.metric("ML Bets", 0)
    with col4:
        if not bets.empty and "Edge" in bets.columns:
            avg_edge = bets["Edge"].mean() * 100 if bets["Edge"].max() <= 1 else bets["Edge"].mean()
            st.metric("Avg Edge", f"{avg_edge:.1f}%")
        else:
            st.metric("Avg Edge", "—")

    # Value bets
    if not bets.empty:
        st.markdown('<div class="section-header">💰 VALUE BETS</div>', unsafe_allow_html=True)

        for bet_type in ["Moneyline", "Run Line", "Total"]:
            type_col = "Type" if "Type" in bets.columns else "type"
            if type_col not in bets.columns:
                continue
            subset = bets[bets[type_col] == bet_type]
            if subset.empty:
                continue

            st.markdown(f"**{bet_type}**")
            for _, row in subset.iterrows():
                game    = row.get("Game", row.get("game", ""))
                bet     = row.get("Bet",  row.get("bet",  ""))
                odds    = row.get("Odds", row.get("odds", ""))
                book    = row.get("Book_Prob", row.get("book_prob", ""))
                model   = row.get("Model_Prob", row.get("model_prob", ""))
                edge    = row.get("Edge", row.get("edge", ""))
                rating  = row.get("Rating", row.get("rating", "Lean"))

                card_class = rating_color(str(rating))
                type_tag   = bet_type_tag(bet_type)

                st.markdown(f"""
                <div class="bet-card {card_class}">
                    {type_tag}
                    <strong class="mono">{bet}</strong>
                    <span style="float:right;font-family:'DM Mono';font-size:0.9rem;color:#3fb950">{fmt_odds(odds)}</span>
                    <br>
                    <span style="color:#8b949e;font-size:0.8rem">{game}</span>
                    <br>
                    <span style="font-size:0.78rem;color:#8b949e">
                        Book: <strong style="color:#e6edf3">{fmt_pct(book) if float(str(book).replace('%','')) < 2 else book}</strong>
                        &nbsp;|&nbsp;
                        Model: <strong style="color:#e6edf3">{fmt_pct(model) if float(str(model).replace('%','')) < 2 else model}</strong>
                        &nbsp;|&nbsp;
                        Edge: <strong style="color:#3fb950">{fmt_edge(edge)}</strong>
                        &nbsp;|&nbsp;
                        {str(rating)}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # Predictions table
    if not preds.empty:
        st.markdown('<div class="section-header">📋 ALL PREDICTIONS</div>', unsafe_allow_html=True)

        display_cols = ["Home", "Away", "Home_Win_Pct", "Away_Win_Pct", "xTotal_Runs",
                        "Home_SP", "Away_SP", "Park_Type", "Weather", "H2H"]
        avail = [c for c in display_cols if c in preds.columns]
        df_show = preds[avail].copy()

        if "Home_Win_Pct" in df_show.columns:
            df_show["Home_Win_Pct"] = df_show["Home_Win_Pct"].apply(lambda x: f"{x:.1f}%")
        if "Away_Win_Pct" in df_show.columns:
            df_show["Away_Win_Pct"] = df_show["Away_Win_Pct"].apply(lambda x: f"{x:.1f}%")
        if "xTotal_Runs" in df_show.columns:
            df_show["xTotal_Runs"] = df_show["xTotal_Runs"].apply(lambda x: f"{x:.1f}")

        rename = {
            "Home_Win_Pct": "H%", "Away_Win_Pct": "A%",
            "xTotal_Runs": "xRuns", "Home_SP": "Home SP",
            "Away_SP": "Away SP", "Park_Type": "Park",
        }
        df_show = df_show.rename(columns=rename)
        st.dataframe(df_show, use_container_width=True, hide_index=True)


# ── Page: Performance ─────────────────────────────────────────────────────────

elif page == "📈 Performance":

    st.markdown("# PERFORMANCE TRACKER")

    if all_bets.empty:
        st.warning("No historical bet data found yet.")
        st.stop()

    bets_df = all_bets.copy()

    # Normalize columns
    if "won" in bets_df.columns:
        bets_df["won"] = bets_df["won"].astype(str).str.upper().map(
            {"TRUE": True, "FALSE": False, "1": True, "0": False}
        )
    if "profit" in bets_df.columns:
        bets_df["profit"] = pd.to_numeric(bets_df["profit"], errors="coerce").fillna(0)
    if "date" in bets_df.columns:
        bets_df["date"] = pd.to_datetime(bets_df["date"])
        bets_df = bets_df.sort_values("date")

    completed = bets_df[bets_df["won"].notna()]
    if completed.empty:
        st.warning("No completed bets yet.")
        st.stop()

    wins   = int(completed["won"].sum())
    losses = len(completed) - wins
    profit = completed["profit"].sum()
    roi    = profit / len(completed) * 100

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total Bets", len(completed))
    with col2: st.metric("Record", f"{wins}-{losses}")
    with col3: st.metric("Hit Rate", f"{wins/len(completed)*100:.1f}%")
    with col4: st.metric("Net Profit", f"${profit:+.2f}")
    with col5: st.metric("ROI", f"{roi:+.1f}%")

    # Cumulative profit chart
    st.markdown('<div class="section-header">📈 CUMULATIVE PROFIT</div>', unsafe_allow_html=True)

    completed_sorted = completed.sort_values("date").copy()
    completed_sorted["cumulative"] = completed_sorted["profit"].cumsum()
    completed_sorted["bet_num"]    = range(1, len(completed_sorted) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=completed_sorted["bet_num"],
        y=completed_sorted["cumulative"],
        mode="lines",
        line=dict(color="#3fb950", width=2),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.1)",
        name="Net Profit",
    ))
    fig.add_hline(y=0, line_color="#30363d", line_dash="dash")
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(showgrid=False, title="Bet #", color="#8b949e"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", title="Profit ($1/bet)", color="#8b949e"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # By type breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">BY BET TYPE</div>', unsafe_allow_html=True)
        if "type" in completed.columns:
            type_col = "type"
        elif "Type" in completed.columns:
            type_col = "Type"
        else:
            type_col = None

        if type_col:
            type_stats = []
            for bt in completed[type_col].unique():
                sub = completed[completed[type_col] == bt]
                w   = int(sub["won"].sum())
                l   = len(sub) - w
                p   = sub["profit"].sum()
                type_stats.append({
                    "Type": bt,
                    "W": w, "L": l,
                    "Hit%": f"{w/len(sub)*100:.0f}%",
                    "Profit": f"${p:+.2f}",
                })
            st.dataframe(pd.DataFrame(type_stats), use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">BY EDGE TIER</div>', unsafe_allow_html=True)
        if "edge" in completed.columns:
            edge_col = "edge"
        elif "Edge" in completed.columns:
            edge_col = "Edge"
        else:
            edge_col = None

        if edge_col:
            completed["edge_float"] = pd.to_numeric(completed[edge_col], errors="coerce")
            tiers = [
                ("Strong (≥12%)", 0.12, 1.0),
                ("Good (8-12%)",  0.08, 0.12),
                ("Lean (5-8%)",   0.05, 0.08),
            ]
            tier_stats = []
            for label, lo, hi in tiers:
                sub = completed[(completed["edge_float"] >= lo) & (completed["edge_float"] < hi)]
                if sub.empty:
                    continue
                w = int(sub["won"].sum())
                l = len(sub) - w
                p = sub["profit"].sum()
                tier_stats.append({
                    "Tier": label,
                    "W": w, "L": l,
                    "Hit%": f"{w/len(sub)*100:.0f}%",
                    "Profit": f"${p:+.2f}",
                })
            st.dataframe(pd.DataFrame(tier_stats), use_container_width=True, hide_index=True)

    # Daily profit chart
    st.markdown('<div class="section-header">📅 DAILY PROFIT</div>', unsafe_allow_html=True)
    if "date" in completed.columns:
        daily = completed.groupby(completed["date"].dt.date)["profit"].sum().reset_index()
        daily.columns = ["date", "profit"]
        colors = ["#3fb950" if p >= 0 else "#f85149" for p in daily["profit"]]
        fig2 = go.Figure(go.Bar(
            x=daily["date"],
            y=daily["profit"],
            marker_color=colors,
        ))
        fig2.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="DM Sans"),
            xaxis=dict(showgrid=False, color="#8b949e"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", color="#8b949e", title="Profit ($)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=250,
        )
        fig2.add_hline(y=0, line_color="#30363d")
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Bet History ─────────────────────────────────────────────────────────

elif page == "📋 Bet History":

    st.markdown("# BET HISTORY")

    if all_bets.empty:
        st.warning("No historical bet data found yet.")
        st.stop()

    bets_df = all_bets.copy()
    if "won" in bets_df.columns:
        bets_df["won"] = bets_df["won"].astype(str).str.upper().map(
            {"TRUE": True, "FALSE": False, "1": True, "0": False}
        )
    if "profit" in bets_df.columns:
        bets_df["profit"] = pd.to_numeric(bets_df["profit"], errors="coerce").fillna(0)
    if "date" in bets_df.columns:
        bets_df["date"] = pd.to_datetime(bets_df["date"])

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        type_options = ["All"] + sorted(bets_df["type"].dropna().unique().tolist()) if "type" in bets_df.columns else ["All"]
        type_filter  = st.selectbox("Bet Type", type_options)
    with col2:
        result_options = ["All", "Wins", "Losses"]
        result_filter  = st.selectbox("Result", result_options)
    with col3:
        sort_by = st.selectbox("Sort By", ["Date (newest)", "Edge (highest)", "Profit"])

    filtered = bets_df.copy()
    if type_filter != "All" and "type" in filtered.columns:
        filtered = filtered[filtered["type"] == type_filter]
    if result_filter == "Wins":
        filtered = filtered[filtered["won"] == True]
    elif result_filter == "Losses":
        filtered = filtered[filtered["won"] == False]

    if sort_by == "Date (newest)" and "date" in filtered.columns:
        filtered = filtered.sort_values("date", ascending=False)
    elif sort_by == "Edge (highest)" and "edge" in filtered.columns:
        filtered = filtered.sort_values("edge", ascending=False)
    elif sort_by == "Profit" and "profit" in filtered.columns:
        filtered = filtered.sort_values("profit", ascending=False)

    st.markdown(f"**{len(filtered)} bets**")

    # Display as cards
    for _, row in filtered.head(100).iterrows():
        won    = row.get("won")
        profit = row.get("profit", 0)
        game   = row.get("game", "")
        bet    = row.get("bet",  "")
        odds   = row.get("odds", "")
        edge   = row.get("edge", "")
        dtype  = row.get("type", "")
        ddate  = str(row.get("date", ""))[:10]

        res_tag  = result_tag(won)
        type_tag = bet_type_tag(dtype)
        profit_color = "#3fb950" if float(profit) >= 0 else "#f85149"

        st.markdown(f"""
        <div class="bet-card">
            {type_tag} {res_tag}
            <strong class="mono">{bet}</strong>
            <span style="float:right;font-family:'DM Mono';color:{profit_color}">${float(profit):+.2f}</span>
            <br>
            <span style="color:#8b949e;font-size:0.8rem">{game} &nbsp;|&nbsp; {ddate}</span>
            <br>
            <span style="font-size:0.75rem;color:#8b949e">
                Odds: <strong style="color:#e6edf3">{fmt_odds(odds)}</strong>
                &nbsp;|&nbsp; Edge: <strong style="color:#58a6ff">{fmt_edge(edge)}</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)


# ── Page: Accuracy ────────────────────────────────────────────────────────────

elif page == "🎯 Accuracy":

    st.markdown("# MODEL ACCURACY")

    if all_games.empty:
        st.warning("No game result data found yet.")
        st.stop()

    games_df = all_games.copy()
    if "ml_correct" in games_df.columns:
        games_df["ml_correct"] = games_df["ml_correct"].astype(str).str.upper().map(
            {"TRUE": True, "FALSE": False, "1": True, "0": False}
        )

    total  = len(games_df)
    if total == 0:
        st.warning("No games found.")
        st.stop()

    correct = int(games_df["ml_correct"].sum()) if "ml_correct" in games_df.columns else 0
    acc     = correct / total * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Games", total)
    with col2: st.metric("Correct Picks", correct)
    with col3: st.metric("ML Accuracy", f"{acc:.1f}%")
    with col4:
        if "runs_error" in games_df.columns:
            avg_err = games_df["runs_error"].mean()
            st.metric("Avg xRuns Error", f"{avg_err:.2f}")

    # Accuracy by confidence tier
    st.markdown('<div class="section-header">ACCURACY BY CONFIDENCE</div>', unsafe_allow_html=True)

    if "confidence" in games_df.columns and "ml_correct" in games_df.columns:
        tiers = [
            ("High (≥65%)",   games_df[games_df["confidence"] >= 65]),
            ("Medium (55-65%)", games_df[(games_df["confidence"] >= 55) & (games_df["confidence"] < 65)]),
            ("Low (<55%)",    games_df[games_df["confidence"] < 55]),
        ]
        tier_data = []
        for label, sub in tiers:
            if sub.empty:
                continue
            w = int(sub["ml_correct"].sum())
            l = len(sub) - w
            tier_data.append({
                "Confidence Tier": label,
                "W": w, "L": l,
                "Games": len(sub),
                "Accuracy": f"{w/len(sub)*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)

        # Accuracy bar chart
        fig = go.Figure(go.Bar(
            x=[t["Confidence Tier"] for t in tier_data],
            y=[float(t["Accuracy"].replace("%","")) for t in tier_data],
            marker_color=["#3fb950", "#d29922", "#58a6ff"],
            text=[t["Accuracy"] for t in tier_data],
            textposition="outside",
        ))
        fig.add_hline(y=50, line_color="#30363d", line_dash="dash",
                      annotation_text="50% baseline", annotation_font_color="#8b949e")
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="DM Sans"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", range=[0, 100], title="Accuracy %"),
            xaxis=dict(showgrid=False),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # xRuns bias
    st.markdown('<div class="section-header">xRUNS CALIBRATION</div>', unsafe_allow_html=True)
    if "model_bias" in games_df.columns:
        high = len(games_df[games_df["model_bias"] == "HIGH"])
        low  = len(games_df[games_df["model_bias"] == "LOW"])
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Model Overestimated", high, help="Model xRuns > actual")
        with col2: st.metric("Model Underestimated", low, help="Model xRuns < actual")
        with col3:
            bias_pct = (high - low) / total * 100
            direction = "HIGH" if bias_pct > 0 else "LOW"
            st.metric("Bias", f"{abs(bias_pct):.0f}% {direction}")

    # O/U results if available
    if "ou_result" in games_df.columns:
        st.markdown('<div class="section-header">OVER/UNDER RESULTS</div>', unsafe_allow_html=True)
        ou_counts = games_df["ou_result"].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("OVER",  ou_counts.get("OVER",  0))
        with col2: st.metric("UNDER", ou_counts.get("UNDER", 0))
        with col3: st.metric("PUSH",  ou_counts.get("PUSH",  0))
