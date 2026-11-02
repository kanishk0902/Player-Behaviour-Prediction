"""
Streamlit prototype: Predicting Batter Aggression in the IPL (Logistic Regression)
Run:  streamlit run app.py
"""
import hashlib
import io
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ipl_model as M

st.set_page_config(page_title="IPL Batter Aggression", page_icon="🏏", layout="wide")

RED, BLUE, GREY = "#e4572e", "#4c78a8", "#9aa5b1"
DEFAULT_CSV = next((p for p in ("ball_by_ball_ipl.csv", "ball_by_ball_ipl.csv.gz") if os.path.exists(p)),
                   "ball_by_ball_ipl.csv")
CACHE_DIR = ".cache"
ANY_BOWLER, ANY_GROUND = "Any bowler (whole attack)", "Any ground (average)"


# ----------------------------------------------------------------------------------
# Loading / training (cached in memory and on disk)
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle(csv_bytes: bytes | None, csv_path: str, _key: str):
    """Build features and train the model once; reuse from disk on later launches."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"bundle_{_key}.joblib")
    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    raw = pd.read_csv(io.BytesIO(csv_bytes)) if csv_bytes is not None else pd.read_csv(csv_path)
    d, glob = M.prepare(raw)
    results = M.train_and_evaluate(d, glob)
    bundle = dict(
        glob=glob, results=results, lookups=M.build_lookups(d),
        meta=dict(rows=len(raw), legal=len(d), matches=int(d["Match ID"].nunique()),
                  batters=int(d["Batter"].nunique()), bowlers=int(d["Bowler"].nunique()),
                  venues=int(d["Venue"].nunique()), first=str(d.Date.min().date()), last=str(d.Date.max().date())),
        eda=make_eda(d),
    )
    joblib.dump(bundle, cache_file, compress=3)
    return bundle


def make_eda(d: pd.DataFrame) -> dict:
    tr = d[d.year <= M.TRAIN_END_YEAR]
    bins = pd.cut(tr.bat_balls_before, [-1, 0, 2, 5, 10, 20, 40, 200],
                  labels=["0", "1-2", "3-5", "6-10", "11-20", "21-40", "40+"])
    bat = tr.groupby("Batter").agg(balls=("Valid Ball", "sum"), bound=(M.TARGET, "mean"))
    return dict(
        class_balance=tr[M.TARGET].value_counts(normalize=True).sort_index() * 100,
        by_over=tr.groupby("Over")[M.TARGET].mean() * 100,
        settle=tr.groupby(bins, observed=True)[M.TARGET].mean() * 100,
        by_year=d.groupby("year")[M.TARGET].mean() * 100,
        top_batters=bat[bat.balls >= 1500].sort_values("bound", ascending=False).head(12),
        phase=d.groupby("phase").agg(balls=("Valid Ball", "sum"), bound=(M.TARGET, "sum"))
              .assign(pct=lambda x: x.bound / x.balls * 100).reindex(["PP", "MID", "DEATH"]),
    )


def file_key(data: bytes | None, path: str) -> str:
    if data is not None:
        return hashlib.md5(data).hexdigest()[:10]
    st_ = os.stat(path)
    return hashlib.md5(f"{path}{st_.st_size}{st_.st_mtime}".encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------------
# Sidebar: data source
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.title("🏏 IPL Batter Aggression")
    st.caption("Mini project · CCE3131 AI/ML Lab · Logistic Regression")
    uploaded = st.file_uploader("Dataset (ball_by_ball_ipl.csv)", type="csv",
                                help="Not needed if the CSV sits next to app.py")

csv_bytes = uploaded.getvalue() if uploaded else None
if csv_bytes is None and not os.path.exists(DEFAULT_CSV):
    st.title("🏏 Predicting Batter Aggression in the IPL")
    st.info(f"Upload **{DEFAULT_CSV}** in the sidebar, or place it in the same folder as `app.py` and refresh.\n\n"
            "The dataset is on Kaggle: *jamiewelsh2/ball-by-ball-ipl*.")
    st.stop()

try:
    with st.spinner("First run: building leak-free features and training the model (about a minute)..."):
        B = get_bundle(csv_bytes, DEFAULT_CSV, file_key(csv_bytes, DEFAULT_CSV))
except Exception as e:
    st.error(f"Could not process the dataset: {e}")
    st.stop()

R, L, G, meta, eda = B["results"], B["lookups"], B["glob"], B["meta"], B["eda"]
AVG_B, AVG_S = G["GLOBAL_B"], R["avg_six"]


# ----------------------------------------------------------------------------------
# Small chart helpers
# ----------------------------------------------------------------------------------
def style(fig, h=340, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white", **kw)
    return fig


def gauge_bar(label, p, avg, scale, color):
    fig = go.Figure(go.Bar(x=[p * 100], y=[label], orientation="h", marker_color=color,
                           text=[f"{p:.1%}"], textposition="outside", cliponaxis=False))
    fig.add_vline(x=avg * 100, line_dash="dash", line_color="#555",
                  annotation_text=f"avg {avg:.1%}", annotation_position="top")
    fig.update_xaxes(range=[0, scale * 100], ticksuffix="%", showgrid=False)
    fig.update_yaxes(showticklabels=False)
    return style(fig, h=110)


# ----------------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------------
st.title("Predicting Batter Aggression in the IPL")
st.caption(f"Will the batter hit a boundary (4 or 6) on the next legal ball? Trained on IPL {meta['first'][:4]}–2021, "
           f"tested on 2022–2023 · {meta['legal']:,} legal deliveries · {meta['matches']:,} matches")

tab_pred, tab_perf, tab_drv, tab_data, tab_about = st.tabs(
    ["🎯 Predict a ball", "📊 Model performance", "🧠 What drives aggression", "🔎 Data explorer", "ℹ️ About"])

# ================================ PREDICT =========================================
with tab_pred:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Who is playing?")
        teams = L["teams"]
        c1, c2 = st.columns(2)
        bat_team = c1.selectbox("Batting team", teams,
                                index=teams.index("Royal Challengers Bangalore") if "Royal Challengers Bangalore" in teams else 0)
        opp_opts = [t for t in teams if t != bat_team]
        opp = c2.selectbox("Opposition", opp_opts,
                           index=opp_opts.index("Kolkata Knight Riders") if "Kolkata Knight Riders" in opp_opts else 0)
        bat_opts = L["squad_bat"][bat_team]
        batter = st.selectbox("Batter", bat_opts, index=bat_opts.index("V Kohli") if "V Kohli" in bat_opts else 0)
        bowl_opts = [ANY_BOWLER] + L["squad_bowl"][opp]
        bowler = st.selectbox("Bowler", bowl_opts)
        venue_opts = [ANY_GROUND] + L["venues"]
        venue = st.selectbox("Venue", venue_opts,
                             index=venue_opts.index("Eden Gardens") if "Eden Gardens" in venue_opts else 0)

        st.subheader("Match situation")
        innings = st.radio("Innings", [1, 2], horizontal=True,
                           format_func=lambda i: "1st (batting first)" if i == 1 else "2nd (chasing)")
        c1, c2 = st.columns(2)
        over = c1.slider("Over", 1, 20, 12)
        ball = c2.slider("Ball of the over", 1, 6, 1)
        c1, c2 = st.columns(2)
        wickets = c1.slider("Wickets down", 0, 9, 2)
        faced = c2.slider("Balls faced by batter", 0, 60, 10)
        c1, c2, c3 = st.columns(3)
        sr = c1.number_input("Batter strike rate", 0.0, 400.0, 130.0, 5.0)
        crr = c2.number_input("Team run rate", 0.0, 20.0, 8.5, 0.1)
        rrr = c3.number_input("Required run rate", 0.0, 36.0, 9.0 if innings == 2 else 0.0, 0.1,
                              disabled=(innings == 1))
        last6 = st.slider("Team runs in last 6 balls", 0, 30, 7)

    kw = dict(batter=batter, bowler="" if bowler == ANY_BOWLER else bowler,
              venue="" if venue == ANY_GROUND else venue, innings=innings, over=over, ball=ball,
              wickets=wickets, balls_faced=faced, strike_rate=sr, run_rate=crr, required_rate=rrr,
              last6_runs=last6, opposition=opp)
    res = M.predict_ball(B, **kw)

    with right:
        st.subheader(f"{res['batter']} vs {res['bowler']}")
        st.caption(f"{res['venue']} · innings {innings}, over {over}.{ball} ({res['phase']}), "
                   f"{wickets} down, {faced} balls faced")
        m1, m2, m3 = st.columns(3)
        m1.metric("Boundary (4/6)", f"{res['p_bound']:.1%}", f"x{res['bound_ratio']:.2f} vs average ball",
                  delta_color="off", delta_arrow="off")
        m2.metric("Six", f"{res['p_six']:.1%}", f"x{res['six_ratio']:.2f} vs average ball", delta_color="off", delta_arrow="off")
        m3.metric("≥1 six in next 6 balls", f"{res['p_six_over']:.1%}", help="Approximation: 1 − (1 − p)^6")
        st.plotly_chart(gauge_bar("Boundary", res["p_bound"], AVG_B, 0.40, RED), width='stretch')
        st.plotly_chart(gauge_bar("Six", res["p_six"], AVG_S, 0.20, BLUE), width='stretch')

        verdict_fn = st.success if res["six_ratio"] > 1.15 else (st.warning if res["six_ratio"] < 0.85 else st.info)
        verdict_fn(f"**{res['verdict']}.**  At the tuned threshold ({R['best_thr']:.3f}) the model "
                   f"{'**flags**' if res['flagged'] else 'does **not** flag'} this ball as boundary-likely.")

        st.markdown("**Why this probability?**  Push of each factor compared with an average ball")
        c = res["contrib"].rename(index=M.FEATURE_LABELS)
        c = c[c.abs() > 0.005].sort_values()
        if len(c) > 10:
            c = pd.concat([c.head(5), c.tail(5)])
        fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                               marker_color=[RED if v > 0 else BLUE for v in c.values]))
        fig.update_xaxes(title="change in log-odds of a boundary (red ↑ more likely, blue ↓ less likely)")
        st.plotly_chart(style(fig, h=330), width='stretch')

        with st.expander("How does it change through the innings?"):
            sweep = []
            for o in range(1, 21):
                k = dict(kw, over=o, ball=1)
                sweep.append(M.predict_ball(B, **k)["p_bound"])
            fig = go.Figure(go.Scatter(x=list(range(1, 21)), y=np.array(sweep) * 100, mode="lines+markers",
                                       line_color=RED))
            fig.add_hline(y=AVG_B * 100, line_dash="dash", line_color="#555", annotation_text="avg ball")
            fig.add_vline(x=over, line_color=GREY)
            fig.update_layout(xaxis_title="Over (other inputs held fixed)", yaxis_title="P(boundary) %")
            st.plotly_chart(style(fig, h=280), width='stretch')
            st.caption("Only the phase flag changes with the over here; wickets, run rate and balls faced stay as entered.")

        st.caption(f"{res['batter']}: {res['career_balls']:,} IPL balls on record, "
                   f"career boundary rate {res['career_bound']:.1%}. "
                   f"Per-ball probability from a Logistic Regression model trained on IPL 2008–2021; "
                   f"player history up to {L['last_year']}.")

# ================================ PERFORMANCE =====================================
with tab_perf:
    st.subheader("Held-out seasons 2022–2023")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC", f"{R['auc']:.3f}", help="0.5 = random guessing")
    k2.metric("Log loss", f"{R['logloss']:.3f}",
              f"{(R['logloss_base'] - R['logloss']) / R['logloss_base']:.1%} better than constant",
              help=f"Constant-probability baseline: {R['logloss_base']:.3f}")
    k3.metric("Tuned threshold", f"{R['best_thr']:.3f}", help="F1-optimal on out-of-fold training predictions")
    k4.metric("Always-'no boundary' accuracy", f"{R['base_acc']:.1%}",
              help="Why accuracy alone misleads on imbalanced data")

    st.dataframe(R["metrics"].style.format("{:.4f}"), width='stretch')

    cols = st.columns(3)
    for col, (name, cm) in zip(cols, R["confusion"].items()):
        fig = go.Figure(go.Heatmap(z=cm, x=["No boundary", "Boundary"], y=["No boundary", "Boundary"],
                                   colorscale="Blues", showscale=False, texttemplate="%{z:,}"))
        fig.update_yaxes(autorange="reversed", title="Actual")
        fig.update_xaxes(title="Predicted")
        fig.update_layout(title=dict(text=name, font_size=13))
        col.plotly_chart(style(fig, h=300), width='stretch')

    c1, c2, c3 = st.columns(3)
    fpr, tpr = R["roc"]
    fig = go.Figure([go.Scatter(x=fpr, y=tpr, name=f"LogReg (AUC {R['auc']:.3f})", line_color=RED),
                     go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="#555"))])
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    c1.plotly_chart(style(fig), width='stretch')

    mp, fp = R["calib"]
    fig = go.Figure([go.Scatter(x=mp, y=fp, mode="lines+markers", name="Model", line_color=BLUE),
                     go.Scatter(x=[0, .5], y=[0, .5], name="Perfect", line=dict(dash="dash", color="#555"))])
    fig.update_layout(title="Calibration", xaxis_title="Predicted probability", yaxis_title="Observed frequency")
    c2.plotly_chart(style(fig), width='stretch')

    thr, f1s = R["f1_curve"]
    fig = go.Figure(go.Scatter(x=thr, y=f1s, line_color=RED))
    fig.add_vline(x=R["best_thr"], line_dash="dash", annotation_text=f"best {R['best_thr']:.3f}")
    fig.update_layout(title="F1 vs threshold (training, out-of-fold)", xaxis_title="Threshold", yaxis_title="F1")
    c3.plotly_chart(style(fig), width='stretch')

    st.subheader("Player level: predicted vs actual boundary rate")
    pl = R["players"]
    c1, c2 = st.columns([1.3, 1])
    corr = pl[["actual", "predicted"]].corr().iloc[0, 1]
    lim = [min(pl.actual.min(), pl.predicted.min()) * 100 - 1, max(pl.actual.max(), pl.predicted.max()) * 100 + 1]
    fig = go.Figure([go.Scatter(x=pl.predicted * 100, y=pl.actual * 100, mode="markers", text=pl.index,
                                marker=dict(size=pl.balls / 25, color=BLUE, opacity=.6),
                                hovertemplate="%{text}<br>predicted %{x:.1f}%<br>actual %{y:.1f}%<extra></extra>"),
                     go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#555"), showlegend=False)])
    fig.update_layout(title=f"{len(pl)} batters with ≥250 balls (r = {corr:.2f})", showlegend=False,
                      xaxis_title="Mean predicted boundary %", yaxis_title="Actual boundary %")
    c1.plotly_chart(style(fig, h=380), width='stretch')
    show = (pl.head(10).assign(actual=lambda x: (x.actual * 100).round(1),
                               predicted=lambda x: (x.predicted * 100).round(1))
            .rename(columns={"balls": "Balls", "actual": "Actual %", "predicted": "Predicted %"}))
    c2.markdown("**Top 10 most aggressive batters, 2022–23**")
    c2.dataframe(show, width='stretch')
    st.caption("The model ranks aggressive batters above cautious ones but under-predicts the biggest hitters "
               "because the smoothing pulls extreme players towards the league average.")

# ================================ DRIVERS =========================================
with tab_drv:
    st.subheader("Standardised logistic-regression coefficients")
    cd = R["coef_df"].copy()
    cd.index = [M.FEATURE_LABELS[i] for i in cd.index]
    cd = cd.sort_values("coefficient")
    fig = go.Figure(go.Bar(x=cd.coefficient, y=cd.index, orientation="h",
                           marker_color=[RED if v > 0 else BLUE for v in cd.coefficient]))
    fig.update_xaxes(title="log-odds change per +1 SD (red = more boundaries, blue = fewer)")
    c1, c2 = st.columns([1.4, 1])
    c1.plotly_chart(style(fig, h=560), width='stretch')
    c2.dataframe(cd.sort_values("coefficient", ascending=False).round(3), width='stretch', height=560)
    st.markdown(
        "- **Settling in** is the strongest driver: the longer a batter has been at the crease, the likelier a boundary.\n"
        "- **Middle overs** suppress boundaries compared with the Powerplay.\n"
        "- **Player history** persists: bowlers who leak boundaries and batters who hit them in this phase keep doing so.\n"
        "- **Pressure**: a higher required run rate pushes batters to attack; losing wickets makes them careful.")

    st.subheader("Scenario comparison")
    med = R["feat_median"].to_dict()
    scen = {
        "Typical batter, middle overs, innings 1": dict(chase=0, phase_MID=1, phase_DEATH=0),
        "Aggressive finisher, death overs, chasing 12+ an over":
            dict(chase=1, phase_MID=0, phase_DEATH=1, inn_wkts_before=5, crr=8.5, rrr=12, team_last6=10,
                 bat_balls_log=np.log1p(15), new_bat=0, bat_sr_inn=150,
                 bat_hist_bound=med["bat_hist_bound"] * 1.6, bat_phase_bound=med["bat_phase_bound"] * 1.6),
        "Tail-ender just arrived, middle overs, 7 down":
            dict(chase=1, phase_MID=1, phase_DEATH=0, inn_wkts_before=7, new_bat=1, bat_balls_log=0, bat_sr_inn=0,
                 bat_hist_bound=med["bat_hist_bound"] * 0.5, bat_phase_bound=med["bat_phase_bound"] * 0.5),
    }
    rows = []
    for name, ov in scen.items():
        row = dict(med, season=L["last_year"] - 2008)
        row.update(ov)
        p = R["model"].predict_proba(pd.DataFrame([row])[M.FEATURES])[0, 1]
        rows.append({"Scenario": name, "P(boundary)": f"{p:.1%}", "vs average ball": f"x{p / AVG_B:.2f}"})
    st.table(pd.DataFrame(rows).set_index("Scenario"))

# ================================ DATA EXPLORER ===================================
with tab_data:
    st.subheader("Dataset")
    c = st.columns(5)
    c[0].metric("Deliveries", f"{meta['rows']:,}")
    c[1].metric("Legal (modelled)", f"{meta['legal']:,}")
    c[2].metric("Matches", f"{meta['matches']:,}")
    c[3].metric("Batters / bowlers", f"{meta['batters']} / {meta['bowlers']}")
    c[4].metric("Venues", meta["venues"])

    c1, c2, c3 = st.columns(3)
    cb = eda["class_balance"]
    fig = go.Figure(go.Bar(x=["No boundary", "Boundary"], y=cb.values, marker_color=[GREY, RED],
                           text=[f"{v:.1f}%" for v in cb.values], textposition="outside"))
    fig.update_layout(title="Class balance (training)", yaxis_title="% of deliveries", yaxis_range=[0, 100])
    c1.plotly_chart(style(fig), width='stretch')

    bo = eda["by_over"]
    fig = go.Figure(go.Scatter(x=bo.index, y=bo.values, mode="lines+markers", line_color=RED))
    fig.add_vrect(x0=.5, x1=6.5, fillcolor=BLUE, opacity=.08, line_width=0)
    fig.add_vrect(x0=15.5, x1=20.5, fillcolor=RED, opacity=.08, line_width=0)
    fig.update_layout(title="Boundary % by over (blue PP, red death)", xaxis_title="Over", yaxis_title="% boundaries")
    c2.plotly_chart(style(fig), width='stretch')

    sn = eda["settle"]
    fig = go.Figure(go.Bar(x=sn.index.astype(str), y=sn.values, marker_color=BLUE))
    fig.update_layout(title="Boundary % vs balls already faced", xaxis_title="Balls faced so far",
                      yaxis_title="% boundaries")
    c3.plotly_chart(style(fig), width='stretch')

    c1, c2 = st.columns(2)
    yr = eda["by_year"]
    fig = go.Figure(go.Scatter(x=yr.index, y=yr.values, mode="lines+markers", line_color=RED))
    fig.add_vline(x=M.TRAIN_END_YEAR + .5, line_dash="dash", annotation_text="test period")
    fig.update_layout(title="Boundary % per season", xaxis_title="Season", yaxis_title="% boundaries")
    c1.plotly_chart(style(fig, h=380), width='stretch')

    tb = eda["top_batters"].iloc[::-1]
    fig = go.Figure(go.Bar(x=tb.bound * 100, y=tb.index, orientation="h", marker_color=BLUE))
    fig.add_vline(x=AVG_B * 100, line_dash="dash", annotation_text="league avg")
    fig.update_layout(title="Most aggressive batters (≥1,500 balls, training seasons)", xaxis_title="% boundaries")
    c2.plotly_chart(style(fig, h=380), width='stretch')

    ph = eda["phase"].rename(index={"PP": "Powerplay (1-6)", "MID": "Middle (7-15)", "DEATH": "Death (16-20)"})
    st.dataframe(ph.rename(columns={"balls": "Deliveries", "bound": "Boundaries", "pct": "Boundary %"})
                 .style.format({"Deliveries": "{:,.0f}", "Boundaries": "{:,.0f}", "Boundary %": "{:.1f}"}),
                 width='stretch')

# ================================ ABOUT ===========================================
with tab_about:
    st.subheader("Method in brief")
    st.markdown(f"""
**Goal.** Predict whether the batter hits a boundary (4 or 6) on the next legal ball, using only information known *before* the ball.

**Pipeline.** CSV → chronological sort & legal-ball filter → leak-free features → StandardScaler + Logistic Regression
(C = {M.BEST_C}, picked by 5-fold GroupKFold grouped by match) → F1-optimal threshold ({R['best_thr']:.3f}) → evaluation on unseen seasons.

**Leakage control.** Running totals in the raw data (innings runs, batter totals) are updated after the ball, so every
match-state feature is rebuilt as *running total − current ball*. Player, bowler, phase and venue history use *earlier matches only*
and are smoothed towards the league average.

**Split.** Train 2008–2021 ({R['n_train']:,} balls, boundary rate {R['rate_train']:.2%}); test 2022–2023
({R['n_test']:,} balls, {R['rate_test']:.2%}). Chronological, because cricket drifts over seasons.

**19 features.** Match situation (8) · batter in innings (3) · behaviour history (7) · season trend (1).
""")
    st.warning("**Limitations.** One ball is very random (ROC-AUC ≈ 0.61). The dataset has no bowler type, batter handedness, "
               "field or weather. Boundary rates drift upward over seasons, so the model slightly under-predicts recent play "
               "and should be retrained periodically. The probabilities are calibrated; individual predictions are still uncertain.")
    st.caption("Kanishk Satyajit Das · Reg. no. 2428010208 · Computer and Communication Engineering, "
               "Manipal University Jaipur · Guide: Dr Sandeep Kumar Sharma")
