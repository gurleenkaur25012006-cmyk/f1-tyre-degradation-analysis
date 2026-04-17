import streamlit as st
import fastf1
import pandas as pd
import analysis
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Tyre Degradation Analyser",
    page_icon="🏎️",
    layout="wide",
)

os.makedirs("fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("fastf1_cache")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏎️ F1 Tyre Analyser")
    st.markdown("---")

    year = st.number_input("Season", min_value=2018, max_value=2025, value=2023, step=1)
    gp = st.text_input("Grand Prix", value="Bahrain")
    session_type = st.selectbox(
        "Session",
        options=["R", "Q", "FP1", "FP2", "FP3"],
        format_func=lambda x: {
            "R": "Race", "Q": "Qualifying",
            "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"
        }[x],
    )

    load_btn = st.button("🔄 Load Session", use_container_width=True)
    st.markdown("---")
    st.caption("Data via FastF1 · Ergast API")

# ── Session loading (cached) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading session data…")
def get_session_data(year, gp, session_type):
    session = analysis.load_session(year, gp, session_type)
    laps_cleaned = analysis.clean_laps(session)
    degradation_df = analysis.compute_degradation(laps_cleaned, session)
    driver_summary, consistency = analysis.compute_driver_summary(degradation_df, laps_cleaned)
    final = analysis.compute_final_ranking(driver_summary, consistency)
    pace = laps_cleaned.groupby("Driver")["LapTimeSeconds"].mean()
    comparison = analysis.classify_drivers(driver_summary, pace)
    compound_summary = analysis.compute_compound_summary(degradation_df)
    recommendations = analysis.compute_stint_recommendations(compound_summary)
    pred_df = analysis.predict_future_laps(laps_cleaned)
    return session, laps_cleaned, degradation_df, driver_summary, consistency, final, comparison, compound_summary, recommendations, pred_df


# ── Main content ──────────────────────────────────────────────────────────────
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if load_btn:
    st.session_state.loaded = True
    st.session_state.params = (year, gp, session_type)

if not st.session_state.loaded:
    st.title("F1 Tyre Degradation Analyser")
    st.info("👈 Enter a season, Grand Prix, and session in the sidebar, then click **Load Session**.")
    st.stop()

# Load data
(
    session, laps_cleaned, degradation_df, driver_summary,
    consistency, final, comparison, compound_summary,
    recommendations, pred_df,
) = get_session_data(*st.session_state.params)

year_s, gp_s, sess_s = st.session_state.params

st.title(f"🏁 {gp_s} GP {year_s} — {sess_s}")
st.markdown("---")

# ── Top KPI row ───────────────────────────────────────────────────────────────
best_driver = driver_summary.idxmin()
worst_driver = driver_summary.idxmax()
best_compound = compound_summary.idxmin()
top_overall = final.index[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("👑 Best Tyre Manager", best_driver, f"{driver_summary[best_driver]:.1f} ms/lap")
k2.metric("🔥 Highest Degradation", worst_driver, f"{driver_summary[worst_driver]:.1f} ms/lap")
k3.metric("🏆 Best Overall (Deg + Consistency)", top_overall)
k4.metric("🟢 Least Degrading Compound", best_compound, f"{compound_summary[best_compound]:.1f} ms/lap")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Degradation Lines",
    "📊 Driver Comparison",
    "🎯 Driver Classification",
    "📅 Strategy",
    "🔮 Predictions",
])

# ── Tab 1: Degradation line plot ──────────────────────────────────────────────
with tab1:
    st.subheader("Tyre Degradation by Stint")
    st.caption("Only stints with ≥5 laps and R² ≥ 0.3 are shown. Dashed lines = linear fit.")
    fig = analysis.plot_degradation_lines(laps_cleaned, degradation_df, session, gp_s, year_s)
    st.pyplot(fig)

    with st.expander("📋 Raw Degradation Data"):
        st.dataframe(
            degradation_df[["Driver", "Stint", "Compound", "Degradation(ms/lap)", "R2"]]
            .sort_values("Degradation(ms/lap)")
            .reset_index(drop=True)
            .style.format({"Degradation(ms/lap)": "{:.2f}", "R2": "{:.3f}"}),
            use_container_width=True,
        )

# ── Tab 2: Driver barplot + compound boxplot ───────────────────────────────────
with tab2:
    st.subheader("Driver Degradation by Compound")
    top_n = st.slider("Number of drivers to show", 4, 15, 8)
    fig2 = analysis.plot_driver_barplot(degradation_df, top_n=top_n)
    st.pyplot(fig2)

    st.subheader("Compound Degradation Distribution")
    fig3 = analysis.plot_compound_boxplot(degradation_df)
    st.pyplot(fig3)

# ── Tab 3: Driver classification scatter ──────────────────────────────────────
with tab3:
    st.subheader("Smooth vs Aggressive Drivers")
    st.caption("Classified by whether degradation is below (Smooth) or above (Aggressive) the median.")
    fig4 = analysis.plot_driver_scatter(comparison)
    st.pyplot(fig4)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Full Rankings (Deg + Consistency)")
        st.dataframe(
            final[["Degradation", "Consistency", "Score"]]
            .style.format("{:.3f}")
            .background_gradient(subset=["Score"], cmap="RdYlGn_r"),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Driver Types")
        st.dataframe(comparison[["Type", "Pace", "Degradation"]].style.format({
            "Pace": "{:.3f}", "Degradation": "{:.2f}"
        }), use_container_width=True)

# ── Tab 4: Strategy recommendations ──────────────────────────────────────────
with tab4:
    st.subheader("Recommended Stint Lengths")
    st.caption("Estimated laps before degradation accumulates to ~1 second of lap time loss.")

    rec_col, summary_col = st.columns(2)

    with rec_col:
        rec_rows = [{"Compound": c, "Max Recommended Laps": l} for c, l in recommendations.items()]
        st.dataframe(pd.DataFrame(rec_rows).set_index("Compound"), use_container_width=True)

    with summary_col:
        st.subheader("Compound Summary (avg ms/lap)")
        st.dataframe(
            compound_summary.rename("Avg Degradation (ms/lap)")
            .to_frame()
            .style.format("{:.2f}")
            .background_gradient(cmap="RdYlGn_r"),
            use_container_width=True,
        )

    st.subheader("Driver Strategies Used")
    strategy = laps_cleaned.groupby("Driver")["Compound"].apply(
        lambda x: " → ".join(pd.unique(x))
    )
    st.dataframe(strategy.rename("Compounds Used").to_frame(), use_container_width=True)

# ── Tab 5: Predictions ────────────────────────────────────────────────────────
with tab5:
    st.subheader("Future Lap Time Predictions")
    st.caption("Linear extrapolation from each driver's degradation trend.")

    available_drivers = sorted(pred_df["Driver"].unique())
    selected_driver = st.selectbox("Select Driver", available_drivers)

    fig5 = analysis.plot_prediction(pred_df, selected_driver)
    st.pyplot(fig5)

    with st.expander("📋 Prediction Table"):
        st.dataframe(
            pred_df[pred_df["Driver"] == selected_driver]
            .style.format({"PredictedLapTime": "{:.3f}"}),
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with FastF1 · Streamlit · Seaborn | Data: Ergast / OpenF1")
