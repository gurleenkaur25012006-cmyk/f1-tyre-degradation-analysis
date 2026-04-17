import fastf1
import fastf1.plotting as fplot
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def load_session(year: int, gp: str, session_type: str):
    """Load and return a FastF1 session object."""
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def clean_laps(session) -> pd.DataFrame:
    """Filter laps to only accurate, clean race laps."""
    laps = session.laps
    laps_cleaned = laps.loc[
        (laps["IsAccurate"] == True)
        & (laps["LapTime"].notna())
        & (laps["PitOutTime"].isna())
        & (laps["PitInTime"].isna())
        & (laps["TrackStatus"] == "1")
    ].copy()
    laps_cleaned["LapTimeSeconds"] = laps_cleaned["LapTime"].dt.total_seconds()
    return laps_cleaned


def compute_degradation(laps_cleaned: pd.DataFrame, session) -> pd.DataFrame:
    """
    For each driver-stint with >=5 laps, run linear regression on lap time vs stint lap.
    Returns a DataFrame with columns: Driver, Stint, Compound, Degradation(ms/lap), R2.
    Only includes stints with R2 >= 0.3 and positive slope (actual degradation).
    """
    drivers = pd.unique(laps_cleaned["Driver"])
    degradation = []

    for driver in drivers:
        driver_laps = laps_cleaned.loc[laps_cleaned["Driver"] == driver]
        for stint in driver_laps["Stint"].unique():
            stint_laps = driver_laps.loc[driver_laps["Stint"] == stint].copy()
            if len(stint_laps) < 5:
                continue
            compound = stint_laps["Compound"].iloc[0]
            stint_laps["StintLapNumber"] = (
                stint_laps["LapNumber"] - stint_laps["LapNumber"].min() + 1
            )
            slope, intercept, r_value, p_value, std_err = linregress(
                stint_laps["StintLapNumber"], stint_laps["LapTimeSeconds"]
            )
            if r_value ** 2 < 0.3 or slope <= 0:
                continue
            degradation.append(
                {
                    "Driver": driver,
                    "Stint": stint,
                    "Compound": compound,
                    "Degradation(ms/lap)": slope * 1000,
                    "R2": r_value ** 2,
                    "_slope": slope,
                    "_intercept": intercept,
                }
            )

    df = pd.DataFrame(degradation)
    if df.empty:
        return pd.DataFrame(columns=[
            "Driver", "Stint", "Compound",
            "Degradation(ms/lap)", "R2", "_slope", "_intercept"
        ])

    return df


def compute_driver_summary(degradation_df: pd.DataFrame, laps_cleaned: pd.DataFrame):
    """Returns driver_summary (mean degradation) and consistency (lap time std)."""
    driver_summary = degradation_df.groupby("Driver")["Degradation(ms/lap)"].mean()
    consistency = laps_cleaned.groupby("Driver")["LapTimeSeconds"].std()
    return driver_summary, consistency


def compute_final_ranking(driver_summary: pd.Series, consistency: pd.Series) -> pd.DataFrame:
    """Rank drivers by combined degradation + consistency score (lower = better)."""
    final = pd.DataFrame(
        {"Degradation": driver_summary, "Consistency": consistency}
    ).dropna()
    final["Score"] = final["Degradation"].rank() + final["Consistency"].rank()
    return final.sort_values("Score")


def classify_drivers(driver_summary: pd.Series, pace: pd.Series) -> pd.DataFrame:
    """Label drivers as Smooth or Aggressive based on median degradation."""
    comparison = pd.DataFrame({"Pace": pace, "Degradation": driver_summary}).dropna()
    comparison["Type"] = comparison["Degradation"].apply(
        lambda x: "Smooth" if x < comparison["Degradation"].median() else "Aggressive"
    )
    return comparison


def compute_compound_summary(degradation_df: pd.DataFrame) -> pd.Series:
    return degradation_df.groupby("Compound")["Degradation(ms/lap)"].mean()


def compute_stint_recommendations(compound_summary: pd.Series, threshold_ms: int = 1000) -> dict:
    """Estimate max recommended laps per compound before losing ~threshold_ms."""
    recs = {}
    for compound, deg in compound_summary.items():
        if deg > 0:
            recs[compound] = int(threshold_ms / deg)
    return recs


def predict_future_laps(laps_cleaned: pd.DataFrame, laps_ahead: int = 3) -> pd.DataFrame:
    """Project lap times for each driver-stint laps_ahead laps into the future."""
    drivers = pd.unique(laps_cleaned["Driver"])
    predictions = []

    for driver in drivers:
        driver_laps = laps_cleaned[laps_cleaned["Driver"] == driver]
        for stint in driver_laps["Stint"].unique():
            stint_laps = driver_laps[driver_laps["Stint"] == stint].copy()
            if len(stint_laps) < 5:
                continue
            stint_laps["StintLapNumber"] = (
                stint_laps["LapNumber"] - stint_laps["LapNumber"].min() + 1
            )
            slope, intercept, _, _, _ = linregress(
                stint_laps["StintLapNumber"], stint_laps["LapTimeSeconds"]
            )
            last_lap = stint_laps["StintLapNumber"].max()
            for i in range(1, laps_ahead + 1):
                future_lap = last_lap + i
                predictions.append(
                    {
                        "Driver": driver,
                        "Stint": stint,
                        "FutureLap": future_lap,
                        "PredictedLapTime": intercept + slope * future_lap,
                    }
                )

    return pd.DataFrame(predictions)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_degradation_lines(laps_cleaned: pd.DataFrame, degradation_df: pd.DataFrame, session, gp: str, year: int):
    compound_colors = {
        c: fastf1.plotting.get_compound_color(c, session=session)
        for c in laps_cleaned["Compound"].unique()
    }
    fig, ax = plt.subplots(figsize=(18, 8))

    for _, row in degradation_df.iterrows():
        driver = row["Driver"]
        stint = row["Stint"]
        compound = row["Compound"]
        color = compound_colors.get(compound, "grey")

        stint_laps = laps_cleaned[
            (laps_cleaned["Driver"] == driver) & (laps_cleaned["Stint"] == stint)
        ].copy()
        stint_laps["StintLapNumber"] = (
            stint_laps["LapNumber"] - stint_laps["LapNumber"].min() + 1
        )

        ax.plot(
            stint_laps["StintLapNumber"],
            stint_laps["LapTimeSeconds"],
            marker="o",
            linestyle="-",
            color=color,
            alpha=0.7,
        )
        ax.plot(
            stint_laps["StintLapNumber"],
            row["_intercept"] + row["_slope"] * stint_laps["StintLapNumber"],
            linestyle="--",
            color=color,
            alpha=0.4,
        )

    legend_handles = [
        mpatches.Patch(color=v, label=k) for k, v in compound_colors.items()
    ]
    ax.legend(handles=legend_handles, title="Compound")
    ax.set_title(f"Tyre Degradation — {gp} GP {year}")
    ax.set_xlabel("Laps in Stint")
    ax.set_ylabel("Lap Time (s)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    return fig


def plot_driver_barplot(degradation_df: pd.DataFrame, top_n: int = 8):
    top_drivers = degradation_df["Driver"].value_counts().head(top_n).index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=degradation_df[degradation_df["Driver"].isin(top_drivers)],
        x="Driver",
        y="Degradation(ms/lap)",
        hue="Compound",
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Drivers: Tyre Degradation by Compound")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_compound_boxplot(degradation_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=degradation_df, x="Compound", y="Degradation(ms/lap)", ax=ax)
    ax.set_title("Degradation Distribution by Compound")
    plt.tight_layout()
    return fig


def plot_driver_scatter(comparison: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"Smooth": "#2ecc71", "Aggressive": "#e74c3c"}
    sns.scatterplot(
        data=comparison,
        x="Degradation",
        y="Pace",
        hue="Type",
        palette=palette,
        s=120,
        ax=ax,
    )
    for driver in comparison.index:
        ax.text(
            comparison.loc[driver, "Degradation"] + 0.3,
            comparison.loc[driver, "Pace"],
            driver,
            fontsize=8,
        )
    ax.set_xlabel("Degradation (ms/lap)")
    ax.set_ylabel("Average Lap Time (s)")
    ax.set_title("Driver Classification: Smooth vs Aggressive")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_prediction(pred_df: pd.DataFrame, driver: str):
    sample = pred_df[pred_df["Driver"] == driver]
    fig, ax = plt.subplots(figsize=(8, 5))
    for stint, grp in sample.groupby("Stint"):
        ax.plot(grp["FutureLap"], grp["PredictedLapTime"], marker="o", label=f"Stint {int(stint)}")
    ax.set_title(f"Future Lap Prediction — {driver}")
    ax.set_xlabel("Future Lap Number")
    ax.set_ylabel("Predicted Lap Time (s)")
    ax.legend(title="Stint")
    ax.grid(True)
    plt.tight_layout()
    return fig
