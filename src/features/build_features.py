import pandas as pd
import os
import itertools


# ---------------------------------------------------------------------------
# PHYSICAL CEILING  (unchanged — climate hard-limit)
# ---------------------------------------------------------------------------

def _passive_cooling_ceiling(temp: float) -> float:
    """Maximum achievable comfort via passive means given outdoor temp."""
    if temp <= 26:
        return 95.0
    elif temp <= 35:
        return 100.0 - 3.5 * (temp - 26)
    else:
        return max(0.0, 68.5 - 5.5 * (temp - 35))


# ---------------------------------------------------------------------------
# STEP 1 — BASELINE CLIMATE COMFORT  (temp/humidity/wind, no design)
# ---------------------------------------------------------------------------

def compute_climate_baseline(temp: float, humidity: float, wind: float) -> float:
    """
    Pure climate comfort score — no design involved.
    Temperature is the dominant driver here, exactly as intended.
    """
    score = 100.0

    # --- Temperature penalties (progressive) ---
    if temp > 24:
        score -= 3.5 * (temp - 24)
    if temp > 30:
        score -= 3.0 * (temp - 30)
    if temp > 36:
        score -= 4.0 * (temp - 36)

    # --- Humidity penalty ---
    if humidity > 60:
        score -= 1.5 * (humidity - 60)

    # --- Wind effect ---
    WIND_NEUTRAL = 35.0
    wind_capped = min(wind, 4.0)

    if temp < WIND_NEUTRAL:
        evap    = max(0.0, 1.0 - humidity / 100.0)
        cooling = (WIND_NEUTRAL - temp) / WIND_NEUTRAL
        score  += 3.2 * wind_capped * evap * cooling
    else:
        score  -= wind_capped * 0.4 * (1.0 + (temp - WIND_NEUTRAL) / 20.0)

    # --- Combined hot + humid penalty ---
    if temp > 28 and humidity > 70:
        score -= 18.0

    ceiling = _passive_cooling_ceiling(temp)
    return max(0.0, min(ceiling, score))


# ---------------------------------------------------------------------------
# STEP 2 — DESIGN BONUS  (additive, calibrated)
# ---------------------------------------------------------------------------

# How much of the remaining gap between baseline and ceiling can each
# design strategy recover?  Tune these constants to shift the balance.
#
#   design_bonus = DESIGN_INFLUENCE * gap * combined_factor
#
# DESIGN_INFLUENCE = 0.0  → design has no effect   (your current problem)
# DESIGN_INFLUENCE = 1.0  → design fully closes the comfort gap
# 0.35 gives design ~35 % of the remaining gap — meaningful but climate-dominated.

DESIGN_INFLUENCE = 0.38   # ← primary tuning knob

# Per-parameter contribution weights (all sum to 1.0 internally per category)

SHAPE_FACTORS = {
    "rectangular": 0.00,   # baseline — no bonus
    "L":           0.18,
    "H":           0.32,
    "U":           0.42,
    "courtyard":   0.60,   # best passive ventilation
}

INSULATION_FACTORS = {
    0: 0.00,
    1: 0.30,
    2: 0.60,
}

WINDOW_FACTORS = {
    # At moderate temps: larger windows capture breeze → positive
    # At high temps: larger windows add solar heat → negative offset applied separately
    0.2: 0.10,
    0.4: 0.30,
    0.6: 0.45,
}

WINDOW_HEAT_PENALTY = {
    # Extra comfort loss from solar gain through large glazing (only when hot)
    0.2: 0.0,
    0.4: 0.0,
    0.6: 0.12,   # 12 % of gap wasted back when temp > 30
}

SMART_WINDOW_BONUS = 0.20   # smart glazing adds 20 % more to window contribution


def compute_design_bonus(
    temp: float,
    humidity: float,
    baseline: float,
    ceiling: float,
    shape: str,
    insulation: int,
    window_ratio: float,
    smart_window: int,
) -> float:
    """
    Returns a comfort bonus (≥ 0) representing how much design recovers
    from the climate baseline toward the physical ceiling.

    Design CAN help even in extreme heat — just less so.
    """
    gap = max(0.0, ceiling - baseline)
    if gap < 1e-6:
        return 0.0   # already at ceiling; design can't help further

    # --- Shape: ventilation and shading geometry ---
    shape_contrib = SHAPE_FACTORS.get(shape, 0.0)

    # Courtyard / H / U shapes lose some benefit in very high humidity
    # (natural ventilation less effective when air is saturated)
    if humidity > 75:
        humidity_damp = max(0.5, 1.0 - (humidity - 75) / 50.0)
        shape_contrib *= humidity_damp

    # --- Insulation: effectiveness degrades less steeply than before ---
    # Original formula zeroed out insulation fast; here it's more gradual.
    if temp > 30:
        ins_eff = max(0.3, 1.0 - (temp - 30) / 40.0)   # still useful at 40 °C
    else:
        ins_eff = 1.0
    ins_contrib = INSULATION_FACTORS.get(insulation, 0.0) * ins_eff

    # --- Window ratio: ventilation gain vs solar penalty ---
    win_base    = WINDOW_FACTORS.get(window_ratio, 0.0)
    smart_boost = SMART_WINDOW_BONUS if smart_window else 0.0
    win_contrib = win_base * (1.0 + smart_boost)

    # Solar heat penalty for large windows when it's hot
    if temp > 30:
        solar_pen    = WINDOW_HEAT_PENALTY.get(window_ratio, 0.0)
        heat_factor  = min(1.0, (temp - 30) / 15.0)   # ramps 30 → 45 °C
        win_contrib -= solar_pen * heat_factor          # can reduce contribution
        win_contrib  = max(0.0, win_contrib)

    # --- Combine: geometric mean to avoid any single parameter dominating ---
    # (arithmetic mean inflates result when one factor is high but others low)
    n = 3
    combined = (
        (shape_contrib + 0.01) *
        (ins_contrib   + 0.01) *
        (win_contrib   + 0.01)
    ) ** (1.0 / n)

    # Normalise so the best possible combination ≈ 1.0
    # Best: courtyard(0.60) * ins2(0.60) * win0.6+smart(0.45*1.20=0.54)
    # geometric mean of (0.61 * 0.61 * 0.55)^(1/3) ≈ 0.589
    MAX_COMBINED = (0.61 * 0.61 * 0.55) ** (1.0 / 3.0)
    normalised   = min(1.0, combined / MAX_COMBINED)

    bonus = DESIGN_INFLUENCE * gap * normalised
    return bonus


# ---------------------------------------------------------------------------
# STEP 3 — FINAL COMFORT  (climate baseline + design bonus)
# ---------------------------------------------------------------------------

def compute_comfort(
    temp: float,
    humidity: float,
    wind: float,
    shape: str,
    insulation: int,
    window_ratio: float,
    smart_window: int,
) -> float:
    """
    Full comfort score combining climate baseline and design bonus.

    Architecture:
        baseline  = f(temp, humidity, wind)          climate dominates
        bonus     = g(design params, gap, ceiling)   design meaningfully adjusts
        comfort   = clip(baseline + bonus, 0, ceiling)
    """
    ceiling  = _passive_cooling_ceiling(temp)
    baseline = compute_climate_baseline(temp, humidity, wind)

    bonus    = compute_design_bonus(
        temp, humidity, baseline, ceiling,
        shape, insulation, window_ratio, smart_window
    )

    return round(max(0.0, min(ceiling, baseline + bonus)), 6)


# ---------------------------------------------------------------------------
# DATASET GENERATION
# ---------------------------------------------------------------------------

def generate_dataset() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    input_path  = os.path.join(base_dir, "data", "processed", "monthly_climate.csv")
    output_path = os.path.join(base_dir, "data", "processed", "comfort_dataset.csv")

    df = pd.read_csv(input_path)

    shapes       = ["rectangular", "L", "H", "U", "courtyard"]
    insulations  = [0, 1, 2]
    window_ratios = [0.2, 0.4, 0.6]
    smart_windows = [0, 1]

    results = []

    for _, row in df.iterrows():
        for shape, ins, win, smart in itertools.product(
            shapes, insulations, window_ratios, smart_windows
        ):
            comfort = compute_comfort(
                row["temp"], row["humidity"], row["wind"],
                shape, ins, win, bool(smart)
            )

            results.append({
                "city":         row["city"],
                "year":         row["year"],
                "month":        row["month"],
                "temp": row["temp"],
                "humidity": row["humidity"],
                "wind": row["wind"],
                "shape":        shape,
                "insulation":   ins,
                "window_ratio": win,
                "smart_window": smart,
                "comfort":      comfort,
            })

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    return final_df


# ---------------------------------------------------------------------------
# ANALYSIS HELPERS  (unchanged API)
# ---------------------------------------------------------------------------

def get_best_per_month(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(["city", "year", "month"])["comfort"].idxmax()]


def comfort_range_per_city(best_df: pd.DataFrame) -> pd.DataFrame:
    summary = best_df.groupby("city")["comfort"].agg(["min", "max", "mean"])
    return summary.sort_values("min")


def design_performance(best_df: pd.DataFrame) -> pd.Series:
    return best_df.groupby("shape")["comfort"].mean().sort_values(ascending=False)


def window_analysis(best_df: pd.DataFrame) -> pd.Series:
    return best_df.groupby("window_ratio")["comfort"].mean().sort_values(ascending=False)


def smart_window_analysis(best_df: pd.DataFrame) -> pd.Series:
    return best_df.groupby("smart_window")["comfort"].mean()


def worst_month_per_city(best_df: pd.DataFrame) -> pd.DataFrame:
    worst = best_df.loc[
        best_df.groupby(["city", "year"])["comfort"].idxmin()
    ]
    return worst.sort_values(["city", "year"])


def ahmedabad_extremes(best_df: pd.DataFrame) -> None:
    df = best_df[best_df["city"] == "ahmedabad"]
    print("\n🔥 Worst months (Ahmedabad):")
    print(df.sort_values("comfort").head(5)[["year", "month", "comfort", "shape", "window_ratio", "smart_window"]])
    print("\n❄️ Best months (Ahmedabad):")
    print(df.sort_values("comfort", ascending=False).head(5)[["year", "month", "comfort", "shape", "window_ratio", "smart_window"]])


def compare_city_months(best_df: pd.DataFrame, city: str) -> None:
    city_df = best_df[best_df["city"] == city].sort_values(["year", "month"])
    print(f"\n📍 {city.title()} — first 12 months:")
    print(city_df[["year", "month", "shape", "window_ratio", "smart_window", "comfort"]].head(12).to_string())

def fair_design_comparison(df: pd.DataFrame, city: str, year: int, month: int) -> pd.DataFrame:
    """
    Compare all design options under the SAME climate condition.
    This removes climate bias and shows true design effectiveness.
    """
    subset = df[
        (df["city"] == city) &
        (df["year"] == year) &
        (df["month"] == month)
    ]

    return subset.sort_values("comfort", ascending=False)

def print_design_comparison(df: pd.DataFrame, city: str, year: int, month: int) -> None:
    print(f"\n🔍 Design comparison — {city.title()} {year} Month {month}")
    comparison = fair_design_comparison(df, city, year, month)
    print(comparison.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# DIAGNOSTIC: show design spread for a single climate month
# ---------------------------------------------------------------------------

def design_spread_diagnostic(temp: float, humidity: float, wind: float) -> None:
    """
    Print comfort for every design combination at fixed climate inputs.
    Use this to verify design differentiation is meaningful.
    """
    shapes        = ["rectangular", "L", "H", "U", "courtyard"]
    insulations   = [0, 1, 2]
    window_ratios = [0.2, 0.4, 0.6]
    smart_windows = [0, 1]

    rows = []
    for shape, ins, win, smart in itertools.product(shapes, insulations, window_ratios, smart_windows):
        c = compute_comfort(temp, humidity, wind, shape, ins, win, bool(smart))
        rows.append({"shape": shape, "insulation": ins, "window_ratio": win, "smart_window": smart, "comfort": c})

    diag = pd.DataFrame(rows).sort_values("comfort", ascending=False)
    baseline = compute_climate_baseline(temp, humidity, wind)
    ceiling  = _passive_cooling_ceiling(temp)

    print(f"\n🌡  Climate: temp={temp}°C  humidity={humidity}%  wind={wind} m/s")
    print(f"   Baseline (no design): {baseline:.2f}   |   Ceiling: {ceiling:.2f}")
    print(f"   Design spread: {diag['comfort'].min():.2f} – {diag['comfort'].max():.2f}  "
          f"(Δ = {diag['comfort'].max() - diag['comfort'].min():.2f})\n")
    print(diag.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Sanity check before generating the full dataset ---
    print("=" * 60)
    print("DESIGN SPREAD DIAGNOSTICS")
    print("=" * 60)
    design_spread_diagnostic(temp=38, humidity=55, wind=2.0)   # hot dry
    design_spread_diagnostic(temp=32, humidity=80, wind=1.5)   # hot humid
    design_spread_diagnostic(temp=25, humidity=65, wind=3.0)   # mild

    # --- Generate full dataset ---
    print("\n" + "=" * 60)
    print("GENERATING DATASET …")
    print("=" * 60)
    df      = generate_dataset()
    best_df = get_best_per_month(df)

    print("\n--- Best design per month (head) ---")
    print(best_df.head())

    print("\n--- Shape distribution in best designs ---")
    print(best_df["shape"].value_counts())

    print("\n--- Window ratio distribution in best designs ---")
    print(best_df["window_ratio"].value_counts())

    print("\n--- Smart window distribution in best designs ---")
    print(best_df["smart_window"].value_counts())

    compare_city_months(best_df, "ahmedabad")
    compare_city_months(best_df, "chennai")

    print("\n--- Comfort range per city ---")
    print(comfort_range_per_city(best_df))

    print("\n--- Design performance (shape) ---")
    print(design_performance(best_df))

    print("\n--- Window ratio performance ---")
    print(window_analysis(best_df))

    print("\n--- Smart window performance ---")
    print(smart_window_analysis(best_df))

    ahmedabad_extremes(best_df)

    print_design_comparison(df, "ahmedabad", 1981, 5)  # peak summer
    print_design_comparison(df, "chennai", 1981, 8)    # humid month