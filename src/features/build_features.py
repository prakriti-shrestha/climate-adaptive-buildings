import pandas as pd
import os
import itertools


# ---------------------------------------------------------------------------
# PHYSICAL CEILING  (climate hard-limit — unchanged)
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
# STEP 1 — BASELINE CLIMATE COMFORT
# ---------------------------------------------------------------------------

def compute_climate_baseline(temp: float, humidity: float, wind: float) -> float:
    score = 100.0

    if temp > 24:
        score -= 3.5 * (temp - 24)
    if temp > 30:
        score -= 3.0 * (temp - 30)
    if temp > 36:
        score -= 4.0 * (temp - 36)

    if humidity > 60:
        score -= 1.5 * (humidity - 60)

    WIND_NEUTRAL = 35.0
    wind_capped = min(wind, 4.0)

    if temp < WIND_NEUTRAL:
        evap    = max(0.0, 1.0 - humidity / 100.0)
        cooling = (WIND_NEUTRAL - temp) / WIND_NEUTRAL
        score  += 3.2 * wind_capped * evap * cooling
    else:
        score  -= wind_capped * 0.4 * (1.0 + (temp - WIND_NEUTRAL) / 20.0)

    if temp > 28 and humidity > 70:
        score -= 18.0

    ceiling = _passive_cooling_ceiling(temp)
    return max(0.0, min(ceiling, score))


# ---------------------------------------------------------------------------
# STEP 2 — CLIMATE-SENSITIVE SHAPE FACTORS
#
# KEY FIX: Shape effectiveness depends on the climate regime.
#
# Regime classification:
#   hot_dry   : temp > 30, humidity < 50  → Ahmedabad summer, Delhi summer
#   hot_humid : temp > 28, humidity > 65  → Chennai, Mumbai, Kolkata
#   cold      : temp < 16                 → Delhi winter
#   mild      : otherwise                 → Bangalore, transitional months
#
# Physical rationale:
#   Courtyard  — radiant cooling at night is powerful in dry air; but traps
#                stagnant humid air, making it worse than H/U in humid climates.
#   H / U      — dual-wing geometries channel prevailing wind through the core;
#                most effective when humidity blunts the courtyard stack effect.
#   L          — asymmetric exposure, good in mild climates with variable wind.
#   Rectangular — compact envelope; best in cold conditions to minimise surface
#                area and heat loss.
# ---------------------------------------------------------------------------

def _shape_factor(shape: str, temp: float, humidity: float) -> float:
    """
    Returns a [0, 1] shape effectiveness factor that varies with climate.
    """
    # ── Classify the thermal regime ──────────────────────────────────────
    hot_dry   = temp > 30 and humidity < 50
    hot_humid = temp > 28 and humidity > 65
    mild_warm = 20 <= temp <= 30 and humidity <= 65
    cold      = temp < 16

    if hot_dry:
        # Courtyard stack-effect thrives; insulation important; glazing careful
        factors = {
            "courtyard":   0.72,
            "U":           0.52,
            "H":           0.42,
            "L":           0.24,
            "rectangular": 0.10,
        }
    elif hot_humid:
        # Courtyard traps humid air — H/U cross-ventilation wins here
        # Humidity damping applied to enclosed shapes
        hum_damp = max(0.6, 1.0 - (humidity - 65) / 60.0)
        factors = {
            "H":           0.68,
            "U":           0.60 * hum_damp,
            "courtyard":   0.44 * hum_damp,   # penalised for stagnant air trap
            "L":           0.36,
            "rectangular": 0.18,
        }
    elif cold:
        # Compact envelope minimises heat loss; courtyard exposed to cold wind
        factors = {
            "rectangular": 0.55,
            "L":           0.40,
            "H":           0.30,
            "U":           0.22,
            "courtyard":   0.12,   # worst in cold — exposed, loses heat rapidly
        }
    elif mild_warm:
        # Moderate climate — all shapes viable, courtyard still good
        factors = {
            "courtyard":   0.58,
            "U":           0.50,
            "H":           0.44,
            "L":           0.32,
            "rectangular": 0.18,
        }
    else:
        # Transitional / default
        factors = {
            "courtyard":   0.52,
            "U":           0.46,
            "H":           0.40,
            "L":           0.28,
            "rectangular": 0.14,
        }

    return factors.get(shape, 0.10)


# ---------------------------------------------------------------------------
# STEP 3 — CLIMATE-SENSITIVE INSULATION FACTORS
#
# In cold climates insulation is critical (traps warmth).
# In hot-humid climates it helps but ventilation is more important.
# In hot-dry climates it delays heat ingress — highly effective.
# ---------------------------------------------------------------------------

def _insulation_factor(insulation: int, temp: float) -> float:
    base = {0: 0.00, 1: 0.30, 2: 0.60}.get(insulation, 0.0)

    if temp < 16:
        # Cold: insulation very important, scale up
        eff = 1.4
    elif temp > 30:
        # Hot: insulation still degrades but less than before
        eff = max(0.35, 1.0 - (temp - 30) / 40.0)
    else:
        eff = 1.0

    return min(1.0, base * eff)


# ---------------------------------------------------------------------------
# STEP 4 — CLIMATE-SENSITIVE WINDOW FACTORS
# ---------------------------------------------------------------------------

WINDOW_BASE = {0.2: 0.10, 0.4: 0.30, 0.6: 0.45}

WINDOW_HEAT_PENALTY = {
    # Solar gain penalty for large windows in hot conditions
    0.2: 0.00,
    0.4: 0.00,
    0.6: 0.14,
}

SMART_WINDOW_BONUS = 0.22


def _window_factor(
    window_ratio: float,
    smart_window: int,
    temp: float,
    humidity: float,
) -> float:
    win = WINDOW_BASE.get(window_ratio, 0.0)
    smart_boost = SMART_WINDOW_BONUS if smart_window else 0.0
    win_contrib = win * (1.0 + smart_boost)

    # Solar heat penalty when hot
    if temp > 30:
        solar_pen   = WINDOW_HEAT_PENALTY.get(window_ratio, 0.0)
        heat_factor = min(1.0, (temp - 30) / 15.0)
        win_contrib -= solar_pen * heat_factor
        win_contrib  = max(0.0, win_contrib)

    # In cold climates, larger windows increase heat loss
    if temp < 16:
        cold_pen     = {0.2: 0.0, 0.4: 0.05, 0.6: 0.15}.get(window_ratio, 0.0)
        cold_factor  = min(1.0, (16 - temp) / 12.0)
        win_contrib -= cold_pen * cold_factor * (0 if smart_window else 1)  # smart glass mitigates
        win_contrib  = max(0.0, win_contrib)

    return win_contrib


# ---------------------------------------------------------------------------
# DESIGN INFLUENCE & BONUS  (climate-aware combination)
# ---------------------------------------------------------------------------

DESIGN_INFLUENCE = 0.40   # design recovers ~40% of the gap between baseline and ceiling


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
    gap = max(0.0, ceiling - baseline)
    if gap < 1e-6:
        return 0.0

    shape_contrib = _shape_factor(shape, temp, humidity)
    ins_contrib   = _insulation_factor(insulation, temp)
    win_contrib   = _window_factor(window_ratio, smart_window, temp, humidity)

    # Geometric mean — prevents a single strong parameter from dominating
    combined = (
        (shape_contrib + 0.01) *
        (ins_contrib   + 0.01) *
        (win_contrib   + 0.01)
    ) ** (1.0 / 3.0)

    # Normalise: best hot-dry case courtyard(0.72) * ins2(0.60*1.0=0.60) * win0.6+smart
    # win = 0.45 * 1.22 = 0.549; geometric mean of (0.73 * 0.61 * 0.559)^(1/3) ≈ 0.624
    MAX_COMBINED = (0.73 * 0.61 * 0.559) ** (1.0 / 3.0)
    normalised   = min(1.0, combined / MAX_COMBINED)

    return DESIGN_INFLUENCE * gap * normalised


# ---------------------------------------------------------------------------
# FULL COMFORT SCORE
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

    shapes        = ["rectangular", "L", "H", "U", "courtyard"]
    insulations   = [0, 1, 2]
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
                "temp":         row["temp"],
                "humidity":     row["humidity"],
                "wind":         row["wind"],
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
# ANALYSIS HELPERS
# ---------------------------------------------------------------------------

def get_best_per_month(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(["city", "year", "month"])["comfort"].idxmax()]


def comfort_range_per_city(best_df: pd.DataFrame) -> pd.DataFrame:
    return best_df.groupby("city")["comfort"].agg(["min", "max", "mean"]).sort_values("min")


def design_performance(best_df: pd.DataFrame) -> pd.Series:
    return best_df.groupby("shape")["comfort"].mean().sort_values(ascending=False)


def window_analysis(best_df: pd.DataFrame) -> pd.Series:
    return best_df.groupby("window_ratio")["comfort"].mean().sort_values(ascending=False)


def smart_window_analysis(best_df: pd.DataFrame) -> pd.Series:
    return best_df.groupby("smart_window")["comfort"].mean()


def worst_month_per_city(best_df: pd.DataFrame) -> pd.DataFrame:
    worst = best_df.loc[best_df.groupby(["city", "year"])["comfort"].idxmin()]
    return worst.sort_values(["city", "year"])


# ---------------------------------------------------------------------------
# DIAGNOSTIC — verify climate-driven design differentiation
# ---------------------------------------------------------------------------

def design_spread_diagnostic(label: str, temp: float, humidity: float, wind: float) -> None:
    """
    Print comfort for every design combo at fixed climate inputs.
    Use this to verify that different climate regimes yield different optimal shapes.
    """
    shapes        = ["rectangular", "L", "H", "U", "courtyard"]
    insulations   = [0, 1, 2]
    window_ratios = [0.2, 0.4, 0.6]
    smart_windows = [0, 1]

    rows = []
    for shape, ins, win, smart in itertools.product(shapes, insulations, window_ratios, smart_windows):
        c = compute_comfort(temp, humidity, wind, shape, ins, win, bool(smart))
        rows.append({"shape": shape, "insulation": ins, "window_ratio": win, "smart_window": smart, "comfort": c})

    diag     = pd.DataFrame(rows).sort_values("comfort", ascending=False)
    baseline = compute_climate_baseline(temp, humidity, wind)
    ceiling  = _passive_cooling_ceiling(temp)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  temp={temp}°C  humidity={humidity}%  wind={wind} m/s")
    print(f"  Baseline: {baseline:.2f}  |  Ceiling: {ceiling:.2f}")
    print(f"  Design spread: {diag['comfort'].min():.2f} – {diag['comfort'].max():.2f}  "
          f"(Δ = {diag['comfort'].max() - diag['comfort'].min():.2f})")
    print(f"  🏆 Best shape: {diag.iloc[0]['shape']}  "
          f"(insulation={diag.iloc[0]['insulation']}, "
          f"window={diag.iloc[0]['window_ratio']}, "
          f"smart={diag.iloc[0]['smart_window']})")
    print(diag[["shape","insulation","window_ratio","smart_window","comfort"]].head(6).to_string(index=False))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CLIMATE-SENSITIVE DESIGN SPREAD DIAGNOSTICS")
    print("Expected: DIFFERENT optimal shapes per regime")
    print("=" * 60)

    # Hot-dry (Ahmedabad peak summer) → expect COURTYARD
    design_spread_diagnostic("HOT-DRY  (Ahmedabad May)", temp=40, humidity=35, wind=2.5)

    # Hot-humid (Chennai August) → expect H or U, NOT courtyard
    design_spread_diagnostic("HOT-HUMID (Chennai Aug)", temp=32, humidity=82, wind=1.5)

    # Cold (Delhi January) → expect RECTANGULAR
    design_spread_diagnostic("COLD (Delhi Jan)", temp=12, humidity=75, wind=1.0)

    # Mild (Bangalore) → expect courtyard or U
    design_spread_diagnostic("MILD (Bangalore Mar)", temp=26, humidity=55, wind=2.8)

    # Hot-humid coastal (Mumbai June) → expect H
    design_spread_diagnostic("HOT-HUMID COASTAL (Mumbai Jun)", temp=30, humidity=88, wind=3.5)

    print("\n" + "=" * 60)
    print("GENERATING FULL DATASET ...")
    print("=" * 60)
    df      = generate_dataset()
    best_df = get_best_per_month(df)

    print("\n--- Shape distribution in best designs (should be DIVERSE) ---")
    print(best_df["shape"].value_counts())

    print("\n--- Best shape per city (should differ by city) ---")
    city_shapes = best_df.groupby("city")["shape"].agg(lambda x: x.value_counts().index[0])
    print(city_shapes)

    print("\n--- Comfort range per city ---")
    print(comfort_range_per_city(best_df))