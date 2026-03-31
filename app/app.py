import streamlit as st
import pandas as pd
import os
import joblib

# ----------------------------
# LOAD MODEL + DATA
# ----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir,"src", "models", "rf_model.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "processed", "comfort_dataset.csv")
    return pd.read_csv(path)

@st.cache_data
def load_features():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "src", "models", "features.pkl")
    return joblib.load(path)

# ----------------------------
# ENCODING
# ----------------------------
def encode(df, reference_columns):
    df = pd.get_dummies(df)

    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[reference_columns]
    return df

# ----------------------------
# DESIGN GENERATION
# ----------------------------
def generate_designs():
    shapes = ["rectangular", "L", "H", "U", "courtyard"]
    insulations = [0, 1, 2]
    windows = [0.2, 0.4, 0.6]
    smart = [0, 1]

    designs = []
    for s in shapes:
        for i in insulations:
            for w in windows:
                for sm in smart:
                    designs.append({
                        "shape": s,
                        "insulation": i,
                        "window_ratio": w,
                        "smart_window": sm
                    })
    return designs

# ----------------------------
# CORE LOGIC
# ----------------------------
def find_best_design_yearly(model, df, city, reference_columns):
    designs = generate_designs()

    best_score = -1
    best_design = None

    for design in designs:
        scores = []

        for month in range(1, 13):
            subset = df[(df["city"] == city) & (df["month"] == month)]
            climate = subset.iloc[0]

            row = {
                "year": 2026,
                "month": month,
                "temp": climate["temp"],
                "humidity": climate["humidity"],
                "wind": climate["wind"],
                "shape": design["shape"],
                "insulation": design["insulation"],
                "window_ratio": design["window_ratio"],
                "smart_window": design["smart_window"],
                "city": city
            }

            inp = pd.DataFrame([row])
            inp = encode(inp, reference_columns)

            score = model.predict(inp)[0]
            scores.append(score)

        final_score = min(scores)  # worst-case design

        if final_score > best_score:
            best_score = final_score
            best_design = design

    return best_design, best_score

# ----------------------------
# EXPLANATION
# ----------------------------
def generate_explanation(city, score, avg_temp, avg_humidity):
    if score < 50:
        return f"{city.title()} experiences significant thermal stress due to high temperatures and/or humidity. Passive design helps, but active cooling may be required."

    elif score < 75:
        return f"{city.title()} has moderate thermal stress. High humidity ({avg_humidity:.1f}%) reduces cooling efficiency, so airflow-based designs like courtyards are effective."

    else:
        return f"{city.title()} has relatively comfortable climate conditions. Standard design works, but optimized ventilation improves comfort."

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Climate Adaptive Buildings", layout="centered")

st.title("🏢 Climate Adaptive Building Design System")

st.markdown("Select a city to get the optimal building design based on climate conditions.")

# Load everything
df = load_data()
model = load_model()
features = load_features()

# City selection
cities = sorted(df["city"].unique())
city = st.selectbox("Select City", cities)

if st.button("Get Best Design"):

    best_design, score = find_best_design_yearly(model, df, city, features)

    # compute avg climate for explanation
    city_data = df[df["city"] == city]
    avg_temp = city_data["temp"].mean()
    avg_humidity = city_data["humidity"].mean()

    st.subheader("🏆 Recommended Design")

    st.write(f"**Shape:** {best_design['shape']}")
    st.write(f"**Insulation Level:** {best_design['insulation']}")
    st.write(f"**Window Ratio:** {best_design['window_ratio']}")
    st.write(f"**Smart Windows:** {'Yes' if best_design['smart_window'] else 'No'}")

    st.subheader("📊 Comfort Score")
    st.metric("Score", round(score, 2))

    st.subheader("🧠 Explanation")
    explanation = generate_explanation(city, score, avg_temp, avg_humidity)
    st.write(explanation)