import streamlit as st
import pandas as pd
import os
import joblib

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def get_best_design_cached(city):
    return find_best_design_yearly(model, df, city, features)

def generate_explanation(city, score, avg_temp, avg_humidity, design):
    explanation = f"""
### Climate Analysis for {city.title()}

- Average Temperature: **{avg_temp:.1f}°C**
- Average Humidity: **{avg_humidity:.1f}%**

This indicates that the region experiences {'high humidity' if avg_humidity > 65 else 'moderate humidity'}, 
which significantly affects thermal comfort.

---

### Why this design?

- The **{design['shape']} layout** is recommended because this region benefits from improved airflow due to {'high humidity' if avg_humidity > 65 else 'moderate humidity'}  
- **{design['insulation']}** helps reduce heat transfer from outside  
- **{design['window']}** ensures a balance between ventilation and solar heat gain  

---

### Key Insight

Even with optimal design, climate remains the dominant factor.  
This design minimizes discomfort but may still require mechanical cooling during extreme months.
"""
    return explanation

# ----------------------------
# UI
# ----------------------------

def format_design(design):
    insulation_map = {
        0: "Low (basic wall)",
        1: "Medium (insulated wall)",
        2: "High (thick insulated wall)"
    }

    window_map = {
        0.2: "Small windows (low heat gain)",
        0.4: "Balanced windows (recommended)",
        0.6: "Large windows (high ventilation, more heat)"
    }

    shape_map = {
        "rectangular": "Rectangular (standard design)",
        "L": "L-shaped (moderate airflow)",
        "H": "H-shaped (good ventilation)",
        "U": "U-shaped (semi-enclosed airflow)",
        "courtyard": "Courtyard (maximum natural ventilation)"
    }

    return {
        "shape": shape_map[design["shape"]],
        "insulation": insulation_map[design["insulation"]],
        "window": window_map[design["window_ratio"]],
        "smart": "Yes (automatic adjustment)" if design["smart_window"] else "No"
    }

def interpret_score(score):
    if score < 40:
        return " Poor – High discomfort, active cooling required"
    elif score < 60:
        return " Moderate – Noticeable discomfort in hot months"
    elif score < 80:
        return " Good – Comfortable with proper design"
    else:
        return " Excellent – Naturally comfortable most of the year"

st.set_page_config(page_title="Climate Adaptive Buildings", layout="centered")

st.title("Climate Adaptive Building Design System")
st.markdown("### AI-powered climate-responsive building design")

st.markdown("Select a city to get the optimal building design based on climate conditions.")
st.divider()

# Load everything
df = load_data()
model = load_model()
features = load_features()

# City selection
cities = sorted(df["city"].unique())
city = st.selectbox("Select City", cities)

if st.button("Get Best Design"):

    best_design, score = get_best_design_cached(city)

    # compute avg climate for explanation
    city_data = df[df["city"] == city]
    avg_temp = city_data["temp"].mean()
    avg_humidity = city_data["humidity"].mean()

    st.markdown("## Recommended Design")

    formatted = format_design(best_design)

    st.write(f"**Shape:** {formatted['shape']}")
    st.write(f"**Insulation:** {formatted['insulation']}")
    st.write(f"**Windows:** {formatted['window']}")
    st.write(f"**Smart Windows:** {formatted['smart']}")
    st.divider()

    st.markdown("## Comfort Score")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Comfort Score", round(score, 2))

    with col2:
        st.write(interpret_score(score))
    
    st.progress(score / 100)
    st.markdown("### What this means")

    st.write(
        "This design was selected because it performs consistently across all seasons, "
        "especially during the most uncomfortable months. It avoids extreme discomfort rather than optimizing only for mild conditions."
    )

    st.divider()

    st.markdown("## Explanation")
    explanation = generate_explanation(city, score, avg_temp, avg_humidity, formatted)
    st.write(explanation)
    st.divider()