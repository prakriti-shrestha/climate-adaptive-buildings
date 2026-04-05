import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor

# LOAD MODEL + DATA
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "src/models", "rf_model.pkl")

    return joblib.load(model_path)

def load_features():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, "src/models", "features.pkl")
    return joblib.load(path)


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, "data", "processed", "comfort_dataset.csv")

    return pd.read_csv(path)

# PREPROCESS SINGLE INPUT
def prepare_input(df, city, month, design):
    # get climate for that city + month (average)
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

    return pd.DataFrame([row])

# ENCODING (same as training)
def encode(df, reference_columns):
    df = pd.get_dummies(df)

    # add missing columns
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[reference_columns]

    return df

# GENERATE ALL DESIGN OPTIONS
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

# FIND BEST DESIGN
def find_best_design_yearly(model, df, city, reference_columns):
    designs = generate_designs()

    best_score = -1
    best_design = None

    for design in designs:
        yearly_scores = []

        # evaluate across all months
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
            yearly_scores.append(score)

        # 🔥 aggregate performance
        score = min(yearly_scores)  # worst-case optimization

        if score > best_score:
            best_score = score
            best_design = design

    return best_design, best_score

# EXPLANATION GENERATOR
def generate_explanation(city, score):
    if score < 50:
        return f"Buildings in {city} will experience significant thermal discomfort across the year. Passive design alone may not be sufficient, and active cooling strategies may be required."

    elif score < 75:
        return f"The climate in {city} presents moderate thermal stress across seasons. Optimized design with ventilation and insulation can significantly improve comfort."

    else:
        return f"The climate in {city} is relatively favorable across the year. Standard design strategies are sufficient, with minor improvements through ventilation."

# MAIN
if __name__ == "__main__":
    df = load_data()
    model = load_model()

    # important: use same columns as training   
    reference_columns = load_features()

    city = "chennai"
    month = 5  # example

    best_design, score = find_best_design_yearly(model, df, city, reference_columns)

    print("\nBest Design:")
    print(best_design)

    print("\nPredicted Comfort Score:", round(score, 2))

    print("\nExplanation:")
    print(generate_explanation(city, score))