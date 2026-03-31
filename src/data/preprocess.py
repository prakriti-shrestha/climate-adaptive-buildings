import pandas as pd
import os

def preprocess_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    input_path = os.path.join(base_dir, "data", "processed", "combined.csv")
    output_path = os.path.join(base_dir, "data", "processed", "monthly_climate.csv")

    df = pd.read_csv(input_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize units 
    df = standardize_units(df)

    # Handle missing values
    df = df.infer_objects(copy=False)
    df = df.interpolate()

    # Create month column
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format='%Y') + pd.to_timedelta(df["DOY"] - 1, unit='D')
    df["month"] = df["date"].dt.month

    # Group by city + year + month
    monthly_df = df.groupby(['city', 'YEAR', 'month']).agg({
        'T2M': 'mean',
        'RH2M': 'mean',
        'WS2M': 'mean'
    }).reset_index()

    # Rename columns
    monthly_df.columns = ['city', 'year', 'month', 'temp', 'humidity', 'wind']

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    monthly_df.to_csv(output_path, index=False)

    return monthly_df


def standardize_units(df):
    fahrenheit_cities = [
        "ahmedabad", "chennai", "delhi",
        "hyderabad", "kolkata", "mumbai", "pune"
    ]

    for city in df["city"].unique():
        mask = df["city"] == city

        if city in fahrenheit_cities:
            df.loc[mask, "T2M"] = (df.loc[mask, "T2M"] - 32) * 5/9
            df.loc[mask, "WS2M"] = df.loc[mask, "WS2M"] * 0.44704
            print(f"{city}: Converted")
        else:
            print(f"{city}: Already correct")

    return df


if __name__ == "__main__":
    df = preprocess_data()
    print(df.head())
    df[df["city"] == "ahmedabad"].sort_values(["year", "month"]).head(12)
    print(sorted(df["month"].unique()))
    df[df["city"] == "ahmedabad"].sort_values(["year", "month"]).head(20)
    df.groupby("city")["temp"].max()
