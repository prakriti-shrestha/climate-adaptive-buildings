import pandas as pd
import os

def load_and_merge_data():
    # Get project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    data_path = os.path.join(base_dir, "data", "raw")
    
    all_files = os.listdir(data_path)
    
    df_list = []

    for file in all_files:
        if file.endswith(".csv"):
            city_name = file.split(".")[0]
            
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)

            df["city"] = city_name.lower()

            df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df


if __name__ == "__main__":
    df = load_and_merge_data()
    
    print(df.head())
    print(df["city"].unique())

    # Save output safely
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(base_dir, "data", "processed", "combined.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)