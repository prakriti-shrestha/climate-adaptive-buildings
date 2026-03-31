import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, "data", "processed", "comfort_dataset.csv")

    df = pd.read_csv(path)
    return df


def preprocess(df):
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["city", "shape"], drop_first=True)

    # Features and target
    X = df.drop(columns=["comfort"])
    y = df["comfort"]

    return X, y


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\n Evaluation:")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 Score:", r2_score(y_test, preds))

def feature_importance(model, X):
    import pandas as pd

    importance = model.feature_importances_
    feature_names = X.columns

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(imp_df.head(10))

if __name__ == "__main__":
    df = load_data()

    X, y = preprocess(df)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, "src\models")
    os.makedirs(model_dir, exist_ok=True)

    # save feature names
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "features.pkl"))
    
    df_extreme = df[(df["temp"] > 30) | (df["humidity"] > 70)]

    X_ext, y_ext = preprocess(df_extreme)

    X_train, X_test, y_train, y_test = train_test_split(
        X_ext, y_ext, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)
    feature_importance(model, X)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, "src\models")

    # create folder if not exists
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "rf_model.pkl")

    joblib.dump(model, model_path)