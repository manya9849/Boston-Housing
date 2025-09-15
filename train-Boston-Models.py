import os, json, math, argparse
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import joblib


def main(args):
    # 1. Load dataset
    df = pd.read_csv(args.data)
    df = df.replace([np.inf, -np.inf], np.nan)   # replace inf → NaN

    # 2. Detect target column (use MEDV if present, else last column)
    target_col = next((c for c in df.columns if c.strip().lower() == "medv"), df.columns[-1])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # 4. Preprocessing (with imputation)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 5. Models
    models = {
        "LinearRegression": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
        "RidgeCV": Pipeline([
            ("prep", preprocessor),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 15), cv=3))
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=150, random_state=args.seed, n_jobs=-1))
        ]),
    }

    # 6. Train & evaluate
    results = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        row = {"Model": name, "Test_RMSE": round(rmse, 4), "Test_R2": round(r2, 4)}
        if name == "RidgeCV":
            row["Chosen_alpha"] = float(pipe.named_steps["model"].alpha_)
        results.append(row)

    results_df = pd.DataFrame(results).sort_values(by="Test_RMSE").reset_index(drop=True)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    # 7. Best model
    best_name = results_df.iloc[0]["Model"]
    best_pipe = models[best_name]

    # 8. Save artifacts
    joblib.dump(best_pipe, "boston_best_model.pkl")
    with open("feature_columns.json", "w") as f:
        json.dump({
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "all_features": X.columns.tolist(),
            "target": target_col,
        }, f, indent=2)

    with open("model_card.md", "w") as f:
        f.write(f"# Boston Housing Model Card\n\n")
        f.write(f"- **Best Model**: {best_name}\n")
        f.write(f"- **Test RMSE**: {results_df.iloc[0]['Test_RMSE']}\n")
        f.write(f"- **Test R²**: {results_df.iloc[0]['Test_R2']}\n")
        if "Chosen_alpha" in results_df.columns and not pd.isna(results_df.iloc[0].get("Chosen_alpha", np.nan)):
            f.write(f"- **Chosen Alpha**: {results_df.iloc[0]['Chosen_alpha']}\n")

    print("\nArtifacts saved:")
    print(" - boston_best_model.pkl")
    print(" - feature_columns.json")
    print(" - model_card.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
