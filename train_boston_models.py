
"""
train_boston_models.py

Train multiple regression models on the Boston Housing dataset (provided as data.csv),
evaluate them, pick the best by lowest Test RMSE, and save deployment-ready artifacts.

Artifacts saved:
- boston_best_model.pkl        (sklearn Pipeline: preprocessing + model)
- feature_columns.json         (metadata for features & target)
- model_card.md                (summary of best model and metrics)
- residual_plot.png            (diagnostic plot)
- pred_vs_actual.png           (diagnostic plot)

Usage:
    python train_boston_models.py --data data.csv

Optional args:
    --test_size 0.2 --seed 42
"""

import os
import json
import math
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import joblib


def main(args):
    data_path = args.data
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"
    df = pd.read_csv(data_path).dropna().reset_index(drop=True)

    # Detect target (MEDV typical for Boston dataset; fallback to last column)
    target_col = next((c for c in df.columns if c.strip().lower() == "medv"), df.columns[-1])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        [("num", numeric_transformer, numeric_features),
         ("cat", categorical_transformer, categorical_features)]
    )

    # Models
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

    # Train & evaluate
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

    # Sort results
    results_df = pd.DataFrame(results).sort_values(by="Test_RMSE").reset_index(drop=True)
    print("\n=== Model Comparison (sorted by Test RMSE) ===")
    print(results_df.to_string(index=False))

    # Best model
    best_name = results_df.iloc[0]["Model"]
    best_pipe = models[best_name]

    # Save artifacts
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
        f.write("\n## Features\n")
        f.write(f"- Numeric: {numeric_features}\n")
        f.write(f"- Categorical: {categorical_features}\n")
        f.write(f"- Target: {target_col}\n")

    # Diagnostics
    y_pred_best = best_pipe.predict(X_test)
    residuals = y_test - y_pred_best

    # Residual plot
    plt.figure()
    plt.scatter(y_pred_best, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (y_true - y_pred)")
    plt.title(f"Residual Plot — {best_name}")
    plt.savefig("residual_plot.png", bbox_inches="tight")
    plt.close()

    # Predicted vs Actual
    plt.figure()
    plt.scatter(y_test, y_pred_best, alpha=0.7)
    lo = min(y_test.min(), y_pred_best.min())
    hi = max(y_test.max(), y_pred_best.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual — {best_name}")
    plt.savefig("pred_vs_actual.png", bbox_inches="tight")
    plt.close()

    print("\nArtifacts saved:")
    print(" - boston_best_model.pkl")
    print(" - feature_columns.json")
    print(" - model_card.md")
    print(" - residual_plot.png")
    print(" - pred_vs_actual.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.csv", help="Path to CSV dataset")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
