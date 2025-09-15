# train_boston_models_all.py
import os, json, math, argparse
import pandas as pd, numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_and_eval(X_train, X_test, y_train, y_test, preprocessor, seed):
    models = {
        "LinearRegression": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 15), cv=3),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
    }

    results = {}
    for name, est in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", est)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        info = {
            "model_name": name,
            "test_rmse": float(round(rmse, 4)),
            "test_r2": float(round(r2, 4)),
        }
        # capture alpha for Ridge
        if name == "RidgeCV":
            info["chosen_alpha"] = float(pipe.named_steps["model"].alpha_)
        results[name] = {"pipeline_path": f"boston_{name}.pkl", "metrics": info}

        # save pipeline
        joblib.dump(pipe, f"boston_{name}.pkl")

    # pick best by RMSE
    best_name = sorted(results.values(), key=lambda d: d["metrics"]["test_rmse"])[0]["metrics"]["model_name"]
    index = {
        "best": best_name,
        "models": results,
        "note": "Pipelines include imputation + scaling + (if any) encoding."
    }
    with open("models_index.json", "w") as f:
        json.dump(index, f, indent=2)

    return index

def main(args):
    df = pd.read_csv(args.data)
    df = df.replace([np.inf, -np.inf], np.nan)

    # detect target
    target = next((c for c in df.columns if c.strip().lower()=="medv"), df.columns[-1])
    X = df.drop(columns=[target])
    y = df[target]

    # feature types
    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in X.columns if c not in num_feats]

    # UI defaults (medians/modes) for app
    ui_defaults = {}
    ui_defaults["numeric_defaults"] = {c: (float(X[c].median()) if pd.api.types.is_numeric_dtype(X[c]) else None) for c in num_feats}
    ui_defaults["categorical_defaults"] = {}
    for c in cat_feats:
        mode = X[c].mode(dropna=True)
        ui_defaults["categorical_defaults"][c] = (None if mode.empty else str(mode.iloc[0]))
    ui_defaults["all_features"] = X.columns.tolist()
    ui_defaults["target"] = target
    ui_defaults["numeric_features"] = num_feats
    ui_defaults["categorical_features"] = cat_feats
    with open("ui_defaults.json", "w") as f:
        json.dump(ui_defaults, f, indent=2)

    # preprocessing
    num_xf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_xf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_xf, num_feats),
        ("cat", cat_xf, cat_feats),
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    index = train_and_eval(Xtr, Xte, ytr, yte, preprocessor, args.seed)

    # also save a single "best" alias for backward compatibility
    best_name = index["best"]
    best_path = index["models"][best_name]["pipeline_path"]
    pipe = joblib.load(best_path)
    joblib.dump(pipe, "boston_best_model.pkl")

    # minimal model card
    card = [
        "# Boston Housing Model Card",
        f"- **Best Model**: {best_name}",
        f"- **Test RMSE**: {index['models'][best_name]['metrics']['test_rmse']}",
        f"- **Test R²**: {index['models'][best_name]['metrics']['test_r2']}",
    ]
    if best_name == "RidgeCV":
        card.append(f"- **Chosen Alpha**: {index['models'][best_name]['metrics'].get('chosen_alpha')}")
    card.append("- Pipelines include imputation + scaling + encoding (if needed).")
    with open("model_card.md", "w") as f:
        f.write("\n".join(card))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
