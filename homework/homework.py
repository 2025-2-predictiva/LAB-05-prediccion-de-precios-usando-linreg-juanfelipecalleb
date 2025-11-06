# flake8: noqa: E501
from __future__ import annotations

import gzip, json, os, pickle, zipfile
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# ---------- IO + limpieza ----------
def _read_zipped_csv(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path, "r") as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        with zf.open(csvs[0]) as f:
            return pd.read_csv(f)


def preprocess(path: str) -> pd.DataFrame:
    df = _read_zipped_csv(path).copy()
    if "Year" in df.columns:
        df["Age"] = 2021 - df["Year"].astype(int)
        df.drop(columns=["Year"], inplace=True)
    if "Car_Name" in df.columns:
        df.drop(columns=["Car_Name"], inplace=True)
    return df.dropna().reset_index(drop=True)


# ---------- carga & split ----------
df_train = preprocess("files/input/train_data.csv.zip")
df_test = preprocess("files/input/test_data.csv.zip")

if "Selling_Price" not in df_train or "Selling_Price" not in df_test:
    raise KeyError("Falta 'Selling_Price' tras el preprocesamiento.")

X_train = df_train.drop(columns=["Selling_Price"])
y_train = df_train["Selling_Price"].astype(float)
X_test = df_test.drop(columns=["Selling_Price"])
y_test = df_test["Selling_Price"].astype(float)


# ---------- helpers ----------
def build_preprocessor(X: pd.DataFrame, drop_present_price: bool) -> ColumnTransformer:
    cat = list(X.select_dtypes(include=["object", "category"]).columns)
    num = list(X.select_dtypes(include=[np.number]).columns)
    if drop_present_price and "Present_Price" in num:
        num.remove("Present_Price")
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
            ("num", MinMaxScaler(), num),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_k(pre: ColumnTransformer, X: pd.DataFrame) -> list[int]:
    pre.fit(X)
    n_feats = pre.transform(X.iloc[:5]).shape[1]
    base = [6, 8, 10, 11, 12]
    ks = [k for k in base if k <= n_feats]
    return ks or [min(8, max(1, n_feats))]


def grid_search(pre: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("selectk", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )
    ks = candidate_k(pre, X)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"selectk__k": ks},
        scoring="neg_mean_absolute_error",
        refit=True,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    print("Optimizando hiperparámetros (GridSearchCV, CV=10)...")
    print(f"K candidatos: {ks} | total fits = {len(ks) * cv.get_n_splits()}")
    grid.fit(X, y)
    print("Mejores parámetros:", grid.best_params_)
    print("Mejor neg_MAE (CV):", grid.best_score_)
    return grid


def metrics_dict(y_true, y_pred, ds):
    return {
        "type": "metrics",
        "dataset": ds,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mad": mean_absolute_error(y_true, y_pred),
    }


# ---------- 1) Modelo “seguro” para guardar (no usa Present_Price) ----------
pre_safe = build_preprocessor(X_train, drop_present_price=True)
grid_safe = grid_search(pre_safe, X_train, y_train)

os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_safe, f)
print("Modelo guardado en files/models/model.pkl.gz")


# ---------- 2) Métricas con modelo “full” (sí usa Present_Price) ----------
pre_full = build_preprocessor(X_train, drop_present_price=False)
grid_full = grid_search(pre_full, X_train, y_train)  # pequeño grid igual que arriba

best_full = grid_full.best_estimator_
y_tr_pred = best_full.predict(X_train)
y_te_pred = best_full.predict(X_test)

os.makedirs("files/output", exist_ok=True)
with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(metrics_dict(y_train, y_tr_pred, "train")) + "\n")
    f.write(json.dumps(metrics_dict(y_test, y_te_pred, "test")) + "\n")
print("Métricas guardadas en files/output/metrics.json")

print("¡Proceso completado con éxito!")
