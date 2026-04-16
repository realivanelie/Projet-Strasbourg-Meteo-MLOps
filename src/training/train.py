import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.db_manager import get_connection

DATA_PATH = 'data/open-meteo-48.61N7.74E150m.csv'
MODEL_REGISTRY_PATH = "model/registry/"
MODEL_PKL_PATH = "model/registry/model_lightgbm.pkl"
SEUIL_FIN_TRAIN = "2023-12-31 23:00"
SEUIL_FIN_VALIDATION = "2024-12-31 23:00"

COLS_FEATURES = [
    'heure_sin', 'heure_cos', 'annee_sin', 'annee_cos',
    'lag_1', 'lag_2', 'lag_3', 'lag_24h',
    'moyenne_mobile_24h', 'ecart_type_24h', 'tendance_3h'
]

# Configuration MLflow
os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)
mlflow.set_tracking_uri(f"file://{os.path.abspath(MODEL_REGISTRY_PATH)}")
mlflow.set_experiment("Meteo_LightGBM_Production")


def load_csv(filepath):
    """Charge le CSV Open-Meteo historique (2018-2026) agrégé à 3H."""
    df = pd.read_csv(filepath, skiprows=3)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df_3h = df.resample('3H', closed='left', label='left').mean()
    df_3h.rename(columns={'temperature_2m (°C)': 'temperature'}, inplace=True)
    return df_3h[['temperature']].dropna()


def load_sqlite():
    """Charge les observations récentes depuis SQLite (observations_historiques)."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT date_time, temperature FROM observations_historiques ORDER BY date_time",
        conn
    )
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=['temperature'])
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index('date_time', inplace=True)
    df.index.name = 'time'
    return df[['temperature']]


def build_features(df_temp):
    """Calcule les 11 features à partir d'une série temporelle de température."""
    df = df_temp.copy()
    df['heure_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['heure_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['annee_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['annee_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['lag_1'] = df['temperature'].shift(1)
    df['lag_2'] = df['temperature'].shift(2)
    df['lag_3'] = df['temperature'].shift(3)
    df['lag_24h'] = df['temperature'].shift(8)
    df['moyenne_mobile_24h'] = df['temperature'].shift(1).rolling(window=8).mean()
    df['ecart_type_24h'] = df['temperature'].shift(1).rolling(window=8).std()
    df['tendance_3h'] = df['temperature'].shift(1) - df['temperature'].shift(2)
    return df[['temperature'] + COLS_FEATURES].dropna()


def prepare_data():
    """Fusionne CSV historique + observations SQLite récentes, puis calcule les features."""
    print(" Chargement des données...")
    df_csv = load_csv(DATA_PATH)
    df_sqlite = load_sqlite()

    if not df_sqlite.empty:
        # On ne garde que les nouvelles lignes (après la fin du CSV)
        csv_end = df_csv.index.max()
        df_new = df_sqlite[df_sqlite.index > csv_end]
        if not df_new.empty:
            df_combined = pd.concat([df_csv, df_new]).sort_index()
            print(f"   CSV     : {len(df_csv)} points → {df_csv.index.min().date()} à {csv_end.date()}")
            print(f"   SQLite  : {len(df_new)} nouveaux points → {df_new.index.min().date()} à {df_new.index.max().date()}")
        else:
            df_combined = df_csv
            print(f"   CSV uniquement : {len(df_csv)} points (SQLite sans nouvelles données après {csv_end.date()})")
    else:
        df_combined = df_csv
        print(f"   CSV uniquement : {len(df_csv)} points (SQLite vide)")

    print(f"   Dataset final  : {len(df_combined)} points ({df_combined.index.min().date()} → {df_combined.index.max().date()})")
    return build_features(df_combined)


def split_data(dataframe):
    """Découpage chronologique : train ≤ 2023, test ≥ 2025."""
    train = dataframe.loc[:SEUIL_FIN_TRAIN]
    test = dataframe.loc[SEUIL_FIN_VALIDATION:].iloc[1:]
    return train, test


def run_training_pipeline():
    print(" Démarrage du Pipeline d'entraînement MLOps...")

    # 1. Données fusionnées CSV + SQLite
    df_features = prepare_data()
    data_train, data_test = split_data(df_features)

    X_train = data_train.drop(columns=['temperature'])
    y_train = data_train['temperature']
    X_test = data_test.drop(columns=['temperature'])
    y_test = data_test['temperature']

    print(f"   Train : {len(X_train)} points | Test : {len(X_test)} points")

    # 2. Paramètres du modèle
    lgbm_params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    }

    # 3. Traçage MLflow
    print(" Lancement du Run MLflow...")
    from datetime import datetime as _dt
    run_name = f"LightGBM_V_du{_dt.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(lgbm_params)
        mlflow.log_param("features_count", X_train.shape[1])
        mlflow.log_param("train_start", str(data_train.index[0]))
        mlflow.log_param("train_end", str(data_train.index[-1]))

        print(" Entraînement du modèle LightGBM.")
        model = lgb.LGBMRegressor(**lgbm_params, verbosity=-1)
        model.fit(X_train, y_train)

        # Évaluation
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = mean_absolute_percentage_error(y_test, pred) * 100

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)

        # Sauvegarde MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lightgbm_model",
            registered_model_name="Meteo_LightGBM"
        )

        # Export .pkl pour batch_predict.py
        joblib.dump(model, MODEL_PKL_PATH)
        print(f"   Modèle exporté → {MODEL_PKL_PATH}")

        print("\n Pipeline terminé avec succès !")
        print(f"   MAE Test  : {mae:.3f} °C")
        print(f"   RMSE Test : {rmse:.3f} °C")
        print(f"   Run ID    : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run_training_pipeline()