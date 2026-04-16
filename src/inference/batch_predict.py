import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.db_manager import insert_predictions, get_connection

MODEL_PATH = "model/registry/model_lightgbm.pkl" 
MODEL_ID = "LightGBM_V1"

def load_recent_data_from_db(limit=15):
    """Récupère suffisamment de données pour calculer les lags et moyennes mobiles."""
    conn = get_connection()
    query = f"SELECT date_time, temperature FROM observations_historiques ORDER BY date_time DESC LIMIT {limit}"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        raise ValueError("La base est vide. Lancez d'abord src/data/fetch_data.py")
        
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index('date_time', inplace=True)
    df.sort_index(inplace=True) 
    return df['temperature']

def generate_features(history, next_time):
    """Génère les 11 features exactes attendues par le modèle."""
    feat = {}
    
    # Features cycliques
    feat['heure_sin'] = np.sin(2 * np.pi * next_time.hour / 24)
    feat['heure_cos'] = np.cos(2 * np.pi * next_time.hour / 24)
    feat['annee_sin'] = np.sin(2 * np.pi * next_time.dayofyear / 365.25)
    feat['annee_cos'] = np.cos(2 * np.pi * next_time.dayofyear / 365.25)
    
    feat['lag_1'] = history.iloc[-1]
    feat['lag_2'] = history.iloc[-2]
    feat['lag_3'] = history.iloc[-3]
    feat['lag_24h'] = history.iloc[-8] 
    
    #  Statistiques glissantes 
    last_8 = history.iloc[-8:]
    feat['moyenne_mobile_24h'] = last_8.mean()
    feat['ecart_type_24h'] = last_8.std()
    
    # Dynamique (Tendance)
    feat['tendance_3h'] = feat['lag_1'] - feat['lag_2']
    
    # Création du DataFrame avec l'ordre strict des colonnes du training
    cols_ordre = [
        'heure_sin', 'heure_cos', 'annee_sin', 'annee_cos', 
        'lag_1', 'lag_2', 'lag_3', 'lag_24h', 
        'moyenne_mobile_24h', 'ecart_type_24h', 'tendance_3h'
    ]
    
    return pd.DataFrame([feat], index=[next_time])[cols_ordre]

def run_batch_inference(horizon_steps=24):
    print(" Démarrage de l'inférence Batch .")
    
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH_ALT = "model/registry/champion_lightgbm.pkl"
        if os.path.exists(MODEL_PATH_ALT):
            model = joblib.load(MODEL_PATH_ALT)
        else:
            raise FileNotFoundError("Fichier modèle .pkl introuvable dans model/registry/")
    else:
        model = joblib.load(MODEL_PATH)
        
    # Charger l'historique des données pour les features
    history = load_recent_data_from_db(limit=15) 
    current_time = history.index[-1]
    
    predictions_to_save = []
    
    for _ in range(horizon_steps):
        next_time = current_time + timedelta(hours=3)
        
        # Inférence itérative
        df_feat = generate_features(history, next_time)
        pred_temp = model.predict(df_feat)[0]
        
        predictions_to_save.append((next_time.strftime("%Y-%m-%d %H:%M:%S"), float(pred_temp)))
        
        # Mise à jour de la fenêtre glissante pour le pas suivant qui sert de base pour les features
        history.loc[next_time] = pred_temp
        current_time = next_time
        
    insert_predictions(predictions_to_save, MODEL_ID)
    print(f" Inférence réussie. {horizon_steps} prédictions stockées dans SQLite.")

if __name__ == "__main__":
    run_batch_inference()