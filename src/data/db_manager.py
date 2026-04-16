import sqlite3
import pandas as pd
import os
from datetime import datetime


DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "meteo_predictions.db")

def get_connection():
    """Crée et retourne une connexion à la base SQLite."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialise les tables SQL si elles n'existent pas encore."""
    print("Initialisation de la base de données SQLite...")
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1 : Historique des observations réelles
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations_historiques (
            date_time TEXT PRIMARY KEY,
            temperature REAL
        )
    ''')

    # Table 2 : Prédictions générées par notre modèle 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions_batch (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_inference TEXT,        
            date_cible TEXT,            
            temperature_predite REAL,
            model_id TEXT               
        )
    ''')

    conn.commit()
    conn.close()
    print(f" Base de données prête : {DB_PATH}")

def insert_observations(df_obs):
    """
    Insère de nouvelles observations réelles téléchargées.
    df_obs doit avoir l'index 'time' et une colonne 'temperature'.
    """
    conn = get_connection()
    for index, row in df_obs.iterrows():
        conn.execute('''
            INSERT OR IGNORE INTO observations_historiques (date_time, temperature)
            VALUES (?, ?)
        ''', (str(index), row['temperature']))
    conn.commit()
    conn.close()

def insert_predictions(predictions_list, model_id):
    """
    Insère un lot de prédictions.
    predictions_list : liste de tuples (date_cible, temperature_predite)
    """
    conn = get_connection()
    date_inference = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor = conn.cursor()
    for date_cible, temp_pred in predictions_list:
        cursor.execute('''
            INSERT INTO predictions_batch (date_inference, date_cible, temperature_predite, model_id)
            VALUES (?, ?, ?, ?)
        ''', (date_inference, str(date_cible), temp_pred, model_id))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()