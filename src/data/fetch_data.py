import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.db_manager import insert_observations

def fetch_from_archive(days_back=35):
    """
    Télécharge les données historiques depuis l'API archive Open-Meteo.
    Délai ~5 jours : données disponibles jusqu'à aujourd'hui - 5 jours.
    """
    end_date   = datetime.now() - timedelta(days=5)
    start_date = end_date - timedelta(days=days_back)

    print(f" [Archive] Téléchargement du {start_date.date()} au {end_date.date()}...")
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 48.61, "longitude": 7.74,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m"
    })
    response.raise_for_status()
    return response.json()


def fetch_from_forecast(days_back=7):
    """
    Télécharge les données récentes depuis l'API forecast Open-Meteo.
    Couvre les 7 derniers jours jusqu'à aujourd'hui (pas de délai).
    """
    print(f" [Forecast] Téléchargement des {days_back} derniers jours...")
    response = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": 48.61, "longitude": 7.74,
        "hourly": "temperature_2m",
        "past_days": days_back,
        "forecast_days": 1
    })
    response.raise_for_status()
    return response.json()


def parse_and_insert(data):
    """Convertit la réponse API en DataFrame 3H et insère dans SQLite."""
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"]
    })
    df.set_index("time", inplace=True)
    df_3h = df.resample('3h', closed='left', label='left').mean().dropna()
    insert_observations(df_3h)
    return df_3h


def fetch_recent_data():
    """
    Récupère les données récentes en combinant :
    - API archive  : données jusqu'à aujourd'hui - 5 jours (données validées)
    - API forecast : données des 7 derniers jours  (données temps réel)
    Les doublons sont gérés par INSERT OR IGNORE dans SQLite.
    """
    total = 0
    try:
        df = parse_and_insert(fetch_from_archive(days_back=35))
        total += len(df)
        print(f"   Archive  : {len(df)} points insérés")
    except requests.exceptions.RequestException as e:
        print(f" Erreur archive : {e}")

    try:
        df = parse_and_insert(fetch_from_forecast(days_back=7))
        total += len(df)
        print(f"   Forecast : {len(df)} points insérés")
    except requests.exceptions.RequestException as e:
        print(f" Erreur forecast : {e}")

    print(f" Total : {total} observations insérées dans SQLite.")


if __name__ == "__main__":
    fetch_recent_data()