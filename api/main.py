from fastapi import FastAPI, HTTPException, Request, Response
import sqlite3
import pandas as pd
import os
import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# CONFIGURATION DU LOGGING
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),                             
        RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)  
    ]
)
logger = logging.getLogger("api.meteo")

# MÉTRIQUES PROMETHEUS

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Nombre total de requetes par endpoint",
    ["endpoint"],
)

REQUEST_ERROR_COUNT = Counter(
    "prediction_request_errors_total",
    "Nombre total d'erreurs par endpoint",
    ["endpoint"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latence des requetes en secondes",
    ["endpoint"],
)

MODEL_INFO = Gauge(
    "model_info",
    "Informations sur le modele charge",
    ["model_version"],
)

MODEL_MAE = Gauge(
    "model_mae",
    "MAE du modele sur les dernieres predictions vs observations",
)

MODEL_RMSE = Gauge(
    "model_rmse",
    "RMSE du modele sur les dernieres predictions vs observations",
)

MODEL_PREDICTION_COUNT = Gauge(
    "model_prediction_count",
    "Nombre de predictions comparees aux observations reelles",
)

# APPLICATION FASTAPI
app = FastAPI(
    title="API Météo BIHAR - Strasbourg",
    version="0.0.0",
    description="API de prévision de température pour Strasbourg — Projet BIHAR2026"
)

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/meteo_predictions.db")
MODEL_VERSION = "LightGBM_V1"

# Log du démarrage de l'API avec les infos de version
logger.info("=" * 60)
logger.info(f"Démarrage API Météo BIHAR2026")
logger.info(f"Modèle chargé    : {MODEL_VERSION}")
logger.info(f"Base de données  : {DB_PATH}")
logger.info(f"Environnement    : {os.getenv('COMMIT_ID', 'local')}")
logger.info("=" * 60)

# Initialise la gauge avec le modèle chargé au démarrage
MODEL_INFO.labels(model_version=MODEL_VERSION).set(1)


# MIDDLEWARE : Logging des requêtes (latence, statut, méthode)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency_s  = time.time() - start_time
    latency_ms = latency_s * 1000

    endpoint = request.url.path

    # Incrémente les compteurs Prometheus
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_s)
    if response.status_code >= 400:
        REQUEST_ERROR_COUNT.labels(endpoint=endpoint).inc()

    # Niveau WARNING si erreur HTTP (4xx/5xx), INFO sinon
    log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(
        log_level,
        f"[{request.method}] {endpoint} "
        f"| status={response.status_code} "
        f"| latency={latency_ms:.2f}ms "
        f"| model={MODEL_VERSION}"
    )

    # Header exposant la latence en ms (valeur numérique pure, sans unité)
    response.headers["X-Process-Time"] = f"{latency_ms:.2f}"
    return response


# UTILITAIRE : Connexion SQLite
def get_db_connection():
    """Crée et retourne une connexion à la base SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ENDPOINT : GET /
@app.get("/")
def read_root():
    """Point d'entrée de l'API — description du service."""
    return {
        "message": "Bienvenue sur l'API de prévision météo du projet final BIHAR2026 pour la ville de Strasbourg!",
        "model": MODEL_VERSION,
        "docs": "/docs"
    }


# ENDPOINT : GET /health
@app.get("/health")
def health_check():
    """Health check — vérifie l'état du service et de la base de données."""
    db_status = "ok"
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error(f"[HEALTH] Base de données inaccessible : {e}")

    status = "ok" if db_status == "ok" else "degraded"
    if status != "ok":
        logger.warning(f"[HEALTH] Service dégradé — db_status={db_status}")

    return {
        "status": status,
        "model_version": MODEL_VERSION,
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }


# ENDPOINT : GET /metrics
@app.get("/metrics")
def metrics():
    """Endpoint Prometheus — expose toutes les métriques du service."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ENDPOINT : GET /version
@app.get("/version")
def get_version():
    """Retourne la version logicielle (commit) et la version du modèle."""
    software_version = os.getenv("COMMIT_ID", "0.0.0")
    env = "local" if software_version == "0.0.0" else "production"

    logger.info(f"[VERSION] software={software_version} | model={MODEL_VERSION} | env={env}")
    return {
        "software_version": software_version,
        "model_version": MODEL_VERSION,
        "environment": env
    }


# ENDPOINT : GET /predictions
@app.get("/predictions")
def get_predictions(date: str = None):
    """Retourne les prédictions de température (dernière version par date cible)."""
    logger.info(f"[PREDICTIONS] Requête reçue | filtre date={date}")

    conn = get_db_connection()
    query = """
        SELECT date_cible, temperature_predite, model_id
        FROM predictions_batch
        WHERE id IN (SELECT MAX(id) FROM predictions_batch GROUP BY date_cible)
    """
    params = []
    if date:
        query += " AND date_cible LIKE ?"
        params.append(f"{date}%")

    try:
        df = pd.read_sql(query, conn, params=params if params else None)
    except Exception as e:
        logger.warning(f"[PREDICTIONS] Table absente ou vide : {e}")
        conn.close()
        raise HTTPException(status_code=404, detail="Aucune prédiction trouvée pour cette période.")
    finally:
        conn.close()

    if df.empty:
        logger.warning(f"[PREDICTIONS] Aucune prédiction trouvée pour date={date}")
        raise HTTPException(status_code=404, detail="Aucune prédiction trouvée pour cette période.")

    logger.info(f"[PREDICTIONS] {len(df)} prédictions retournées")
    return df.to_dict(orient="records")


# ENDPOINT : GET /monitoring/data
@app.get("/monitoring/data")
def get_monitoring_data(start_date: str, end_date: str):
    """Retourne les prédictions VS réalité pour le suivi des erreurs."""
    logger.info(f"[MONITORING] Requête période : {start_date} → {end_date}")

    conn = get_db_connection()
    try:
        query_obs = """
            SELECT date_time, temperature as real_temp
            FROM observations_historiques
            WHERE date_time BETWEEN ? AND ?
        """
        query_pred = """
            SELECT date_cible as date_time, temperature_predite
            FROM predictions_batch
            WHERE id IN (SELECT MAX(id) FROM predictions_batch GROUP BY date_cible)
              AND date_cible BETWEEN ? AND ?
        """
        df_obs  = pd.read_sql(query_obs,  conn, params=(start_date, end_date))
        df_pred = pd.read_sql(query_pred, conn, params=(start_date, end_date))
    except Exception as e:
        logger.error(f"[MONITORING] Erreur SQL : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la lecture des données de monitoring.")
    finally:
        conn.close()

    merged = pd.merge(df_obs, df_pred, on="date_time", how="inner")

    if merged.empty:
        logger.warning(f"[MONITORING] Aucune donnée commune sur {start_date} → {end_date}")
        return []

    # Calcul MAE / RMSE sur la période
    errors = merged["real_temp"] - merged["temperature_predite"]
    mae  = errors.abs().mean()
    rmse = (errors ** 2).mean() ** 0.5

    # Mise à jour des Gauges Prometheus
    MODEL_MAE.set(round(mae, 4))
    MODEL_RMSE.set(round(rmse, 4))
    MODEL_PREDICTION_COUNT.set(len(merged))

    logger.info(f"[MONITORING] {len(merged)} points | MAE={mae:.3f}°C | RMSE={rmse:.3f}°C")

    return merged.to_dict(orient="records")


