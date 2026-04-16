"""
Tests d'intégration pour api/main.py (FastAPI).
Utilise TestClient — pas besoin de lancer le serveur.
"""
import os
import sys
from unittest.mock import patch
from fastapi.testclient import TestClient

import pandas as pd  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app

client = TestClient(app)


# ─── Endpoint GET / ────────────────────────────────────────────────────────────

def test_read_root():
    """Vérifie que l'endpoint racine répond correctement."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_root_message_non_vide():
    """Le message de bienvenue ne doit pas être vide."""
    response = client.get("/")
    assert len(response.json()["message"]) > 0


def test_root_content_type_json():
    """La réponse doit être en JSON."""
    response = client.get("/")
    assert "application/json" in response.headers["content-type"]


# ─── Endpoint GET /version ─────────────────────────────────────────────────────

def test_get_version():
    """Vérifie le format du versioning (Exigence MLOps)."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "software_version" in data
    assert "model_version" in data
    assert data["model_version"] == "LightGBM_V1"


def test_version_contient_environment():
    """La réponse /version doit contenir la clé 'environment'."""
    response = client.get("/version")
    assert "environment" in response.json()


def test_version_environment_local_par_defaut():
    """Sans COMMIT_ID, l'environnement doit être 'local'."""
    env_sans_commit = {k: v for k, v in os.environ.items() if k != "COMMIT_ID"}
    with patch.dict(os.environ, env_sans_commit, clear=True):
        response = client.get("/version")
    assert response.json()["environment"] == "local"


def test_version_environment_production_avec_commit():
    """Avec COMMIT_ID défini, l'environnement doit être 'production'."""
    with patch.dict(os.environ, {"COMMIT_ID": "abc123def"}):
        response = client.get("/version")
    assert response.json()["environment"] == "production"


# ─── Endpoint GET /predictions ─────────────────────────────────────────────────

def test_get_predictions_structure():
    """Vérifie que l'endpoint predictions renvoie une liste (si des données existent)."""
    response = client.get("/predictions")
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "temperature_predite" in data[0]
            assert "date_cible" in data[0]
    else:
        assert response.status_code == 404


def test_predictions_retourne_404_si_db_vide():
    """Si la DB ne contient aucune prédiction, l'endpoint doit retourner 404."""
    df_vide = pd.DataFrame(columns=["date_cible", "temperature_predite", "model_id"])
    with patch("pandas.read_sql", return_value=df_vide):
        response = client.get("/predictions")
    assert response.status_code == 404


def test_predictions_filtre_par_date_ne_crash_pas():
    """Le paramètre ?date= ne doit pas provoquer de crash."""
    response = client.get("/predictions?date=2025-01-01")
    assert response.status_code in [200, 404]


# ─── Endpoint GET /monitoring/data ─────────────────────────────────────────────

def test_monitoring_data_parametres_obligatoires():
    """Sans start_date ou end_date, l'API doit retourner 422 (validation error)."""
    response = client.get("/monitoring/data")
    assert response.status_code == 422


def test_monitoring_data_avec_parametres_ne_crash_pas():
    """Avec les paramètres requis, l'endpoint ne doit pas lever d'exception non gérée."""
    response = client.get("/monitoring/data?start_date=2025-01-01&end_date=2025-01-02")
    assert response.status_code in [200, 404, 500]


# ─── Middleware latence ─────────────────────────────────────────────────────────

def test_header_x_process_time_present():
    """Le middleware doit ajouter le header X-Process-Time à chaque réponse."""
    response = client.get("/")
    assert "x-process-time" in response.headers


def test_process_time_est_un_nombre():
    """X-Process-Time doit être un nombre flottant valide."""
    response = client.get("/")
    process_time = float(response.headers["x-process-time"])
    assert process_time >= 0.0


def test_process_time_inferieur_a_2s():
    """La latence de l'endpoint racine doit être raisonnable (< 2s)."""
    response = client.get("/")
    process_time = float(response.headers["x-process-time"])
    assert process_time < 2.0


# ─── Robustesse ────────────────────────────────────────────────────────────────

def test_endpoint_inexistant_retourne_404():
    """Un endpoint inconnu doit retourner HTTP 404."""
    response = client.get("/endpoint_inconnu_xyz")
    assert response.status_code == 404


def test_version_content_type_json():
    """L'endpoint /version doit aussi retourner du JSON."""
    response = client.get("/version")
    assert "application/json" in response.headers["content-type"]
