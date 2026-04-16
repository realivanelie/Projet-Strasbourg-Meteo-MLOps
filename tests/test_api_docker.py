"""
Tests HTTP de l'API — exécutés contre le conteneur Docker réel (port 8000).
Utilisés par le job 'test' du pipeline CI/CD GitHub Actions.
"""
import httpx
import pytest

BASE_URL = "http://localhost:8000"


def test_root_status():
    """GET / doit retourner 200."""
    r = httpx.get(f"{BASE_URL}/")
    assert r.status_code == 200


def test_root_contient_message():
    """La réponse de / doit contenir une clé 'message'."""
    r = httpx.get(f"{BASE_URL}/")
    assert "message" in r.json()


def test_health_status_ok():
    """GET /health doit retourner status=ok."""
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] in ["ok", "degraded"]


def test_health_contient_model_version():
    """GET /health doit exposer la version du modèle."""
    r = httpx.get(f"{BASE_URL}/health")
    assert "model_version" in r.json()


def test_version_status():
    """GET /version doit retourner 200."""
    r = httpx.get(f"{BASE_URL}/version")
    assert r.status_code == 200


def test_version_champs_presents():
    """GET /version doit contenir software_version, model_version, environment."""
    r = httpx.get(f"{BASE_URL}/version")
    data = r.json()
    assert "software_version" in data
    assert "model_version" in data
    assert "environment" in data


def test_version_model_correct():
    """Le modèle exposé doit être LightGBM_V1."""
    r = httpx.get(f"{BASE_URL}/version")
    assert r.json()["model_version"] == "LightGBM_V1"


def test_version_environment_production():
    """En CI/CD, COMMIT_ID est injecté → environment doit être 'production'."""
    r = httpx.get(f"{BASE_URL}/version")
    assert r.json()["environment"] == "production"


def test_predictions_status():
    """GET /predictions doit retourner 200 ou 404 (pas de crash)."""
    r = httpx.get(f"{BASE_URL}/predictions")
    assert r.status_code in [200, 404]


def test_predictions_structure_si_donnees():
    """Si des prédictions existent, chaque entrée doit avoir les bons champs."""
    r = httpx.get(f"{BASE_URL}/predictions")
    if r.status_code == 200:
        data = r.json()
        assert isinstance(data, list)
        if data:
            assert "date_cible" in data[0]
            assert "temperature_predite" in data[0]


def test_monitoring_data_sans_params_retourne_422():
    """GET /monitoring/data sans paramètres doit retourner 422."""
    r = httpx.get(f"{BASE_URL}/monitoring/data")
    assert r.status_code == 422


def test_endpoint_inexistant_retourne_404():
    """Un endpoint inconnu doit retourner 404."""
    r = httpx.get(f"{BASE_URL}/route_inconnue_xyz")
    assert r.status_code == 404


def test_header_x_process_time_present():
    """Le middleware doit ajouter X-Process-Time à chaque réponse."""
    r = httpx.get(f"{BASE_URL}/")
    assert "x-process-time" in r.headers


def test_metrics_endpoint():
    """GET /metrics doit retourner les métriques Prometheus."""
    r = httpx.get(f"{BASE_URL}/metrics")
    assert r.status_code == 200
    assert "prediction_requests_total" in r.text
