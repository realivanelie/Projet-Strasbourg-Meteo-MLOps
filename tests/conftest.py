"""
Fixtures partagées entre tous les fichiers de test.
Utilise une base SQLite EN MEMOIRE pour isoler les tests
(ne touche jamais à la vraie base data/meteo_predictions.db).
"""
import pytest
import sqlite3
import pandas as pd
import numpy as np
from unittest.mock import patch



# FIXTURE : base SQLite en mémoire avec les bonnes tables
@pytest.fixture
def db_connection():
    """Crée une base SQLite en mémoire avec les tables du projet."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE observations_historiques (
            date_time TEXT PRIMARY KEY,
            temperature REAL
        )
    """)
    conn.execute("""
        CREATE TABLE predictions_batch (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_inference TEXT,
            date_cible TEXT,
            temperature_predite REAL,
            model_id TEXT
        )
    """)
    conn.commit()
    yield conn
    conn.close()



# FIXTURE : série de températures simulées (14 jours x 8 pas/jour)
@pytest.fixture
def sample_temperature_series():
    """Série temporelle de 112 observations à pas de 3h (14 jours)."""
    idx = pd.date_range("2025-01-01", periods=112, freq="3H")
    values = 5 + 3 * np.sin(2 * np.pi * np.arange(112) / 8) + np.random.normal(0, 0.5, 112)
    return pd.Series(values, index=idx, name="temperature")


# FIXTURE : modèle factice (mock) qui retourne toujours 10.0°C
@pytest.fixture
def mock_model():
    """Modèle factice retournant une prédiction fixe de 10.0°C."""
    class FakeModel:
        def predict(self, X):
            return np.array([10.0] * len(X))
    return FakeModel()
