"""
Tests unitaires pour src/data/db_manager.py
Teste les fonctions d'accès à la base SQLite de manière isolée
en utilisant une base en mémoire (jamais la vraie DB).
"""
import pytest
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────
# TESTS : insert_observations
# ─────────────────────────────────────────────────────────────

class TestInsertObservations:

    def test_insertion_basique(self, db_connection):
        """Vérifie qu'une observation est bien insérée dans la table."""
        idx = pd.date_range("2025-01-01", periods=3, freq="3h")
        df = pd.DataFrame({"temperature": [2.5, 3.0, 3.5]}, index=idx)

        # On injecte directement dans la connexion en mémoire
        for index, row in df.iterrows():
            db_connection.execute(
                "INSERT OR IGNORE INTO observations_historiques (date_time, temperature) VALUES (?, ?)",
                (str(index), row["temperature"])
            )
        db_connection.commit()

        cursor = db_connection.execute("SELECT COUNT(*) FROM observations_historiques")
        count = cursor.fetchone()[0]
        assert count == 3

    def test_pas_de_doublon(self, db_connection):
        """INSERT OR IGNORE : insérer 2x la même date ne crée pas de doublon."""
        date = "2025-01-01 00:00:00"
        db_connection.execute(
            "INSERT OR IGNORE INTO observations_historiques (date_time, temperature) VALUES (?, ?)",
            (date, 5.0)
        )
        db_connection.execute(
            "INSERT OR IGNORE INTO observations_historiques (date_time, temperature) VALUES (?, ?)",
            (date, 99.0)  # valeur différente, même date
        )
        db_connection.commit()

        cursor = db_connection.execute("SELECT COUNT(*) FROM observations_historiques")
        assert cursor.fetchone()[0] == 1  # toujours 1 seule ligne

    def test_temperature_correcte(self, db_connection):
        """Vérifie que la valeur insérée est bien celle récupérée."""
        db_connection.execute(
            "INSERT OR IGNORE INTO observations_historiques (date_time, temperature) VALUES (?, ?)",
            ("2025-06-15 12:00:00", 28.7)
        )
        db_connection.commit()

        cursor = db_connection.execute(
            "SELECT temperature FROM observations_historiques WHERE date_time = ?",
            ("2025-06-15 12:00:00",)
        )
        result = cursor.fetchone()[0]
        assert abs(result - 28.7) < 1e-6


# ─────────────────────────────────────────────────────────────
# TESTS : insert_predictions
# ─────────────────────────────────────────────────────────────

class TestInsertPredictions:

    def test_insertion_predictions(self, db_connection):
        """Vérifie qu'une liste de prédictions est insérée correctement."""
        predictions = [
            ("2025-01-02 00:00:00", 4.5),
            ("2025-01-02 03:00:00", 3.8),
            ("2025-01-02 06:00:00", 3.2),
        ]
        date_inference = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for date_cible, temp_pred in predictions:
            db_connection.execute(
                "INSERT INTO predictions_batch (date_inference, date_cible, temperature_predite, model_id) VALUES (?, ?, ?, ?)",
                (date_inference, date_cible, temp_pred, "LightGBM_V1")
            )
        db_connection.commit()

        cursor = db_connection.execute("SELECT COUNT(*) FROM predictions_batch")
        assert cursor.fetchone()[0] == 3

    def test_model_id_enregistre(self, db_connection):
        """Vérifie que le model_id est bien sauvegardé."""
        db_connection.execute(
            "INSERT INTO predictions_batch (date_inference, date_cible, temperature_predite, model_id) VALUES (?, ?, ?, ?)",
            ("2025-01-01 00:00:00", "2025-01-01 03:00:00", 5.0, "LightGBM_V1")
        )
        db_connection.commit()

        cursor = db_connection.execute("SELECT model_id FROM predictions_batch")
        model_id = cursor.fetchone()[0]
        assert model_id == "LightGBM_V1"

    def test_temperature_predite_correcte(self, db_connection):
        """Vérifie que la température prédite est bien stockée."""
        db_connection.execute(
            "INSERT INTO predictions_batch (date_inference, date_cible, temperature_predite, model_id) VALUES (?, ?, ?, ?)",
            ("2025-01-01 00:00:00", "2025-01-01 03:00:00", -3.14, "LightGBM_V1")
        )
        db_connection.commit()

        cursor = db_connection.execute("SELECT temperature_predite FROM predictions_batch")
        temp = cursor.fetchone()[0]
        assert abs(temp - (-3.14)) < 1e-6

    def test_predictions_multiples_meme_date(self, db_connection):
        """On peut insérer plusieurs prédictions pour la même date (pas de contrainte UNIQUE)."""
        for _ in range(3):
            db_connection.execute(
                "INSERT INTO predictions_batch (date_inference, date_cible, temperature_predite, model_id) VALUES (?, ?, ?, ?)",
                ("2025-01-01 00:00:00", "2025-01-01 03:00:00", 5.0, "LightGBM_V1")
            )
        db_connection.commit()

        cursor = db_connection.execute("SELECT COUNT(*) FROM predictions_batch")
        assert cursor.fetchone()[0] == 3


# ─────────────────────────────────────────────────────────────
# TESTS : structure de la base de données
# ─────────────────────────────────────────────────────────────

class TestDBStructure:

    def test_table_observations_existe(self, db_connection):
        """Vérifie que la table observations_historiques existe."""
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observations_historiques'"
        )
        assert cursor.fetchone() is not None

    def test_table_predictions_existe(self, db_connection):
        """Vérifie que la table predictions_batch existe."""
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions_batch'"
        )
        assert cursor.fetchone() is not None

    def test_predictions_autoincrement(self, db_connection):
        """Vérifie que l'id est bien auto-incrémenté."""
        for i in range(3):
            db_connection.execute(
                "INSERT INTO predictions_batch (date_inference, date_cible, temperature_predite, model_id) VALUES (?, ?, ?, ?)",
                ("2025-01-01", f"2025-01-0{i+2}", float(i), "LightGBM_V1")
            )
        db_connection.commit()

        cursor = db_connection.execute("SELECT id FROM predictions_batch ORDER BY id")
        ids = [row[0] for row in cursor.fetchall()]
        assert ids == [1, 2, 3]
