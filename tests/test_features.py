"""
Tests unitaires pour le feature engineering (batch_predict.generate_features).
Vérifie que les 11 features sont correctement calculées sans fuite de données.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Import de la fonction à tester
# ─────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.batch_predict import generate_features

# Liste exacte des colonnes attendues par le modèle (ordre du training)
COLONNES_ATTENDUES = [
    'heure_sin', 'heure_cos', 'annee_sin', 'annee_cos',
    'lag_1', 'lag_2', 'lag_3', 'lag_24h',
    'moyenne_mobile_24h', 'ecart_type_24h', 'tendance_3h'
]


# ─────────────────────────────────────────────────────────────
# FIXTURE : historique minimal pour les tests (15 obs)
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def historique_15():
    """Série de 15 observations à pas de 3h (minimum requis pour les lags)."""
    idx = pd.date_range("2025-01-01 00:00:00", periods=15, freq="3h")
    vals = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 7.5, 7.0, 6.5,
            6.0, 5.5, 5.0, 4.5, 4.0]
    return pd.Series(vals, index=idx, name="temperature")


@pytest.fixture
def next_time():
    """Timestamp cible : le pas suivant après l'historique (après les 15 obs)."""
    return pd.Timestamp("2025-01-02 21:00:00")  # t+1 après la 15ème obs à 2025-01-02 18:00


# ─────────────────────────────────────────────────────────────
# TESTS : structure du DataFrame retourné
# ─────────────────────────────────────────────────────────────

class TestGenerateFeaturesStructure:

    def test_retourne_un_dataframe(self, historique_15, next_time):
        """generate_features doit retourner un DataFrame pandas."""
        result = generate_features(historique_15, next_time)
        assert isinstance(result, pd.DataFrame)

    def test_une_seule_ligne(self, historique_15, next_time):
        """Le DataFrame doit avoir exactement 1 ligne (1 pas prédit)."""
        result = generate_features(historique_15, next_time)
        assert result.shape[0] == 1

    def test_11_colonnes(self, historique_15, next_time):
        """Le modèle attend exactement 11 features."""
        result = generate_features(historique_15, next_time)
        assert result.shape[1] == 11

    def test_colonnes_dans_bon_ordre(self, historique_15, next_time):
        """L'ordre des colonnes doit être identique à celui du training."""
        result = generate_features(historique_15, next_time)
        assert list(result.columns) == COLONNES_ATTENDUES

    def test_index_est_next_time(self, historique_15, next_time):
        """L'index du DataFrame doit être le timestamp cible."""
        result = generate_features(historique_15, next_time)
        assert result.index[0] == next_time


# ─────────────────────────────────────────────────────────────
# TESTS : valeurs des features cycliques
# ─────────────────────────────────────────────────────────────

class TestFeaturesCycliques:

    def test_heure_sin_dans_intervalle(self, historique_15, next_time):
        """heure_sin doit être dans [-1, 1]."""
        result = generate_features(historique_15, next_time)
        assert -1.0 <= result['heure_sin'].iloc[0] <= 1.0

    def test_heure_cos_dans_intervalle(self, historique_15, next_time):
        """heure_cos doit être dans [-1, 1]."""
        result = generate_features(historique_15, next_time)
        assert -1.0 <= result['heure_cos'].iloc[0] <= 1.0

    def test_annee_sin_dans_intervalle(self, historique_15, next_time):
        """annee_sin doit être dans [-1, 1]."""
        result = generate_features(historique_15, next_time)
        assert -1.0 <= result['annee_sin'].iloc[0] <= 1.0

    def test_annee_cos_dans_intervalle(self, historique_15, next_time):
        """annee_cos doit être dans [-1, 1]."""
        result = generate_features(historique_15, next_time)
        assert -1.0 <= result['annee_cos'].iloc[0] <= 1.0

    def test_heure_sin_valeur_exacte(self, historique_15):
        """Vérifie la valeur exacte de heure_sin pour minuit (heure=0)."""
        t = pd.Timestamp("2025-06-15 00:00:00")  # minuit → sin(0) = 0
        result = generate_features(historique_15, t)
        assert abs(result['heure_sin'].iloc[0] - 0.0) < 1e-10

    def test_heure_cos_valeur_exacte_midi(self, historique_15):
        """heure_cos pour midi (heure=12) → cos(pi) = -1."""
        t = pd.Timestamp("2025-06-15 12:00:00")
        result = generate_features(historique_15, t)
        assert abs(result['heure_cos'].iloc[0] - (-1.0)) < 1e-6


# ─────────────────────────────────────────────────────────────
# TESTS : valeurs des lags
# ─────────────────────────────────────────────────────────────

class TestFeaturesLags:

    def test_lag1_est_derniere_valeur(self, historique_15, next_time):
        """lag_1 = dernière valeur de l'historique (t-3h)."""
        result = generate_features(historique_15, next_time)
        assert result['lag_1'].iloc[0] == historique_15.iloc[-1]

    def test_lag2_est_avant_derniere(self, historique_15, next_time):
        """lag_2 = avant-dernière valeur (t-6h)."""
        result = generate_features(historique_15, next_time)
        assert result['lag_2'].iloc[0] == historique_15.iloc[-2]

    def test_lag3_est_troisieme_avant(self, historique_15, next_time):
        """lag_3 = 3ème valeur avant la fin (t-9h)."""
        result = generate_features(historique_15, next_time)
        assert result['lag_3'].iloc[0] == historique_15.iloc[-3]

    def test_lag24h_est_8_pas_avant(self, historique_15, next_time):
        """lag_24h = valeur d'il y a 8 pas (= 24h avec pas de 3h)."""
        result = generate_features(historique_15, next_time)
        assert result['lag_24h'].iloc[0] == historique_15.iloc[-8]


# ─────────────────────────────────────────────────────────────
# TESTS : statistiques glissantes
# ─────────────────────────────────────────────────────────────

class TestFeaturesGlissantes:

    def test_moyenne_mobile_24h_correcte(self, historique_15, next_time):
        """moyenne_mobile_24h = moyenne des 8 dernières obs."""
        result = generate_features(historique_15, next_time)
        expected = historique_15.iloc[-8:].mean()
        assert abs(result['moyenne_mobile_24h'].iloc[0] - expected) < 1e-10

    def test_ecart_type_24h_positif(self, historique_15, next_time):
        """ecart_type_24h doit être >= 0."""
        result = generate_features(historique_15, next_time)
        assert result['ecart_type_24h'].iloc[0] >= 0

    def test_tendance_3h_correcte(self, historique_15, next_time):
        """tendance_3h = lag_1 - lag_2 (direction du changement récent)."""
        result = generate_features(historique_15, next_time)
        expected = historique_15.iloc[-1] - historique_15.iloc[-2]
        assert abs(result['tendance_3h'].iloc[0] - expected) < 1e-10

    def test_pas_de_valeurs_nulles(self, historique_15, next_time):
        """Aucune feature ne doit être NaN avec un historique suffisant."""
        result = generate_features(historique_15, next_time)
        assert result.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────
# TESTS : cohérence temporelle (anti-fuite)
# ─────────────────────────────────────────────────────────────

class TestAntiLeakage:

    def test_features_utilisent_uniquement_le_passe(self, historique_15, next_time):
        """
        Les lags doivent provenir uniquement de l'historique passé,
        jamais du next_time lui-même (pas de fuite de données du futur).
        """
        result = generate_features(historique_15, next_time)
        # lag_1 doit être la dernière valeur AVANT next_time
        assert result['lag_1'].iloc[0] == historique_15.iloc[-1]
        # next_time est strictement APRÈS la dernière observation
        assert next_time > historique_15.index[-1]
