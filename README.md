# Prévision de Température — Strasbourg

Système complet de prévision météo basé sur le Machine Learning pour prédire la température à Strasbourg sur un horizon de **72 heures** (pas de 3 heures).  
Modèle champion : **LightGBM** (MAE = 0.742°C) entraîné sur données historiques Open-Meteo + observations récentes.

---

## Architecture & Data Flow

```
Open-Meteo API
├── Archive API  (données jusqu'à J-5)
└── Forecast API (données temps réel)
         │
         ▼
   fetch_data.py
   ├── fetch_from_archive()   → 35 jours validés
   ├── fetch_from_forecast()  → 7 derniers jours
   └── parse_and_insert()     → agrège 1H→3H, insère SQLite
         │
         ▼
   SQLite: meteo_predictions.db
   ├── observations_historiques  (date_time PK, temperature)
   └── predictions_batch         (id, date_inference, date_cible, temperature_predite, model_id)
         │
    ┌────┴────────────────┐
    ▼                     ▼
train.py            batch_predict.py
├── load_csv()      ├── load_recent_data_from_db(limit=15)
├── load_sqlite()   ├── generate_features()   → 11 features
├── build_features()├── boucle autoregressive × 24 pas
├── split_data()    ├── joblib.load(model_lightgbm.pkl)
└── LightGBM.fit()  └── insert_predictions()  → SQLite
    ├── MLflow tracking        │
    └── model_lightgbm.pkl ───┘   (batch_predict recharge le .pkl
                                   à chaque exécution ; l'API n'a
                                   pas besoin d'être redémarrée)
         │
         ▼
   API FastAPI (port 8000)
   (lecture seule sur SQLite — aucun chargement de .pkl)
   ├── GET /                   → description du service
   ├── GET /health             → statut + base de données
   ├── GET /version            → COMMIT_ID + modèle + environnement
   ├── GET /predictions        → prédictions SQLite
   ├── GET /monitoring/data    → MAE / RMSE (prédictions vs réalité)
   └── GET /metrics            → exposition Prometheus
         │
         ▼
   Prometheus (port 9090)     scrape toutes les 5s
         │
         ▼
   Grafana (port 3000)
   ├── Taux de requêtes / erreurs
   ├── Latence p50 / p95 / p99
   ├── MAE & RMSE du modèle (°C)
   └── Évolution MAE/RMSE dans le temps
```

---

## Technologies utilisées

| Technologie | Version | Usage |
|---|---|---|
| **Python** | 3.10 | Langage principal |
| **LightGBM** | >=4.0.0 | Modèle champion (régression) |
| **scikit-learn** | >=1.3.0 | Métriques, baseline |
| **statsmodels** | >=0.14.0 | SARIMAX, tests statistiques |
| **pandas** | >=2.0.0 | Manipulation des données |
| **numpy** | >=1.24.0 | Calculs numériques |
| **MLflow** | >=2.8.0 | Tracking expériences, registre modèle |
| **joblib** | >=1.3.0 | Sérialisation du modèle (.pkl) |
| **FastAPI** | >=0.104.0 | API REST |
| **uvicorn** | >=0.24.0 | Serveur ASGI |
| **prometheus-client** | >=0.19.0 | Exposition métriques Prometheus |
| **pytest** | >=7.4.0 | Tests unitaires et d'intégration |
| **httpx** | >=0.25.0 | Tests HTTP contre le conteneur Docker |
| **Docker** | latest | Conteneurisation de l'API |
| **Docker Compose** | latest | Orchestration API + Prometheus + Grafana |
| **GitHub Actions** | — | Pipeline CI/CD (build, test, lint, docker) |

---

## Commandes

### 1. Initialisation (une seule fois)

```bash
pip install -r requirements.txt

# Créer les tables SQLite
python src/data/db_manager.py

# Entraîner le modèle LightGBM (CSV + SQLite)
python src/training/train.py
```

### 2. Flux de données (à relancer régulièrement)

```bash
# Récupérer les données Open-Meteo (archive + forecast)
python src/data/fetch_data.py

# Réentraîner sur les données récentes
python src/training/train.py

# Générer les prédictions 72h (24 points × 3h)
python src/inference/batch_predict.py
```

### 3. Lancer la stack complète (Docker)

```bash
# Build + lancement API + Prometheus + Grafana
docker compose up --build

# En arrière-plan
docker compose up --build -d

# Arrêter
docker compose down
```

| Service | URL | Identifiants |
|---|---|---|
| API docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

### 4. Alimenter les métriques MAE/RMSE dans Grafana

```bash
curl "http://localhost:8000/monitoring/data?start_date=2026-03-06&end_date=2026-04-01"
```

Les panels MAE/RMSE se mettent à jour dans les 5 secondes.

### 5. Tests

```bash
# Tous les tests (68 tests, dont 1 xfail attendu)
pytest tests/ -v

# Tests HTTP contre le conteneur Docker (nécessite docker compose up)
pytest tests/test_api_docker.py -v
```

### 6. MLflow

```bash
mlflow ui \
  --backend-store-uri file://$(pwd)/model/registry \
  --port 5000
# Accès : http://127.0.0.1:5000
```

---

## Endpoints API

| Endpoint | Méthode | Paramètres | Description |
|---|---|---|---|
| `/` | GET | — | Description du service |
| `/health` | GET | — | Statut API + base de données |
| `/version` | GET | — | COMMIT_ID + modèle + environnement |
| `/predictions` | GET | `date` (optionnel) | Prédictions stockées en SQLite |
| `/monitoring/data` | GET | `start_date`, `end_date` | Prédictions vs réalité → MAE/RMSE |
| `/metrics` | GET | — | Métriques Prometheus (scrape automatique) |

**Exemple `/predictions`** :
```bash
curl "http://localhost:8000/predictions?date=2026-04-07"
```

**Exemple `/monitoring/data`** :
```bash
curl "http://localhost:8000/monitoring/data?start_date=2026-03-06&end_date=2026-04-01"
```

---

## Métriques Prometheus

| Métrique | Type | Labels | Description |
|---|---|---|---|
| `prediction_requests_total` | Counter | `endpoint` | Nombre de requêtes par endpoint |
| `prediction_request_errors_total` | Counter | `endpoint` | Nombre d'erreurs (HTTP ≥400) |
| `prediction_latency_seconds` | Histogram | `endpoint` | Latence des requêtes (p50/p95/p99) |
| `model_info` | Gauge | `model_version` | Modèle actif (valeur = 1) |
| `model_mae` | Gauge | — | MAE prédictions vs réalité (°C) |
| `model_rmse` | Gauge | — | RMSE prédictions vs réalité (°C) |
| `model_prediction_count` | Gauge | — | Nombre de points évalués |

> `model_mae` et `model_rmse` sont mis à jour uniquement lors d'un appel à `/monitoring/data`. Pour les maintenir à jour dans Prometheus, planifie un appel régulier (ex : cron, systemd timer).

> **L'API FastAPI ne charge jamais le fichier `.pkl`**. Elle lit uniquement les prédictions pré-calculées dans SQLite. `model_info` expose simplement la constante `MODEL_VERSION` définie au démarrage de l'API.

---

## Panels Grafana

| Panel | Type | Requête PromQL |
|---|---|---|
| Taux de Requêtes (req/s) | Timeseries | `rate(prediction_requests_total[1m])` |
| Taux d'Erreurs (req/s) | Timeseries | `rate(prediction_request_errors_total[1m])` |
| Latence p50 / p95 / p99 | Timeseries | `histogram_quantile(0.50/0.95/0.99, rate(..._bucket[1m]))` |
| Modèle chargé | Stat | `model_info` |
| MAE du Modèle (°C) | Stat | `model_mae` |
| RMSE du Modèle (°C) | Stat | `model_rmse` |
| Nombre de prédictions évaluées | Stat | `model_prediction_count` |
| Évolution MAE / RMSE dans le temps | Timeseries | `model_mae` + `model_rmse` |

---

## Features du modèle (11 features)

| Feature | Type | Calcul |
|---|---|---|
| `heure_sin` | Cyclique | sin(2π × heure / 24) |
| `heure_cos` | Cyclique | cos(2π × heure / 24) |
| `annee_sin` | Cyclique | sin(2π × jour_année / 365.25) |
| `annee_cos` | Cyclique | cos(2π × jour_année / 365.25) |
| `lag_1` | Lag | Température 3h avant |
| `lag_2` | Lag | Température 6h avant |
| `lag_3` | Lag | Température 9h avant |
| `lag_24h` | Lag | Température 24h avant (8 pas × 3h) |
| `moyenne_mobile_24h` | Rolling | Moyenne sur les 8 derniers points |
| `ecart_type_24h` | Rolling | Écart-type sur les 8 derniers points |
| `tendance_3h` | Momentum | lag_1 − lag_2 |

---

## Schéma SQLite

```sql
-- Observations réelles (fetch_data.py)
CREATE TABLE observations_historiques (
    date_time  TEXT PRIMARY KEY,   -- "YYYY-MM-DD HH:MM:SS"
    temperature REAL               -- °C
);

-- Prédictions générées (batch_predict.py)
CREATE TABLE predictions_batch (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date_inference      TEXT,      -- Moment de l'inférence
    date_cible          TEXT,      -- Timestamp prédit
    temperature_predite REAL,      -- °C
    model_id            TEXT       -- ex: "LightGBM_V1"
);
```

---

## CI/CD — GitHub Actions (4 jobs)

Pipeline déclenché sur chaque **push** et **Pull Request** vers `master`.

### Job 1 — Build & Push Docker (`build`)
- Construit l'image depuis le `Dockerfile`
- Injecte `GIT_COMMIT_ID` (SHA du commit) comme build-arg → `ENV COMMIT_ID`
- Pousse vers **ghcr.io** avec les tags `sha-<commit>` et `latest`
- Cache GitHub Actions pour accélérer les builds suivants
- Authentification via `secrets.GITHUB_TOKEN` (aucun secret en clair)

### Job 2 — Tests unitaires & intégration (`tests`)
- Python 3.10, installe `requirements.txt`
- Lance `pytest tests/ -v` (68 tests, 1 xfail attendu)
- Variable `COMMIT_ID = ${{ github.sha }}` injectée
- Bloquant si un test échoue

### Job 3 — Qualité du code (`lint`)
- `flake8 api/ src/ tests/ --select=E9,F63,F7,F82 --max-line-length=120`
- Vérifie : erreurs de syntaxe, variables non définies, assertions invalides
- Ne bloque pas sur les avertissements de style

### Job 4 — Tests HTTP Docker (`test`)
- **Dépend de `build`** (`needs: build`)
- Pull l'image `:latest` depuis ghcr.io
- Lance le conteneur sur le port 8000 avec `COMMIT_ID` injecté
- Health-check sur `/health` (15 tentatives × 2s)
- `pytest tests/test_api_docker.py -v` (14 tests HTTP réels)
- Arrête et supprime le conteneur après les tests (`if: always()`)

---

## Tests (68 tests)

| Fichier | Tests | Description |
|---|---|---|
| `tests/test_api.py` | 17 | Endpoints FastAPI (TestClient) |
| `tests/test_api_docker.py` | 14 | Tests HTTP contre conteneur Docker réel |
| `tests/test_batch_predict.py` | 7 | Inférence autoregressive (mocks) |
| `tests/test_db_manager.py` | 10 | Insertion/lecture SQLite (DB en mémoire) |
| `tests/test_features.py` | 20 | Feature engineering (11 features, anti-leakage) |
