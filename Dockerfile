# Dockerfile — API Météo BIHAR2026 / Strasbourg

FROM python:3.10-slim

# Argument injecté par GitHub Actions (commit SHA)
ARG GIT_COMMIT_ID=local
ENV COMMIT_ID=${GIT_COMMIT_ID}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY api/   ./api/
COPY src/   ./src/
COPY data/  ./data/
COPY model/ ./model/

EXPOSE 8000

# Initialise la base SQLite (tables vides) si elle n'existe pas encore
RUN python -c "from src.data.db_manager import init_db; init_db()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
