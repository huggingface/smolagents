# Utiliser une image de base plus légère
FROM python:3.12-slim as builder

# Définir les variables d'environnement pour optimiser Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev

# Copier uniquement les fichiers nécessaires pour l'installation des dépendances
# COPY requirements.txt .
# COPY setup.py .
# COPY server.py .

# Installer les dépendances Python dans un environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install uv
RUN uv pip install -r requirements.txt

# Étape finale avec une image plus légère
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Copier l'environnement virtuel de l'étape précédente
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/server.py .

# Exposer le port
EXPOSE 65432

# Utiliser un utilisateur non-root pour plus de sécurité
RUN useradd -m appuser
USER appuser

CMD ["python", "server.py"]
