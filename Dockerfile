"""
le Dockerfile est le fichier de configuration pour 'builder' le conteneur

On va y définir :
    - l'environnement d'execution (Python)
    - l'application (api)
    - les dépendances nécessaires
    - le port utilisé par l'api
    - la commande pour lancer l'api

"""


# 1- Image de base
FROM python:3.11-slim

# 2- Définir le répertoire de travail
WORKDIR /app

# 3- Copier les dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4- Copier le reste du code
COPY . .

# 5- Exposer le port utilisé par l'API
EXPOSE 5000

# 6- Commande pour lancer l'API

# Pour FastAPI (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
