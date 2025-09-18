"""
Script de test de l'API créée avec FastAPI.

"""

import pytest
from fastapi.testclient import TestClient

# TestClient est une classe de FastAPI qui permet de simuler des requêtes HTTP pour tester les endpoints de l'API


from api import app  # on importe l'instance de l'API créée dans api.py

# on crée un client de test pour envoyer des requêtes à l'API
client = TestClient(app)
# on crée une fonction de test pour l'endpoint racine
def test_read_root():
    response = client.get("/")  # on envoie une requête GET à l'endpoint racine
    assert response.status_code == 200  # on vérifie que le code de statut est 200 (OK)
    assert response.json() == {"message": "Welcome to the credit scoring API. Use the /predict endpoint to get predictions."}  # on vérifie que la réponse est correcte

# on crée une fonction de test pour l'endpoint de prédiction
