"""
Script de création d'une API avec FastAPI.

Objectif : Fournir une interface pour interagir avec un modèle de prédiction 

Worflow :

- loader le pipeline de prédiction
- définir l'API avec FastAPI
- définir les endpoints pour les prédictions

"""

# 1- import des bibliothèques nécessaires

from fastapi import FastAPI
import pickle
import pandas as pd




# 2- chargement du pipeline de prédiction

with open("pipeline_lgbmclass.pkl", "rb") as f:
    model_pipeline = pickle.load(f)




# 3- définir l'API = une instance de FastAPI

app = FastAPI()

# création d'une opération de chemin GET à la racine de l'API
@app.get("/")  # endpoint racine : la fonction en dessous sera exécutée lorsqu'une requête GET est envoyée à la racine de l'API
def read_root(): 
    return {"message": "Bienvenue sur l'API de scoring de crédit"}


# création d'un endpoint de prédiction
@app.post("/predict")  # endpoint de prédiction : la fonction en dessous sera exécutée lorsqu'une requête POST est envoyée à /predict
def predict(data): 
    """
    fonction de prédiction
    """
    # 1- conversion du dictionnaire en DataFrame pandas
    # cas 1 ou plusieurs individus :
    if isinstance(data, dict):
    # si un seul individu
        input_data = pd.DataFrame([data])
    # si plusieurs individus
    else :
        input_data = pd.DataFrame(data)
    
    # faire la prédiction avec le pipeline chargé
    prediction = model_pipeline.predict(input_data)
    prediction_proba = model_pipeline.predict_proba(input_data)  # probabilités des classes
    prediction_proba_1 = prediction_proba[:, 1]  # probabilité de la classe 1 (risque de défaut de paiement)
    prediction_proba_0 = prediction_proba[:, 0]  # probabilité de la classe 0 (pas de risque de défaut de paiement)
    
    # retourner la prédiction et la probabilité associée
    return {
        "prediction": int(prediction),  # convertir en int pour une meilleure lisibilité dans la réponse JSON
        "probability": prediction_proba.tolist(),  # convertir en liste pour une meilleure lisibilité dans la réponse JSON
        "probability_0": float(prediction_proba_0),  # probabilité de la classe 0
        "probability_1": float(prediction_proba_1)   # probabilité de la classe 1
    }