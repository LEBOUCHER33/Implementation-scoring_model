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

with open("./Scripts/pipeline_final.pkl", "rb") as f:
    model_pipeline = pickle.load(f)




# 3- définir l'API avec FastAPI

app = FastAPI()


# app est l'instance de l'API, on va définir les endpoints en utilisant app

# création d'un endpoint de test
@app.get("/")  # endpoint racine : la fonction en dessous sera exécutée lorsqu'une requête GET est envoyée à /
def read_root():        
    """
    _Summary_ : fonction de test qui retourne un message de bienvenue.
    _Returns_ :
        dict : dictionnaire contenant le message de bienvenue
    """
    return {"message": "Welcome to the credit scoring API. Use the /predict endpoint to get predictions."}




# création d'un endpoint de prédiction
@app.post("/predict")  # endpoint de prédiction : la fonction en dessous sera exécutée lorsqu'une requête POST est envoyée à /predict
# on prend en entrée de la fonction, les données formatées en JSON (dictionnaire ou liste de dictionnaires)
def predict(data: dict): 
    """
    _Summary_ : fonction de prédiction qui reçoit les données en format JSON et retourne la prédiction et la probabilité de solvabilité associée.
    _Arguments_ :
        data : données en format JSON (un ou plusieurs individus)
    _Returns_ :
        dict : dictionnaire contenant la prédiction et la probabilité associée
    """
    # 1- conversion du dictionnaire en DataFrame pandas pour pouvoir faire la prédiction
    # cas 1 : un seul individu = un dictionnaire
    if isinstance(data):
        input_data = pd.DataFrame([data])
    # cas 2 : plusieurs individus = liste de dictionnaires
    elif isinstance(data, list):
        input_data = pd.DataFrame(data)
    
    # faire la prédiction avec le pipeline chargé
    prediction = model_pipeline.predict(input_data)
    prediction_proba = model_pipeline.predict_proba(input_data)  # probabilités des classes
    
    # retourner la prédiction et la probabilité associée
    return {
        "prediction": prediction.tolist(),
        "probability": prediction_proba.tolist()
    }