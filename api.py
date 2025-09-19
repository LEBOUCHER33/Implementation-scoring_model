"""
Script de création d'une API avec FastAPI.

Objectif : Fournir une interface pour interagir avec un modèle de prédiction de scoring de crédit.
    - recevoir des données en entrée (format JSON)
    - retourner la prédiction, la probabilité associée (format JSON) et les 5 principales features qui ont influencé la prédiction (explainabilité)

Worflow :

- loader le pipeline de prédiction
- définir l'explainabilité avec SHAP
- définir l'API avec FastAPI
- définir les endpoints pour les prédictions

"""

# 1- import des bibliothèques nécessaires

from fastapi import FastAPI, Request
import pickle
import pandas as pd
import shap





# 2- chargement du pipeline de prédiction

with open("./Scripts/pipeline_final.pkl", "rb") as f:
    model_pipeline = pickle.load(f)




# 3- définition de l'explainabilité avec SHAP

explainer = shap.TreeExplainer(model_pipeline.named_steps['model'])


# 4- définir l'API avec FastAPI

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
def predict(data : dict): 
    """
    _Summary_ : fonction de prédiction qui reçoit les données en format JSON et retourne la prédiction et la probabilité de solvabilité associée.
    _Arguments_ :
        data : données en format JSON (un ou plusieurs individus)
    _Returns_ :
        dict : dictionnaire contenant la prédiction et la probabilité associée
    """
    # 1- conversion du dictionnaire en DataFrame pandas pour pouvoir faire la prédiction
    # un seul individu = un dictionnaire / plusieurs individus = liste de dictionnaires
    input_data = pd.DataFrame (data if isinstance(data, list) else [data])
    
    # 2- faire la prédiction avec le pipeline chargé
    prediction = model_pipeline.predict(input_data)
    prediction_proba = model_pipeline.predict_proba(input_data)[:,1]  # probabilité d'être un mauvais payeur (classe 1)

    # 3- explainabilité avec SHAP
    data_transformed = model_pipeline.named_steps['preprocessor'].transform(input_data)  # on applique le préprocesseur aux données d'entrée
    shap_values = explainer.shap_values(data_transformed)  # on calcule les valeurs SHAP

    # on affiche les 5 features les plus importantes pour chaque individu
    explations = []
    for i in range(len(input_data)):
        features_shap = dict(zip(input_data.columns, shap_values[i].tolist()))  # on associe chaque feature à sa valeur SHAP
        top_5_features = sorted(features_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]  # on trie les features par valeur absolue de SHAP et on prend les 5 premières
        explations.append(top_5_features)

    # retourner la prédiction et la probabilité associée
    return {
        "prediction": prediction.tolist(),
        "probabilite d'être non solvable": prediction_proba.tolist(),
        "les 5 features les plus influentes sur le prediction sont ": explations
    }