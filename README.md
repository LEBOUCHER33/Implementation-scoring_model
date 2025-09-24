# Implementation-scoring_model


## Objectif :

L'objectif du projet sera triple :

1- Sélectionner et implémenter un modèle de classification automatique binaire supervisé, adapté aux données bancaires, scorant une probabilité de solvabilité des clients afin de leur accorder ou non un crédit à la consommation 

2- Développer une API sur le cloud pour accéder à l'inférence du modèle 

3- Assurer l'intégration et le déploiement continus de l'API


## Workflow :

1- ML_training : évaluation et comparaison des performances de différents modèles de classification suivant un scoring métier.

2- Implémentation d'une API sur une solution cloud pour l'inférence du modèle entrainé.

3- Automatisation et intégration continue de cette interface API.


## Partie 1 :

- réaliser le data exploring et le feature engineering des données clients; notebook_1
- définir un scoring métier; notebook_2
- tester et entrainer différents algorithmes de classification automatique supervisée en tenant compte du biais de représentativité des classes; notebook_2
- utiliser l'outil de tracking de MLFlow pour logger les métriques de performances et les combinaisons d'hyperparamètres; notebook_2 
- sélectionner le modèle le plus pertinent et le plus performant, analyser la feature importance et la qualité des prédictions; notebook_2 
- enregistrer le modèle; notebook_2 


## Partie 2 :

### 2-1 création d'une API REST

- développer une API pour tester l'inference du modèle; script python api.py
- réaliser des tests unitaires de l'API; test_api.py
- lancer l'api sur le server local
```bash
python -m uvicorn api:app --reload
```
- rédiger un script utilisateur pour tester le fonctionnement de l'api avec des requêtes http en local; notebook_3

### 2-2 déploiement de l'API

- créer une image docker à l'aide d'un Dockerfile à la base du repo git
- construire l'image docker
```bash
docker build -t api_scoring:1.0 .
```
- lancer le conteneur et tester le fonctionnement de l'application en local
```bash
docker run -p 8000:8000 --name api_scoring_container api_scoring:1.0
```
- déployer l'API sur une solution cloud, Render lib
- utiliser Streamlit pour créer une interface utilisateur
- gérer le versioning du code de l'API pour assurer son deploiement continu



## Highlights :

- Data exploring / data engineering : 
    - nettoyage 
    - analyse des distributions 
    - encoding 
    - analyse des corrélations
    - imputation 
    - création de nouvelles variables

- ML et Classifieurs binaires : modèles, métriques, performances

- Cycle de vie d'un projet de ML (mlflow)

- Datadrift (evidently)

- API

- tests unitaires

- outils MLOps
