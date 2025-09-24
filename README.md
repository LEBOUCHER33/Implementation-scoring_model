# Implementation-scoring_model


## Objectif :

L'objectif du projet sera triple :

1- Sélectionner et implémenter un modèle de classification automatique binaire supervisé, adapté aux données bancaires, scorant une probabilité de solvabilité des clients afin de leur accorder ou non un crédit à la consommation (classe 0 ou 1)

2- Développer une API et une interface utilisateur sur le cloud pour accéder à l'inférence du modèle 

3- Assurer l'intégration et le déploiement continus de l'API et son versionning avec git


## Workflow :

1- ML_training : évaluation et comparaison des performances de différents modèles de classification suivant un scoring métier.

2- Implémentation et déploiement d'une API sur une solution cloud pour l'inférence du modèle entrainé.

3- Automatisation et intégration continue de cette interface API.


### Partie 1 :

- réaliser le data exploring et le feature engineering des données clients (notebook_1)
- définir un scoring métier (notebook_2)
- tester et entrainer différents algorithmes de classification automatique supervisée en tenant compte du biais de représentativité des classes (notebook_2)
- utiliser l'outil de tracking de MLFlow pour logger les métriques de performances et les combinaisons d'hyperparamètres (notebook_2) 
- sélectionner le modèle le plus pertinent et le plus performant, analyser la feature importance et la qualité des prédictions (notebook_2) 
- enregistrer le modèle (notebook_2) 


### Partie 2 :

#### 2-1 création d'une API REST

- développer une API pour tester l'inference du modèle (script python api.py)
- réaliser des tests unitaires de l'API (test_api.py)
- lancer l'api sur le server local
```bash
python -m uvicorn api:app --reload
```
- rédiger un script utilisateur pour tester le fonctionnement de l'api avec des requêtes http en local (notebook_3)
```python
url = 'http://127.0.0.1.8000:/predict"
```

#### 2-2 déploiement de l'API

- créer une image docker à l'aide d'un Dockerfile à la base du repo git = instructions pour définir l'image docker :
    - environnement d'execution
    - dependances
    - fichiers
    - commande d'execution
- tester le conteneur en local :
    - construire l'image docker
```bash
docker build -t api_scoring:1.0 .
```
   - lancer le conteneur 
```bash
docker run -p 8000:8000 --name api_scoring_container api_scoring:1.0
```
   - tester le fonctionnement de l'application en local avec des requêtes http
```python
url = 'http://127.0.0.1.8000:/predict"
```
- déployer l'API sur une solution cloud, Render (gère le build et le launch)
- tester l'application avec des requêtes https (protocole crypté des données, obligatoire sur un serveur exterieur)
```python
url = 'https://api.onrender.com/predict'
```
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

- Cycle de vie d'un projet de ML (mlflow) : experience tracking, model register, model deployment

- Datadrift (evidently)

- API REST (FastAPI, uvicorn)

- tests unitaires (TestClient)

- outils MLOps (docker.desktop, Render)
