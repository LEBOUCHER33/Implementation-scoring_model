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

- réaliser le data exploring et le feature engineering des données clients
- définir un scoring métier
- tester et entrainer différents algorithmes de classification automatique supervisée en tenant compte du biais de représentativité des classes
- utiliser l'outil de tracking de MLFlow pour logger les métriques de performances et les combinaisons d'hyperparamètres 
- sélectionner le modèle le plus pertinent et le plus performant, analyser la feature importance et la qualité des prédictions 
- enregistrer le modèle 


## Partie 2 :

- développer une API en local pour tester l'inference du modèle
- réaliser des tests unitaires de l'API en local
- déployer l'API sur une solution cloud / docker
- réaliser des tests unitaires de l'API en production
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
