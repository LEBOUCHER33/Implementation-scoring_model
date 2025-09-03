# Implementation-scoring_model

L'objectif du projet sera d'élaborer et sélectionner un modèle de classification automatique binaire supervisée adapté aux données bancaires, scorant une probabilité de solvabilité des clients afin de leur accorder ou non un crédit à la consommation.
Ce modèle devra être accessible via une API.
Il s'agira également d'assurer l'automatisation et l'intégration continue de cette interface API sur le cloud.

Workflow :

- réaliser le data exploring des données clients
- tester et entrainer différents algorithmes de classification automatique supervisée en tenant compte du biais de représentativité des classes en utilisant l'outil de tracking de MLFlow
- identifier les features importance globales et locales et définir un scoring 
- évaluer et sélectionner le modèle le plus pertinent et performant via l'UI de MLFlow
- enregistrer le modèle le plus performant via MLFlow registry
- développer une API permettant de réaliser des tests unitaires et l'inférence du modèle
- déployer le modèle sur le cloud via l'API développée
- gérer le versioning du code de l'API pour assurer son deploiement continue

Requirements :

- mlflow
- scikit-learn
- pandas
- numpy
- matplotlib
- lightgbm
- xgboost
- Imbalanced-learn
