# Implementation-scoring_model

L'objectif du projet sera de sélectionner et d'implémenter un modèle de classification automatique binaire supervisée adapté aux données bancaires, scorant une probabilité de solvabilité des clients afin de leur accorder ou non un crédit à la consommation.

Plusieurs classifieurs seront évalués et comparés sur leurs perfomances suivant le scoring métier.

L'inférence du modèle entrainé le plus performant se fera via une API sur une solution cloud.

Il s'agira également d'assurer l'automatisation et l'intégration continue de cette interface API.

Workflow :

- réaliser le data exploring et le feature engineering des données clients
- définir un scoring métier, tester et entrainer différents algorithmes de classification automatique supervisée en tenant compte du biais de représentativité des classes
- utiliser l'outil de tracking de MLFlow pour logger les métriques de performances et les combinaisons d'hyperparamètres 
- sélectionner le modèle le plus pertinent et le plus performant, analyser la feature importance et la qualité des prédictions sur le dataset de test
- enregistrer le modèle via MLFlow registry
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
- pyngrok
