"""
Le Dockerfile est un fichier texte qui va permettre de décrire l'environnement et les dépendances nécessaires à la bonne exécution d'une application.
Il sert à décrire l'image docker.

C'est une série d'instructions codifiées qui construiront l'image docker :
    - image de base : l'environnement Python (ou autre langage) + OS minimal (FROM)
    - les dépendances nécessaires à l'execution de l'application 
    - les scripts liés à l'application : création de l'application, lancement de l'application / modèle entrainé / pipeline (COPY)
    - la commande par défaut qui s'éxécutera lors du lancement du conteneur (CMD)

Workflow docker :

    1- création de l'image (Dockerfile)
    2- contruction de l'image
        ```bash
        docker build -t api_scoring:1.0 .       
        # nom de l'image : version
        ```
    3- lancement du conteneur

        3-1 en local

        ```bash
        docker run -p 127.0.0.0.8000:8000 --name api_scoring_contener api_scoring:1.0
        # l'url locale sera : http://127.0.0.1:8000/predict
        # -p relie le port du conteneur au port local // --name donne un nom du conteneur // nom de l'image à lancer
       ```
       si on veut utiliser un autre port 
        ```bash
        docker run -p 5000:8000 --name api_scoring_contener api_scoring:1.0
        # l'url sera : http://127.0.0.1:5000/predict
       ```
        3-2 sur le cloud

        on push le repo git avec le conteneur sur la PlateformeasaService (Render)
        la PaaS gére le build et le launch
        l'url sera cette fois avec le protocole https pour encrypter les données :
            https://my_api.onrender.com/predict

"""

# image docker de base avec Python
FROM python:3.12-alpine 
# ou FROM python:3.12-slim (si beaucoup de lib)

# repertoire de travail dans le conteneur (là où seront copiés les fichiers)
WORKDIR /app

# copie des dépendances 
COPY requirements.txt /

# installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# copie du script lié à l'application
COPY Scripts/api.py /Scripts/api.py

# copie du pipeline et du modèle entrainé
COPY Reports/pipeline_final.pkl /Reports/pipeline_final.pkl
COPY Reports/best_model.pkl /Reports/best_model.pkl

# expose le port local de l'api
EXPOSE 8000

# execution de l'application lors du lancement du conteneur
CMD python -m uvicorn Scripts.api:app --host 0.0.0.0 --port 8000
#["python", "-m", "uvicorn", "Scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]

