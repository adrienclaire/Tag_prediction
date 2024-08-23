# Projet 5 : Catégorisation Automatique des Questions Stack Overflow

## Objectif du Projet

Le but de ce projet est de développer un modèle de classification supervisée capable d’attribuer automatiquement des tags pertinents à chaque question posée sur une plateforme comme Stack Overflow. Ces tags sont essentiels pour aider les utilisateurs à trouver des questions similaires, regrouper les sujets connexes et améliorer la navigation sur le site. Le défi consiste à créer un modèle précis et efficace qui comprend le contenu des questions et prédit les tags les plus appropriés.

## Architecture du Projet

Le projet est organisé en plusieurs dossiers et fichiers comme suit :

- **.ebextensions/** : Contient les configurations spécifiques pour AWS Elastic Beanstalk.
  
- **templates/** : Contient les fichiers HTML utilisés par l'API Flask pour l'interface utilisateur.
  
- **tests/** : Contient les tests unitaires et d'intégration pour valider le bon fonctionnement de l'API et des scripts.
  - `test_api.py` : Tests unitaires pour l'API Flask. (à créer)
  - `test_preprocessing.py` : Tests unitaires pour les scripts de pré-traitement. (à créer)

- **Procfile** : Fichier utilisé par Heroku et Elastic Beanstalk pour indiquer comment lancer l'application.

- **README.md** : Ce fichier contient une description complète du projet, des instructions d'installation, et des détails sur l'architecture du projet.

- **Note_technique.pdf** : Document contenant la note technique du projet, détaillant la mise en œuvre de l'approche MLOps.

- **app.py** : Fichier principal de l'API Flask pour prédire les tags à partir d'une question.

- **buildspec.yml** : Fichier de configuration utilisé par AWS CodeBuild pour automatiser le processus de construction de l'application.

- **mlb.pkl** : Fichier contenant le modèle binaire MultiLabelBinarizer utilisé pour les prédictions.

- **requirements.txt** : Fichier listant les dépendances Python nécessaires pour exécuter le projet.

- **utils.py** : Fichier contenant des fonctions utilitaires pour le projet, telles que le prétraitement des données.

- **__init__.py** : Fichier pour indiquer que le répertoire contient un module Python.

## Installation

### Prérequis

- Python 3.8+
- pip
- virtualenv (optionnel mais recommandé)

## Déployer sur AWS Elastic Beanstalk :

Pour déployer l'application sur AWS Elastic Beanstalk, utilisez le fichier buildspec.yml et suivez les instructions pour configurer le déploiement continu.

## Packages Utilisés
**Flask** : Pour la création de l'API web permettant de prédire les tags à partir d'une question.  
**scikit-learn** : Pour les transformations des données et les algorithmes de classification.  
**Pandas** : Pour la manipulation et l'analyse des données.  
**AWS SDK (boto3)** : Pour l'intégration avec les services AWS comme S3 et Elastic Beanstalk.  
**MLFlow** : Pour le suivi des expérimentations et la gestion des modèles.  
**Gunicorn** : Pour déployer l'application Flask sur un serveur de production.  

### Contributeurs
Adrien Claire - Développeur principal et auteur du projet.

