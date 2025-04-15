import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = joblib.load('xgboost.pkl')

import pandas as pd
import joblib
from termcolor import colored
import os
from sklearn.preprocessing import MinMaxScaler
import logging



# Liste des maladies
MALADIES = ['Arthrose', 'Asthme', 'AVC', 'Bronchite', 'Cancer_foie',
       'Cancer_poumon', 'Cancer_rein', 'Colite_ulcereuse', 'Depression',
       'Diabete', 'Eczema', 'Grippe', 'Hepatite_B', 'Hypertension',
       'Hyperthyroidie', 'Infection_urinaire', 'Maladie_coronarienne',
       'Maladie_Crohn', 'Maladie_Parkinson', 'Migraine', 'Osteoporose',
       'Paludisme', 'Pancreatite', 'Pneumonie', 'Polyarthrite_rhumatoide',
       'Psoriasis', 'Rhinite_allergique', 'Rhume', 'Tuberculose']

FEATURES = [
          'Fievre', 'Toux', 'Fatigue', 'Difficulte_respiratoire', 'Courbatures',
    'Maux_de_tete', 'Nez_bouche', 'Eternuements', 'Demangeaisons', 'Rougeurs',
    'Sifflements', 'Palpitations', 'Perte_poids', 'Douleur_abdominale', 'Nausees',
    'Raideur', 'Tristesse', 'Gonflement_ganglions', 'Diarrhee', 'Sueurs_nocturnes',
    'Tremblements', 'Plaques', 'Desquamation', 'Soif_intense', 'Cicatrisation_lente',
    'Jaunisse', 'Sang_urines', 'Expectorations', 'Frissons', 'Douleur_thoracique',
    'Essoufflement', 'Brulures_urinaires', 'Paralysie_faciale', 'Trouble_parole',
    'Saignements_rectaux', 'Fractures_frequentes', 'Tension_arterielle',
    'Niveau_Cholesterol', 'Age', 'Genre'
    ]

# Classification complète de gravité avec recommandations
GRAVITE_MALADIES = {
    # Urgences absolues
    'AVC': {
        'niveau': 'urgence absolue',
        'conseils': [
            "APPELER LE 15 IMMÉDIATEMENT",
            "Ne pas donner à manger ou boire",
            "Notez l'heure de début des symptômes"
        ],
        'couleur': 'red'
    },

    'Pancreatite': {
        'niveau': 'urgence absolue',
        'conseils': [
            "Hospitalisation en urgence",
            "Jeûne strict nécessaire",
            "Traitement antalgique en urgence"
        ],
        'couleur': 'red'
    },
    'Pneumonie': {
        'niveau': 'urgence',
        'conseils': [
            "Antibiothérapie urgente",
            "Hospitalisation si détresse respiratoire",
            "Surveillance constante de la saturation"
        ],
        'couleur': 'red'
    },
    'Paludisme': {
        'niveau': 'urgence',
        'conseils': [
            "Traitement antipaludéen en urgence",
            "Hospitalisation systématique",
            "Bilan parasitologique sanguin"
        ],
        'couleur': 'red'
    },

    # Maladies graves (consultation rapide nécessaire)
    'Asthme': {
        'niveau': 'grave',
        'conseils': [
            "Bronchodilatateur immédiat",
            "Consultation pneumologue sous 48h",
            "Éviter les facteurs déclenchants"
        ],
        'couleur': 'yellow'
    },
    'Diabete': {
        'niveau': 'grave',
        'conseils': [
            "Contrôle glycémique immédiat",
            "Adaptation du traitement insulinique",
            "Consultation endocrinologue sous 7 jours"
        ],
        'couleur': 'yellow'
    },
    'Hypertension': {
        'niveau': 'grave',
        'conseils': [
            "Mesure tensionnelle répétée",
            "Réduction stricte du sel",
            "Consultation cardiologique sous 72h"
        ],
        'couleur': 'yellow'
    },
    'Tuberculose': {
        'niveau': 'grave',
        'conseils': [
            "Isolement respiratoire immédiat",
            "Traitement antibiotique spécifique",
            "Déclaration obligatoire aux autorités sanitaires"
        ],
        'couleur': 'yellow'
    },
    'Hepatite_B': {
        'niveau': 'grave',
        'conseils': [
            "Bilan hépatique complet",
            "Vaccination de l'entourage",
            "Consultation hépatologue sous 7 jours"
        ],
        'couleur': 'yellow'
    },
    'Maladie_coronarienne': {
        'niveau': 'grave',
        'conseils': [
            "ECG et bilan cardiaque urgent",
            "Arrêt immédiat du tabac si fumeur",
            "Consultation cardiologique sous 48h"
        ],
        'couleur': 'yellow'
    },

    # Maladies chroniques (prise en charge spécialisée)
    'Cancer_foie': {
        'niveau': 'chronique grave',
        'conseils': [
            "Consultation oncologique urgente",
            "Bilan hépatique complet",
            "Prise en charge multidisciplinaire"
        ],
        'couleur': 'magenta'
    },
    'Cancer_poumon': {
        'niveau': 'chronique grave',
        'conseils': [
            "Scanner thoracique urgent",
            "Bilan d'extension oncologique",
            "Consultation pneumo-oncologique"
        ],
        'couleur': 'magenta'
    },
    'Cancer_rein': {
        'niveau': 'chronique grave',
        'conseils': [
            "Échographie rénale urgente",
            "Bilan urologique complet",
            "Consultation oncologique"
        ],
        'couleur': 'magenta'
    },
    'Maladie_Parkinson': {
        'niveau': 'chronique',
        'conseils': [
            "Adaptation du traitement neurologique",
            "Kinésithérapie spécialisée",
            "Consultation neurologique mensuelle"
        ],
        'couleur': 'magenta'
    },
    'Polyarthrite_rhumatoide': {
        'niveau': 'chronique',
        'conseils': [
            "Traitement de fond rhumatologique",
            "Rééducation fonctionnelle",
            "Surveillance radiologique"
        ],
        'couleur': 'magenta'
    },
    'Maladie_Crohn': {
        'niveau': 'chronique',
        'conseils': [
            "Régime alimentaire adapté",
            "Traitement immunosuppresseur",
            "Coloscopies régulières"
        ],
        'couleur': 'magenta'
    },
    'Osteoporose': {
        'niveau': 'chronique',
        'conseils': [
            "Supplémentation en calcium/vitamine D",
            "Traitement anti-ostéoporotique",
            "Prévention des chutes"
        ],
        'couleur': 'magenta'
    },

    # Maladies modérées (consultation sous 15 jours)
    'Grippe': {
        'niveau': 'modéré',
        'conseils': [
            "Repos 5-7 jours",
            "Hydratation abondante",
            "Antipyrétiques si fièvre > 38.5°C"
        ],
        'couleur': 'blue'
    },
    'Bronchite': {
        'niveau': 'modéré',
        'conseils': [
            "Arrêt de travail si nécessaire",
            "Antitussifs si toux sèche",
            "Consultation si persistance > 10 jours"
        ],
        'couleur': 'blue'
    },
    'Infection_urinaire': {
        'niveau': 'modéré',
        'conseils': [
            "Antibiothérapie adaptée",
            "Hydratation intensive",
            "ECBU de contrôle"
        ],
        'couleur': 'blue'
    },
    'Colite_ulcereuse': {
        'niveau': 'modéré',
        'conseils': [
            "Régime sans résidus",
            "Traitement anti-inflammatoire",
            "Coloscopie de contrôle"
        ],
        'couleur': 'blue'
    },
    'Hyperthyroidie': {
        'niveau': 'modéré',
        'conseils': [
            "Bilan thyroïdien complet",
            "Traitement freinateur",
            "Consultation endocrinologue"
        ],
        'couleur': 'blue'
    },
    'Migraine': {
        'niveau': 'modéré',
        'conseils': [
            "Triptans si diagnostiqué",
            "Repos dans le calme et l'obscurité",
            "Tenir un agenda des crises"
        ],
        'couleur': 'blue'
    },
    'Depression': {
        'niveau': 'modéré',
        'conseils': [
            "Consultation psychiatrique",
            "Thérapie cognitivo-comportementale",
            "Suivi régulier nécessaire"
        ],
        'couleur': 'blue'
    },

    # Maladies légères (autogestion possible)
    'Rhume': {
        'niveau': 'léger',
        'conseils': [
            "Lavages nasaux réguliers",
            "Repos 2-3 jours",
            "Pas d'antibiotiques nécessaires"
        ],
        'couleur': 'green'
    },
    'Eczema': {
        'niveau': 'léger',
        'conseils': [
            "Emollients quotidiens",
            "Corticoïdes locaux si poussée",
            "Éviction des allergènes"
        ],
        'couleur': 'green'
    },
    'Rhinite_allergique': {
        'niveau': 'léger',
        'conseils': [
            "Antihistaminiques oraux",
            "Lavages nasaux au sérum physiologique",
            "Éviction des allergènes identifiés"
        ],
        'couleur': 'green'
    },
    'Psoriasis': {
        'niveau': 'léger',
        'conseils': [
            "Hydratation cutanée intensive",
            "Photothérapie si étendu",
            "Consultation dermatologique si persistant"
        ],
        'couleur': 'green'
    },
    'Arthrose': {
        'niveau': 'léger',
        'conseils': [
            "Activité physique adaptée",
            "Antalgiques en cas de poussée douloureuse",
            "Poids santé si surcharge pondérale"
        ],
        'couleur': 'green'
    }
}


def poser_question(symptome):
    """Fonction pour poser une question binaire (Oui/Non)"""
    while True:
        reponse = input(f"Avez-vous {symptome.replace('_', ' ')}? (Oui/Non): ").strip().lower()
        if reponse in ['oui', 'non', 'o', 'n']:
            return 1 if reponse in ['oui', 'o'] else 0
        print("Réponse invalide. Veuillez répondre par 'Oui' ou 'Non'.")


def collecter_reponses():
    """Collecte les réponses de l'utilisateur"""
    print("=== Bienvenue dans le système de prédiction de maladies basé sur les symptômes ===\n")
    print("=== Veuillez répondre aux questions suivantes (Oui/Non) concernant vos symptômes ===\n")
    reponses = {}

    # Poser des questions pour chaque symptôme
    for symptome in FEATURES[:-3]:  # Exclure les 3 derniers (caractéristiques non-symptômes)
        if symptome not in ['Tension_arterielle', 'Niveau_Cholesterol', 'Age', 'Genre']:
            reponses[symptome] = poser_question(symptome)

    # Ajouter les caractéristiques spécifiques
    reponses['Age'] = int(input("Quel est votre âge? "))
    reponses['Genre'] = input("Genre (Homme/Femme): ").strip().lower()
    reponses['Tension_arterielle'] = input("Tension artérielle (Bas/Normale/Elevee): ").strip().lower()
    reponses['Niveau_Cholesterol'] = input("Niveau de cholestérol (Bas/Normal/Eleve): ").strip().lower()

    return reponses

def init_scalers():
    """Initialise et sauvegarde-les scalers s'ils n'existent pas"""
    if not os.path.exists('age_scaler.pkl'):
        scaler = MinMaxScaler()
        scaler.fit([[0], [120]])  # Plage d'âge 0-120 ans
        joblib.dump(scaler, 'age_scaler.pkl')
# Appeler cette fonction au démarrage
#init_scalers()


def encoder_reponses_utilisateur(reponses_utilisateur):
    """Encode les réponses de l'utilisateur pour le modèle"""
    # Création du DataFrame
    df_temp = pd.DataFrame([reponses_utilisateur])

    # Encodage des variables catégorielles
    df_temp['Genre'] = df_temp['Genre'].map({'homme': 1, 'femme': 0}).astype('int8')

    df_temp['Tension_arterielle'] = df_temp['Tension_arterielle'].map(
        {'elevee': 2, 'normale': 1, 'bas': 0}).astype('int8')

    df_temp['Niveau_Cholesterol'] = df_temp['Niveau_Cholesterol'].map(
        {'eleve': 2, 'normal': 1, 'bas': 0}).astype('int8')

    # Normalisation de l'âge
    scaler = joblib.load('age_scaler.pkl')  # Charger le scaler sauvegardé
    df_temp['Age_normalized'] = scaler.transform(df_temp[['Age']])


def generer_recommandation(maladie, probabilite):
    """Génère des recommandations basées sur la maladie et la probabilité"""
    if maladie == "Maladie non identifiée":
        return GRAVITE_MALADIES.get(maladie, {
            'niveau': 'inconnu',
            'conseils': ["Consultation médicale recommandée"],
            'couleur': 'white'
        })

    if maladie == "Bonne santé":
        return {
            'niveau': 'nulle',
            'conseils': ["Maintenez vos bonnes habitudes", "Check-up annuel recommandé"],
            'couleur': 'green',
            'message': "Aucune pathologie détectée",
            'urgence': "aucune"
        }

    infos = GRAVITE_MALADIES.get(maladie, {
        'niveau': 'modérée',
        'conseils': ["Consultation médicale recommandée"],
        'couleur': 'blue'
    })

    if probabilite < 0.6:
        return {
            'niveau': 'incertaine',
            'conseils': ["Diagnostic incertain - Consultation nécessaire"] + infos['conseils'],
            'couleur': 'magenta',
            'message': "Résultat peu certain - Confirmation médicale requise",
            'urgence': "consultation"
        }

    if infos['niveau'] == 'grave':
        urgence = "immédiate"
        message = "🔴 URGENCE: Consultez immédiatement"
    elif infos['niveau'] == 'chronique':
        urgence = "spécialisée"
        message = "🟠 Consultation spécialisée nécessaire"
    else:
        urgence = "programmée"
        message = "🟢 Consultation médicale recommandée"

    return {
        'niveau': infos['niveau'],
        'conseils': infos['conseils'],
        'couleur': infos['couleur'],
        'message': message,
        'urgence': urgence
    }

def diagnostic_complet( input_data):
    """Effectue un diagmodel,nostic complet"""
    try:
        # Définir les features attendues en entrée du modèle
        features_attendues = [f for f in FEATURES if f != 'Age'] + ['Age_normalized']

        # Validation des dimensions
        if input_data.shape[1] != len(features_attendues):
            raise ValueError(f"Dimensions incorrectes. Reçu: {input_data.shape[1]}, Attendu: {len(features_attendues)}")

        # Création du dictionnaire de données
        data_dict = dict(zip(features_attendues, input_data[0]))

        # Vérification du patient sain
        symptomes = [f for f in FEATURES if f not in ['Tension_arterielle', 'Niveau_Cholesterol', 'Age', 'Genre']]
        tous_negatifs = all(data_dict.get(sym, 0) == 0 for sym in symptomes)

        tension_ok = data_dict.get("Tension_arterielle") == 1
        cholesterol_ok = data_dict.get("Niveau_Cholesterol") == 1

        if tous_negatifs and tension_ok and cholesterol_ok:
            reco = generer_recommandation("Bonne santé", 1.0)
            return {
                'maladie': 'Bonne santé',
                'probabilite': 1.0,
                'special_case': 'patient_sain',
                'top_5': [("Bonne santé", 1.0)],
                **reco
            }

        # Prédiction avec le modèle
        probas = model.predict_proba(input_data)[0]
        top_5_idx = np.argsort(probas)[-5:][::-1]
        principal_idx = model.predict(input_data)[0]
        maladie_predite = MALADIES[principal_idx]
        probabilite = float(probas[principal_idx])

        # Cas de doute
        if probabilite < 0.6:
            reco = generer_recommandation("Maladie non identifiée", probabilite)
            return {
                'maladie': 'Maladie non identifiée',
                'probabilite': probabilite,
                'special_case': 'maladie_inconnue',
                'top_5': [(MALADIES[i], float(probas[i])) for i in top_5_idx],
                **reco
            }

        # Cas normal
        reco = generer_recommandation(maladie_predite, probabilite)
        return {
            'maladie': maladie_predite,
            'probabilite': probabilite,
            'top_5': [(MALADIES[i], float(probas[i])) for i in top_5_idx],
            **reco
        }

    except Exception as e:
        logging.error(f"Erreur dans diagnostic_complet: {str(e)}", exc_info=True)
        return {
            'maladie': 'Erreur système',
            'probabilite': 0.0,
            'niveau': 'inconnue',
            'conseils': ["Contactez l'administrateur"],
            'couleur': 'white',
            'message': "Une erreur technique est survenue",
            'urgence': "indéterminée",
            'erreur': str(e)
        }
def afficher_diagnostic(results):
    """Affiche les résultats du diagnostic"""
    if 'erreur' in results:
        print(colored("\n⚠️ ERREUR DE DIAGNOSTIC ⚠️", 'red', attrs=['bold']))
        print(colored(f"Message d'erreur: {results['erreur']}", 'red'))
        return

    couleur = results.get('couleur', 'white')

    print(colored("\n🔍 RÉSULTATS DU DIAGNOSTIC 🔍", 'cyan', attrs=['bold']))

    # Cas spécial : patient sain
    if results.get('maladie') == 'Bonne santé':
        print(colored("\n🎉 FÉLICITATIONS!", 'green', attrs=['bold']))
        print(colored("Vous êtes en bonne santé selon notre analyse", 'green'))
        print(colored("\n💡 CONSEILS MÉDICAUX:", 'green'))
        for i, conseil_item in enumerate(results['conseils'], 1):  # Utilisez un nom différent (conseil_item)
            print(f"Conseil {i}: {conseil_item}")
            print(colored(f"{i}. {conseil_item}", 'green'))
        return

    # Cas spécial : maladie inconnue
    if results.get('maladie') == 'Maladie non identifiée':
        print(colored("\nℹ️ ATTENTION:", 'yellow', attrs=['bold']))
        print(colored("Notre système n'a pas pu identifier clairement votre problème", 'yellow'))
    else:
        # Cas normal
        print(colored(f"\nDiagnostic principal: {results['maladie']}", couleur, attrs=['bold']))

    print(colored(f"Confiance: {results['probabilite'] * 100:.1f}%", couleur))

    # Affichage du top 5 seulement si pertinent
    if 'top_5' in results and len(results['top_5']) > 1:
        print(colored("\nTop 5 des diagnostics possibles:", 'cyan'))
        for maladie, proba in results['top_5']:
            print(f"- {maladie}: {proba * 100:.1f}%")

    print(colored("\n⚕️ RECOMMANDATION:", couleur, attrs=['bold']))
    print(colored(results['message'], couleur, attrs=['bold']))

    print(colored("\n💡 CONSEILS MÉDICAUX:", couleur))
    for i, conseils in enumerate(results['conseils'], 1):  # Nom unifié
        print(colored(f"{i}. {conseils}", couleur))  # Utilisation cohérente

    if results['urgence'] in ["immédiate", "élevée"]:
        print(colored("\n🚨 ACTION REQUISE:", 'red', attrs=['bold', 'blink']))
        if results['urgence'] == "immédiate":
            print(colored("COMPOSEZ LE 15 IMMÉDIATEMENT", 'red', attrs=['bold', 'blink']))
        else:
            print(colored("Rendez-vous aux urgences dans les plus brefs délais", 'yellow', attrs=['bold']))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about-us')
def apropos():
    return render_template("about-us.html")


@app.route('/blog')
def conseil():
    return render_template("blog.html")


@app.route('/loginadm')
def login():
    return render_template("loginadm.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/test', methods=['GET', 'POST'])
def testons():
    if request.method == 'POST':
        try:
            # Récupération des informations générales
            age = request.form.get('Age')
            genre = request.form.get('Genre')
            tension = request.form.get('Tension_arterielle')
            cholesterol = request.form.get('Niveau_Cholesterol')

            # Création du dictionnaire de symptômes
            symptomes = {
                'Age': int(age) if age else 0,
                'Genre': genre.lower() if genre else 'homme',
                'Tension_arterielle': tension.lower() if tension else 'normale',
                'Niveau_Cholesterol': cholesterol.lower() if cholesterol else 'normal'
            }

            # Récupération de tous les symptômes booléens
            for symptom in FEATURES:
                if symptom not in ['Age', 'Genre', 'Tension_arterielle', 'Niveau_Cholesterol']:
                    symptomes[symptom] = 1 if request.form.get(symptom.lower()) else 0

            # Préparation des données pour le modèle
            df = pd.DataFrame([symptomes])

            # Encodage des variables catégorielles
            df['Genre'] = df['Genre'].map({'homme': 1, 'femme': 0}).astype('int8')
            df['Tension_arterielle'] = df['Tension_arterielle'].map(
                {'elevee': 2, 'normale': 1, 'bas': 0}).astype('int8')
            df['Niveau_Cholesterol'] = df['Niveau_Cholesterol'].map(
                {'eleve': 2, 'normal': 1, 'bas': 0}).astype('int8')

            # Normalisation de l'âge
            scaler = joblib.load('age_scaler.pkl')
            df['Age_normalized'] = scaler.transform(df[['Age']])
            df.drop('Age', axis=1, inplace=True)

            # Réorganisation des colonnes
            features_order = [f for f in FEATURES if f != 'Age'] + ['Age_normalized']
            df = df[features_order]
            print(df)

            # Faire la prédiction
            resultat = diagnostic_complet(df.values)
            print(resultat)


            return render_template('test.html',
                                   features=FEATURES,
                                   resultat=resultat,
                                   symptomes=symptomes)

        except Exception as e:
            error_result = {
                'maladie': 'Erreur système',
                'probabilite': 0.0,
                'niveau': 'inconnue',
                'conseils': ["Une erreur est survenue lors de l'analyse"],
                'couleur': 'white',
                'message': "Erreur technique - Veuillez réessayer",
                'urgence': "indéterminée",
                'erreur': str(e)
            }
            print(e)
            return render_template('test.html',
                                   features=FEATURES,
                                   resultat=error_result)


    return render_template('test.html', features=FEATURES)

def analyse_symptomes(symptomes):
    """Logique d'analyse des symptômes"""
    # Symptômes urgents
    if symptomes.get('Difficulte_respiratoire', 0) == 1 or \
            symptomes.get('Douleur_thoracique', 0) == 1 or \
            symptomes.get('Paralysie_faciale', 0) == 1:
        return "URGENCE MÉDICALE - Consultez immédiatement !"

    # Risque élevé pour les seniors
    if int(symptomes.get('Age', 0)) > 60 and \
            (symptomes.get('Tension_arterielle') == 'eleve' or
             symptomes.get('Niveau_Cholesterol') == 'eleve'):
        return "Risque cardiovasculaire élevé - Consultation rapide recommandée"

    # Symptômes grippaux
    if symptomes.get('Fievre', 0) == 1 and \
            (symptomes.get('Toux', 0) == 1 or symptomes.get('Fatigue', 0) == 1):
        return "Symptômes grippaux - Repos et surveillance"

    # Aucun symptôme grave détecté
    if sum(int(v) for k, v in symptomes.items() if
           k not in ['Age', 'Genre', 'Tension_arterielle', 'Niveau_Cholesterol']) == 0:
        return "Aucun symptôme préoccupant détecté"

    return "Symptômes détectés - Surveillance recommandée"



@app.route('/service')
def board():
    return render_template("services.html")


if __name__ == "__main__":
    app.run(debug=True)


# Configuration du logging
logging.basicConfig(filename='app.py', level=logging.ERROR)

def charger_fichier(chemin):
    try:
        with open(chemin, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Fichier {chemin} non trouvé")
        return None
    except Exception as e:
        logging.error(f"Erreur inattendue : {e}")
        raise  # Relance l'exception

