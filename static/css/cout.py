import joblib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from termcolor import colored
import os
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Au début de votre fichier, remplacez le chargement du modèle par :
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgboost.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modèle chargé avec succès depuis {MODEL_PATH}")
except FileNotFoundError:
    print(colored(f"ERREUR: Fichier modèle introuvable à l'emplacement: {MODEL_PATH}", 'red'))
    model = None

# Liste des maladies
MALADIES = np.array(['Arthrose', 'Asthme', 'AVC', 'Bronchite', 'Cancer_foie',
                     'Cancer_poumon', 'Cancer_rein', 'Colite_ulcereuse', 'Depression',
                     'Diabete', 'Eczema', 'Grippe', 'Hepatite_B', 'Hypertension',
                     'Hyperthyroidie', 'Infection_urinaire', 'Maladie_coronarienne',
                     'Maladie_Crohn', 'Maladie_Parkinson', 'Migraine', 'Osteoporose',
                     'Paludisme', 'Pancreatite', 'Pneumonie', 'Polyarthrite_rhumatoide',
                     'Psoriasis', 'Rhinite_allergique', 'Rhume', 'Tuberculose'])

# Liste des symptômes et caractéristiques
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

# Dictionnaire de gravité des maladies (identique à votre nouveau code)
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




def init_scalers():
    """Initialise et sauvegarde-les scalers s'ils n'existent pas"""
    if not os.path.exists('age_scaler.pkl'):
        scaler = MinMaxScaler()
        scaler.fit([[0], [120]])  # Plage d'âge 0-120 ans
        joblib.dump(scaler, 'age_scaler.pkl')


# Appeler cette fonction au démarrage
init_scalers()


def encoder_reponses_utilisateur(reponses_utilisateur):
    """Encode les réponses de l'utilisateur pour le modèle"""
    # Création d'un DataFrame avec toutes les features initialisées à 0
    df_temp = pd.DataFrame(0, index=[0], columns=FEATURES)

    # Remplissage des valeurs reçues
    for feature in FEATURES:
        if feature in reponses_utilisateur:
            df_temp[feature] = reponses_utilisateur[feature]

    # Encodage des variables catégorielles
    genre = str(reponses_utilisateur.get('Genre', 'homme')).lower()
    df_temp['Genre'] = 1 if genre in ['homme', 'male', 'm'] else 0

    tension = str(reponses_utilisateur.get('Tension_arterielle', 'normale')).lower()
    if 'eleve' in tension:
        df_temp['Tension_arterielle'] = 2
    elif 'norm' in tension:
        df_temp['Tension_arterielle'] = 1
    else:
        df_temp['Tension_arterielle'] = 0

    cholesterol = str(reponses_utilisateur.get('Niveau_Cholesterol', 'normal')).lower()
    if 'eleve' in cholesterol:
        df_temp['Niveau_Cholesterol'] = 2
    elif 'norm' in cholesterol:
        df_temp['Niveau_Cholesterol'] = 1
    else:
        df_temp['Niveau_Cholesterol'] = 0

    # Normalisation de l'âge
    scaler = joblib.load('age_scaler.pkl')
    df_temp['Age'] = scaler.transform([[reponses_utilisateur.get('Age', 30)]])[0][0]

    # Vérification finale
    assert df_temp.shape[1] == len(
        FEATURES), f"Nombre de features incorrect: {df_temp.shape[1]} au lieu de {len(FEATURES)}"

    return df_temp.astype('float32')

def generer_recommandation(maladie, probabilite):
    """Génère des recommandations basées sur la maladie et la probabilité"""
    if maladie == "Maladie non identifiée":
        return {
            'niveau': 'inconnu',
            'conseils': ["Consultation médicale recommandée"],
            'couleur': 'white',
            'message': "Diagnostic incertain",
            'urgence': "consultation"
        }

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

    if infos['niveau'] == 'urgence absolue':
        urgence = "immédiate"
        message = "🔴 URGENCE VITALE: Appelez le 15 immédiatement"
    elif infos['niveau'] in ['grave', 'urgence']:
        urgence = "élevée"
        message = "🟠 URGENCE: Consultez aux urgences dans l'heure"
    elif infos['niveau'] == 'chronique grave':
        urgence = "spécialisée"
        message = "🟡 CONSULTATION SPÉCIALISÉE: Prise en charge urgente nécessaire"
    elif infos['niveau'] == 'modéré':
        urgence = "programmée"
        message = "🔵 CONSULTATION: Prenez RDV sous 48h"
    else:
        urgence = "à évaluer"
        message = "ℹ️ CONSULTATION RECOMMANDÉE: Évaluation médicale nécessaire"

    return {
        'niveau': infos['niveau'],
        'conseils': infos['conseils'],
        'couleur': infos['couleur'],
        'message': message,
        'urgence': urgence
    }


def diagnostic_complet(ml_model, input_data):
    """Effectue un diagnostic complet"""
    try:
        data_dict = {feat: input_data[0][i] for i, feat in enumerate(FEATURES[:-1])}
        data_dict['Age'] = input_data[0][-1]  # L'âge est normalement le dernier

        # Vérification patient sain
        symptomes = [f for f in FEATURES if f not in ['Tension_arterielle', 'Niveau_Cholesterol', 'Age', 'Genre']]
        tous_negatifs = all(data_dict.get(sym, 0) == 0 for sym in symptomes)

        tension_ok = data_dict.get("Tension_arterielle", 1) == 1
        cholesterol_ok = data_dict.get("Niveau_Cholesterol", 1) == 1

        if tous_negatifs and tension_ok and cholesterol_ok:
            reco = generer_recommandation("Bonne santé", 1.0)
            return {
                'maladie': 'Bonne santé',
                'probabilite': 1.0,
                'special_case': 'patient_sain',
                'top_5': [("Bonne santé", 1.0)],
                **reco
            }

        # Prédiction
        probas = ml_model.predict_proba(input_data)[0]
        top_5_idx = np.argsort(probas)[-5:][::-1]
        principal_idx = ml_model.predict(input_data)[0]
        maladie_predite = MALADIES[principal_idx]
        probabilite = float(probas[principal_idx])

        if probabilite < 0.6:
            reco = generer_recommandation("Maladie non identifiée", probabilite)
            return {
                'maladie': 'Maladie non identifiée',
                'probabilite': probabilite,
                'special_case': 'maladie_inconnue',
                'top_5': [(MALADIES[i], float(probas[i])) for i in top_5_idx],
                **reco
            }

        reco = generer_recommandation(maladie_predite, probabilite)
        return {
            'maladie': maladie_predite,
            'probabilite': probabilite,
            'top_5': [(MALADIES[i], float(probas[i])) for i in top_5_idx],
            **reco
        }

    except Exception as e:
        print(colored(f"Erreur système: {str(e)}", 'red'))
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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about-us')
def apropos():
    return render_template("about-us.html")


@app.route('/blog')
def conseil():
    return render_template("blog.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/test', methods=["POST", "GET"])
def testons():
    if request.method == "POST":
        try:
            print("Récupération des données du formulaire...")
            reponses = {k: int(v) if v.isdigit() else v
                        for k, v in request.form.items()
                        if k in FEATURES}

            print("Préparation des données...")
            donnees_utilisateur = encoder_reponses_utilisateur(reponses)
            print(f"Données préparées - Shape: {donnees_utilisateur.shape}")
            print(f"Colonnes: {list(donnees_utilisateur.columns)}")

            if model is None:
                raise ValueError("Modèle non chargé")

            print("Lancement du diagnostic...")
            resultat = diagnostic_complet(model, donnees_utilisateur.values)
            print("Résultat obtenu:", resultat)

            return render_template('test.html', resultat=resultat)

        except Exception as e:
            print(f"ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('test.html',
                                 erreur=f"Erreur technique: {str(e)}")

    return render_template("test.html")
@app.route('/service')
def board():
    return render_template("services.html")


if __name__ == "__main__":
    app.run(debug=True)