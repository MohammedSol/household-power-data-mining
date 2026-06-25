# ⚡ Household Power — Data Mining & Time Series Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

> Analyse de séries temporelles, prédiction de la demande énergétique et détection automatique d'anomalies sur 4 années de relevés électriques haute fréquence d'un foyer résidentiel.

---

## 📋 Contexte et Problématique

Avec l'émergence des **Smart Grids** et la transition énergétique, la capacité à anticiper la demande électrique d'un foyer est devenue un enjeu stratégique pour les opérateurs de réseau. Une prévision fiable permet d'**équilibrer la charge en temps réel**, de réduire le gaspillage lié aux surplus de production, et d'identifier proactivement les comportements anormaux (pannes, surconsommations, fraudes).

Ce projet s'attaque à trois questions fondamentales à partir de données historiques réelles :

1. **Analyse des habitudes** — Quels sont les schémas de consommation caractéristiques de ce foyer (profil intra-journalier, saisonnalité) ?
2. **Prédiction de la demande** — Est-il possible de prédire fidèlement la puissance active consommée à partir de caractéristiques calendaires et physiques ?
3. **Détection d'anomalies** — Comment identifier automatiquement les heures de consommation statistiquement atypiques ?

---

## 🗄️ Jeu de Données

| Propriété | Valeur |
|-----------|--------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) |
| **Auteur** | Georges Hébrail & Alice Bérard |
| **Localisation** | Foyer résidentiel, Sceaux, France |
| **Période couverte** | Décembre 2006 – Novembre 2010 |
| **Granularité brute** | 1 relevé par minute |
| **Volume brut** | ~2 080 000 lignes × 9 colonnes |
| **Volume après agrégation** | ~35 000 lignes × 9 colonnes (agrégation horaire) |
| **Format** | Texte délimité par `;` (`.txt`) |
| **Valeurs manquantes** | Encodées par `?` → imputées par interpolation temporelle |

### Variables principales

| Variable | Description | Unité |
|----------|-------------|-------|
| `Global_active_power` | Puissance active globale (**variable cible**) | kW |
| `Global_intensity` | Intensité du courant global | A |
| `Voltage` | Tension du réseau | V |
| `Sub_metering_1` | Cuisine (four, lave-vaisselle) | Wh |
| `Sub_metering_2` | Buanderie (machine à laver, réfrigérateur) | Wh |
| `Sub_metering_3` | Chauffage & climatisation | Wh |

---

## 🏗️ Architecture du Projet

Le pipeline suit une méthodologie rigoureuse en 6 étapes séquentielles, conçue pour respecter les contraintes mémoire d'un dataset de grande taille.

```
Données brutes (~2M lignes)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  1. CHARGEMENT & GESTION DES NaN                │
│     pd.read_csv — na_values=['?', ' ', '']      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  2. NETTOYAGE & AGRÉGATION HORAIRE              │
│     Index datetime · interpolate(method='time') │
│     resample('h').mean() → ~35 000 lignes       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  3. ANALYSE EXPLORATOIRE (EDA)                  │
│     Série temporelle · Profil intra-journalier  │
│     Moyenne mobile 7 jours (Rolling Mean)       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  4. INGÉNIERIE DES CARACTÉRISTIQUES             │
│     hour · dayofweek · month · is_weekend       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  5. SPLIT TEMPOREL STRICT                       │
│     Train : 2006–2009 │ Test : 2010             │
│     (aucun mélange temporel — zéro data leakage)│
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  6. MODÉLISATION & ÉVALUATION                   │
│     Régression Linéaire (baseline)              │
│     Random Forest Regressor (modèle principal)  │
│     Isolation Forest (détection d'anomalies)    │
└─────────────────────────────────────────────────┘
```

---

## 📊 Résultats Clés

### Prédiction de la consommation (Apprentissage supervisé)

| Modèle | MAE (kW) | RMSE (kW) | R² |
|--------|----------|-----------|----|
| Régression Linéaire | — | — | ~0.98 |
| **Random Forest Regressor** | **~0.002** | **~0.003** | **~0.999** |

> **⚠️ Note d'interprétation — Score R² et loi physique**
>
> Le score R² quasi-parfait (> 0.99) s'explique par la présence dans les features des variables **`Voltage`** et **`Global_intensity`**, qui entretiennent une **relation physique directe** avec la puissance active via la loi d'Ohm : `P = U × I`. Dans un contexte de prédiction *ex ante* (prédire demain sans connaître l'intensité de demain), ces variables constitueraient une **fuite d'information (data leakage)**. Ce résultat illustre l'importance d'une sélection de features cohérente avec le scénario d'usage réel.

### Détection d'anomalies (Apprentissage non supervisé)

- **Algorithme :** Isolation Forest
- **Paramètre de contamination :** 1% (hypothèse : 1 heure sur 100 est anormale)
- **Résultat :** ~350 heures anormales détectées sur 4 ans, correspondant à deux profils distincts :
  - **Pics extrêmes** — utilisation simultanée d'appareils énergivores
  - **Chutes anormales** — absences prolongées du foyer ou défaillances capteur

---

## 🚀 Installation et Exécution

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes

**1. Cloner le dépôt**

```bash
git clone https://github.com/MohammedSol/household-power-data-mining.git
cd household-power-data-mining
```

**2. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**3. Télécharger le dataset**

Le fichier `household_power_consumption.txt` dépasse la limite de taille GitHub et n'est pas inclus dans le dépôt. Téléchargez-le depuis l'UCI Repository et placez-le à la racine du projet :

```
https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
```

**4. Lancer le notebook**

```bash
jupyter notebook notebook.ipynb
```

> **Conseil :** Exécutez les cellules dans l'ordre séquentiel. Les sections sont numérotées de 01 à 10 pour guider l'exécution.

---

## 🗂️ Structure du Dépôt

```
household-power-data-mining/
│
├── notebook.ipynb                  # Notebook principal (10 sections)
│
├── requirements.txt                # Dépendances Python
├── .gitignore                      # Fichiers exclus du dépôt
├── README.md                       # Ce fichier
│
└── household_power_consumption.txt # ⚠️ NON INCLUS (trop volumineux, ~127 MB)
                                    #    À télécharger depuis l'UCI Repository
```

### Sections du Notebook

| # | Section | Contenu |
|---|---------|---------|
| 01 | `import_libraries` | Import des bibliothèques (pandas, numpy, sklearn, matplotlib, seaborn) |
| 02 | `load_dataset` | Chargement avec gestion des NaN |
| 03 | `problem_understanding` | Reformulation de la problématique |
| 04 | `dataset_description` | Description des variables et de la cible |
| 05 | `exploratory_data_analysis` | 3 figures : série temporelle, profil horaire, rolling mean |
| 06 | `data_quality_and_cleaning` | Nettoyage, interpolation, agrégation horaire |
| 07 | `preprocessing` | Feature engineering + split temporel strict |
| 08 | `modeling_or_clustering` | Régression Linéaire, Random Forest, Isolation Forest |
| 09 | `evaluation` | Métriques MAE/RMSE/R² + figures de prédiction et d'anomalies |
| 10 | `interpretation_and_limits` | Analyse critique, limites, pistes d'amélioration |

---

## ⚙️ Stack Technique

| Outil | Usage |
|-------|-------|
| **Python 3.10+** | Langage principal |
| **Pandas** | Manipulation et agrégation des données |
| **NumPy** | Calculs numériques |
| **Scikit-Learn** | Modèles supervisés et non supervisés |
| **Matplotlib / Seaborn** | Visualisations scientifiques |
| **Jupyter Notebook** | Environnement de développement interactif |

---

## 👤 Auteur

**Mohammed Solilahy**  
Étudiant en Data Science & Big Data — EMSI  
Projet académique — Module Data Mining, Semestre 1

---

*Dataset source : Hébrail, G. & Bérard, A. (2012). Individual household electric power consumption. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54*
