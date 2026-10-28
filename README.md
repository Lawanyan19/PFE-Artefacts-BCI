# Impact des artefacts EEG sur les performances d'une BCI d'imagerie motrice

Les interfaces cerveau-machine (BCI) basées sur l'imagerie motrice décodent, à partir du signal EEG, le mouvement de main qu'une personne imagine. L'EEG est très sensible aux artefacts (clignements, mouvements oculaires, activité musculaire). Ce projet pose une question simple : **la suppression des artefacts oculaires par Analyse en Composantes Indépendantes (ICA) améliore-t-elle la classification main gauche / main droite ?**

La question est étudiée dans deux situations :

- **Offline**, sur un jeu de données public, avec une comparaison statistique avant et après ICA ;
- **Online**, en temps réel, avec un casque EEG grand public (Emotiv EPOC+).

## Données

Jeu de données d'imagerie motrice [Cho2017](https://moabb.neurotechx.com/), chargé avec MOABB (téléchargement automatique, aucune donnée n'est stockée dans ce dépôt) :

| | |
|---|---|
| Sujets | 52 (20 utilisés dans cette étude) |
| Canaux | 64 EEG |
| Fréquence d'échantillonnage | 512 Hz |
| Classes | main gauche / main droite |
| Essais | 100 par classe et par sujet |

## Pipeline

### Offline

1. **Prétraitement** (`Preprocessing/Preprocess.py`, MNE)
   - Suppression des canaux EMG et Stim, ainsi que des électrodes les plus proches de la mâchoire (P9, P10, TP9, TP10, FT9, FT10), dominées par les artefacts musculaires
   - Filtre passe-haut à 1 Hz pour supprimer les dérives lentes
   - Inspection visuelle et interpolation spatiale des mauvais canaux
   - Découpage en epochs de −2 s à +2 s autour de la consigne, baseline de −2 s à −0,5 s
   - Rejet visuel des epochs bruitées
   - FastICA (99 % de la variance), avec inspection interactive des topographies et des décours temporels des composantes pour supprimer les clignements et les mouvements oculaires
   - Sauvegarde des epochs **avant et après ICA**, pour que les deux versions passent par le même pipeline de classification

2. **Construction des données** (`Offline/pretrain_gauche_droite.py`)
   - Filtre passe-bande 8–30 Hz (rythmes mu et bêta)
   - Fenêtre temporelle de 0,5 à 2,0 s après la consigne
   - `X` : essais × 58 canaux × échantillons, `y` : gauche = 0, droite = 1

3. **Classification** (`Offline/trainLDA_gauche_droite.py`, `Offline/trainSVM_gauche_droite.py`, scikit-learn)
   - Common Spatial Patterns (4 composantes, covariance régularisée par shrinkage, log-variance)
   - **CSP + LDA** avec validation croisée stratifiée en 5 plis
   - **CSP + SVM** (noyau RBF) avec validation croisée imbriquée : 5 plis externes, 3 plis internes avec `GridSearchCV` sur C ∈ {0,1 ; 1 ; 10 ; 100} et gamma ∈ {scale, auto}
   - Métriques : accuracy et AUC-ROC par sujet
   - Test de Wilcoxon signé par paires, avant vs après ICA
   - Export d'un tableau Excel et d'un graphique par sujet

### Online

4. **Adaptation au casque** (`Online/retrain_epoc_channels.py`)
   - Estimation par validation croisée des performances de CSP + LDA et CSP + SVM avec uniquement les 12 canaux de Cho2017 correspondant aux électrodes fonctionnelles de l'EPOC+ (T7 et T8 étaient défectueuses)
   - Calcul d'une matrice ICA de repli sur Cho2017, sauvegardée dans `models/ica_matrix.pkl`, utilisée si la calibration ICA échoue en session online

5. **BCI temps réel** (`Online/online_bci.py`)
   - Emotiv EPOC+ (14 électrodes, 128 Hz), signal transmis par [CyKit](https://github.com/CymatiCorp/CyKit) via un WebSocket local
   - Phase 1 : calibration ICA (30 s de clignements volontaires, sélection automatique des composantes les plus corrélées aux canaux frontaux)
   - Phase 2 : calibration sujet (20 essais) avec rejet automatique des epochs contaminées par l'activité musculaire (puissance au-dessus de 40 Hz), puis entraînement de 4 modèles (LDA et SVM, avec et sans ICA) sur les données du sujet
   - Phase 3 : test en temps réel (20 essais) avec retour immédiat
   - Phase 4 : bilan avec accuracy, AUC et test de permutation à 1000 tirages, exporté en CSV

## Résultats

### Offline (20 sujets)

| Modèle | Accuracy avant ICA | Accuracy après ICA | AUC avant / après | Wilcoxon p |
|---|---|---|---|---|
| CSP + LDA | 66,1 % | 66,1 % | 0,705 / 0,704 | 0,81 |
| CSP + SVM | 65,7 % | 65,1 % | 0,710 / 0,702 | 0,84 |

- **L'ICA n'a aucun effet significatif** sur la classification (p > 0,80 pour les deux modèles).
- **La variabilité inter-sujets est forte** : de 45 % à 94 % d'accuracy environ selon le sujet.
- Avec seulement les 12 canaux disponibles sur l'EPOC+, l'accuracy en validation croisée tombe à environ 57 %.

### Online (5 sessions)

| Session | LDA avant ICA | LDA après ICA | SVM avant ICA | SVM après ICA |
|---|---|---|---|---|
| S1 | 45 % | 55 % | 55 % | 60 % |
| S2 | 65 % | 65 % | **75 %** | 65 % |
| S3 | 55 % | 55 % | 65 % | 50 % |
| S4 | 40 % | 45 % | 40 % | 45 % |
| S5 | 55 % | 40 % | 60 % | 45 % |

Meilleur résultat : CSP + SVM avant ICA, 75 % (AUC 0,76). L'ICA est tantôt bénéfique, tantôt néfaste selon la session.

### Pourquoi l'ICA ne change pas les résultats

1. **Fréquence** : les artefacts oculaires contaminent surtout les basses fréquences (0–8 Hz), que le filtre passe-bande 8–30 Hz élimine déjà avant la classification.
2. **Localisation** : les artefacts oculaires touchent les électrodes frontales (Fp1, Fp2), alors que l'imagerie motrice est décodée au-dessus du cortex sensorimoteur (C3, C4).

**À retenir :** en BCI d'imagerie motrice, la correction des artefacts doit cibler en priorité les **artefacts musculaires**, qui chevauchent directement la bande 8–30 Hz, plutôt que les artefacts oculaires.

## Limites et perspectives

- Absence des électrodes C3/C4 sur l'EPOC+ (remplacées par FC5/FC6), ce qui limite les performances online
- Calibration courte (20 essais) ; 50 à 100 essais seraient plus réalistes
- Automatiser la sélection des composantes avec ICLabel, et explorer une ICA adaptative en temps réel
- Apprentissage par transfert entre sujets pour réduire le temps de calibration

## Structure du dépôt

```
Preprocessing/   prétraitement MNE et ICA
Offline/         construction des données et classification CSP + LDA / SVM
Online/          adaptation au casque et BCI temps réel
```

## Utilisation

Tous les chemins sont relatifs à la racine du dépôt : les scripts doivent être lancés depuis celle-ci.

```bash
pip install -r requirements.txt
mkdir -p data/epochs results

# Offline
python Preprocessing/Preprocess.py         # interactif : mauvais canaux, mauvaises epochs, composantes ICA oculaires
# puis déplacer les fichiers .fif générés dans data/epochs/
python Offline/pretrain_gauche_droite.py   # construit X / y avant et après ICA
python Offline/trainLDA_gauche_droite.py   # CSP + LDA, test de Wilcoxon, tableau Excel + figure
python Offline/trainSVM_gauche_droite.py   # CSP + SVM, test de Wilcoxon, tableau Excel + figure

# Online (nécessite un casque Emotiv EPOC+ et un serveur CyKit sur ws://127.0.0.1:5555)
python Online/retrain_epoc_channels.py
python Online/online_bci.py
```

Le prétraitement repose sur une inspection visuelle dans des fenêtres interactives ; ajuster la plage de sujets dans la boucle de `Preprocess.py` pour les traiter par lots.

## Références

- Cho, H. et al. (2017). EEG datasets for motor imagery brain-computer interface. *GigaScience*.
- Jayaram, V. & Barachant, A. (2018). MOABB: trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering*.
- Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*.
- Minguillon, J. et al. (2017). Trends in EEG-BCI for daily-life: requirements for artifact removal. *Biomedical Signal Processing and Control*.
