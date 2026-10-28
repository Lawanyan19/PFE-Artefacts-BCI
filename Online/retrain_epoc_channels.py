# -*- coding: utf-8 -*-
"""
Ré-entraîne CSP + LDA  ET  CSP + SVM sur les 12 canaux EPOC+ fonctionnels
(T7 et T8 exclus car défectueux sur le casque utilisé).

Produit dans OUTPUT_DIR :
  • ica_matrix.pkl         → matrice de démixage ICA pour usage online
"""

import os, pickle
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.decomposition import FastICA

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH  = "data/epochs"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 14 canaux EPOC+ moins T7/T8 défectueux → 12 canaux actifs
EPOC_ALL      = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
BAD_CHANNELS  = ["T7", "T8"]   # électrodes HS sur le casque
EPOC_CHANNELS = [ch for ch in EPOC_ALL if ch not in BAD_CHANNELS]
print(f"Canaux utilisés ({len(EPOC_CHANNELS)}) : {EPOC_CHANNELS}")

# Indices des canaux frontaux dans EPOC_CHANNELS (pour ICA)
FRONTAL_IDX = [EPOC_CHANNELS.index(ch) for ch in ["AF3","AF4","F7","F8"]
               if ch in EPOC_CHANNELS]

N_SUBJECTS     = 20
TMIN, TMAX     = 0.5, 2.0
L_FREQ, H_FREQ = 8.0, 30.0

# ─────────────────────────────────────────────────────────────────────────────
# 2. CHARGEMENT DES DONNÉES (12 canaux)
# ─────────────────────────────────────────────────────────────────────────────
def load_subject(i: int, condition: str):
    """Charge les epochs d'un sujet, sélectionne les 12 canaux, filtre, crop."""
    f_l = os.path.join(BASE_PATH, f"patient{i:02d}_epochs_{condition}_correction_left.fif")
    f_r = os.path.join(BASE_PATH, f"patient{i:02d}_epochs_{condition}_correction_right.fif")
    ep_l = mne.read_epochs(f_l, preload=True, verbose='ERROR')
    ep_r = mne.read_epochs(f_r, preload=True, verbose='ERROR')
    ep   = mne.concatenate_epochs([ep_l, ep_r])
    ep.pick_channels([ch for ch in EPOC_CHANNELS if ch in ep.ch_names])
    ep.filter(L_FREQ, H_FREQ, method='iir', verbose='ERROR')
    ep.crop(tmin=TMIN, tmax=TMAX)
    X = ep.get_data()
    y = np.concatenate([np.zeros(len(ep_l)), np.ones(len(ep_r))])
    return X, y

print("\nChargement des données Cho2017 (12 canaux)...")
X_b_list, X_a_list, y_list = [], [], []
for i in range(1, N_SUBJECTS + 1):
    try:
        X_b, y_b = load_subject(i, 'before')
        X_a, _   = load_subject(i, 'after')
        X_b_list.append(X_b);  X_a_list.append(X_a);  y_list.append(y_b)
        print(f"  Sujet {i:02d} → {X_b.shape[0]} epochs, {X_b.shape[1]} canaux")
    except FileNotFoundError as e:
        print(f"  [!] Sujet {i:02d} ignoré : {e.filename}")

X_before = np.concatenate(X_b_list)
X_after  = np.concatenate(X_a_list)
y        = np.concatenate(y_list)
print(f"\nTotal : {X_before.shape[0]} epochs | {X_before.shape[1]} canaux | {X_before.shape[2]} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENTRAÎNEMENT LDA
# ─────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_lda(X, y, label):
    clf = make_pipeline(
        CSP(n_components=4, reg='shrinkage', log=True, norm_trace=False),
        LDA()
    )
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"  LDA {label} — CV : {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}%")
    clf.fit(X, y)
    return clf, float(np.mean(scores))

print("\n--- Entraînement LDA ---")
clf_lda_before, score_lda_b = train_lda(X_before, y, "AVANT ICA")
clf_lda_after,  score_lda_a = train_lda(X_after,  y, "APRÈS ICA")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT SVM (GridSearch sur C et gamma)
# ─────────────────────────────────────────────────────────────────────────────
class CSP_SVM_Wrapper:
    """
    Encapsule CSP + SVM dans un objet unique avec interface predict().
    Nécessaire car GridSearchCV ne s'intègre pas directement dans make_pipeline
    quand les paramètres sont préfixés différemment.
    """
    def __init__(self, csp, svm):
        self.csp = csp
        self.svm = svm

    def predict(self, X):
        return self.svm.predict(self.csp.transform(X))

    def predict_proba(self, X):
        return self.svm.predict_proba(self.csp.transform(X))

    def fit(self, X, y):
        self.csp.fit(X, y)
        self.svm.fit(self.csp.transform(X), y)
        return self


def train_svm(X, y, label):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    svm_gs = GridSearchCV(
        Pipeline([('svc', SVC(kernel='rbf', probability=True))]),
        param_grid={'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 'auto']},
        cv=inner_cv, scoring='accuracy', n_jobs=-1, refit=True
    )

    outer_accs = []
    for tr, te in cv.split(X, y):
        csp_fold = CSP(n_components=4, reg='shrinkage', log=True, norm_trace=False)
        csp_fold.fit(X[tr], y[tr])
        X_tr_csp = csp_fold.transform(X[tr])
        X_te_csp = csp_fold.transform(X[te])
        svm_gs.fit(X_tr_csp, y[tr])
        outer_accs.append(svm_gs.score(X_te_csp, y[te]))
    print(f"  SVM {label} — CV : {np.mean(outer_accs)*100:.1f}% ± {np.std(outer_accs)*100:.1f}%")

    csp_final = CSP(n_components=4, reg='shrinkage', log=True, norm_trace=False)
    wrapper   = CSP_SVM_Wrapper(csp_final, svm_gs)
    wrapper.fit(X, y)
    return wrapper, float(np.mean(outer_accs))

print("\n--- Entraînement SVM ---")
clf_svm_before, score_svm_b = train_svm(X_before, y, "AVANT ICA")
clf_svm_after,  score_svm_a = train_svm(X_after,  y, "APRÈS ICA")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MATRICE ICA POUR USAGE ONLINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Calcul de la matrice ICA (Cho2017, signal large bande) ---")
all_data_raw = []
for i in range(1, N_SUBJECTS + 1):
    try:
        f_l = os.path.join(BASE_PATH, f"patient{i:02d}_epochs_before_correction_left.fif")
        f_r = os.path.join(BASE_PATH, f"patient{i:02d}_epochs_before_correction_right.fif")
        ep_l = mne.read_epochs(f_l, preload=True, verbose='ERROR')
        ep_r = mne.read_epochs(f_r, preload=True, verbose='ERROR')
        ep   = mne.concatenate_epochs([ep_l, ep_r])
        ep.pick_channels([ch for ch in EPOC_CHANNELS if ch in ep.ch_names])
        all_data_raw.append(ep.get_data().reshape(len(ep.ch_names), -1).T)
    except FileNotFoundError:
        pass

data_concat = np.vstack(all_data_raw)  # (n_samples_total, n_channels)
n_ch        = data_concat.shape[1]

ica_sk  = FastICA(n_components=n_ch, random_state=42, max_iter=500, tol=0.01)
sources = ica_sk.fit_transform(data_concat)
W       = ica_sk.components_
W_pinv  = np.linalg.pinv(W)

frontal_signal    = data_concat[:, FRONTAL_IDX].mean(axis=1)
corr              = np.array([abs(np.corrcoef(sources[:, k], frontal_signal)[0, 1]) for k in range(n_ch)])
ocular_components = np.argsort(corr)[-2:].tolist()
print(f"  Composantes oculaires (Cho2017) : {ocular_components}")
print(f"  Corrélations frontales          : {corr[ocular_components].round(3).tolist()}")

ica_matrix = {
    'W': W, 'W_pinv': W_pinv,
    'ocular_components': ocular_components,
    'channels': EPOC_CHANNELS,
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
print("\nSauvegarde des modèles...")
to_save = {
    'ica_matrix.pkl':       ica_matrix,
}
for fname, obj in to_save.items():
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  → {path}")

print("\n✓ Modèles sauvegardés.")
print(f"\nRésumé des performances (CV sur l'ensemble des {N_SUBJECTS} sujets) :")
print(f"  LDA — Avant ICA : {score_lda_b*100:.1f}%  |  Après ICA : {score_lda_a*100:.1f}%")
print(f"  SVM — Avant ICA : {score_svm_b*100:.1f}%  |  Après ICA : {score_svm_a*100:.1f}%")
