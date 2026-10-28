# -*- coding: utf-8 -*-
"""
Pipeline BCI temps réel — EMOTIV EPOC+ via CyKit (WebSocket ws://127.0.0.1:5555)

Phase 0  : Connexion WebSocket
Phase 1  : Calibration ICA (30s clignements)
Phase 2  : Calibration sujet (20 essais)
Phase 3  : Session de test (20 essais)
Phase 4  : Résumé + export CSV + AUC-ROC calculée sur la session complète

"""

import os, sys, time, threading, pickle, queue, csv, random, datetime
import numpy as np
import tkinter as tk
from scipy.signal import butter, sosfilt
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.decoding import CSP
import websocket

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
WS_URL      = "ws://127.0.0.1:5555"
OUTPUT_DIR = "models"
RESULTS_DIR = "results"

FS = 128
CYKIT_ORDER   = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
BAD_CHANNELS  = ["T7", "T8"]
ACTIVE_CH     = [ch for ch in CYKIT_ORDER if ch not in BAD_CHANNELS]
ACTIVE_IDX    = [CYKIT_ORDER.index(ch) for ch in ACTIVE_CH]
N_CHANNELS    = len(ACTIVE_CH)   # 12
FRONTAL_IDX   = [ACTIVE_CH.index(ch) for ch in ["AF3","AF4","F7","F8"] if ch in ACTIVE_CH]

# Timing
T_FIXATION      = 1.0    # croix de fixation
T_CUE           = 0.5    # affichage du cue
T_IMAGINE       = 2.5    # collecte signal (on classifie sur 0.5→2.0s)
T_PAUSE_CALIB   = 1.5    # pause entre essais de calibration
T_PAUSE_TEST    = 1.5    # pause entre essais de test

N_CALIB_TRIALS  = 20     # essais de calibration sujet (10 G + 10 D)
N_TEST_TRIALS   = 20     # essais de test (10 G + 10 D)

# Fenêtre de classification dans l'epoch (identique à l'offline)
CLF_START = int(0.5 * FS)   # sample 64
CLF_END   = int(2.0 * FS)   # sample 256

# Seuil rejet musculaire (variance bande >40Hz)
MUSCLE_THRESHOLD = 5e-10

# ── Buffers pour enregistrement continu pendant la session de test ───────────
_record_raw = []    # échantillons bruts continus
_record_ica = []    # échantillons ICA continus 
_recording  = False # True pendant Phase 3

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASSE CSP_SVM_WRAPPER
#    Définie ICI pour que pickle puisse la retrouver au chargement de ica_matrix
# ─────────────────────────────────────────────────────────────────────────────
class CSP_SVM_Wrapper:
    """Encapsule CSP + SVM avec interface predict() identique au pipeline LDA."""
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

# ─────────────────────────────────────────────────────────────────────────────
# 3. CHARGEMENT MATRICE ICA (Cho2017)
# ─────────────────────────────────────────────────────────────────────────────
print("Chargement de la matrice ICA Cho2017...")
try:
    with open(os.path.join(OUTPUT_DIR, 'ica_matrix.pkl'), 'rb') as f:
        ica_base = pickle.load(f)
    print("  ✓ Matrice ICA chargée.")
except FileNotFoundError:
    print("[ERREUR] ica_matrix.pkl introuvable. Lance d'abord retrain_epoc_channels.py")
    sys.exit(1)

# Matrice ICA active (sera affinée lors de la calibration oculaire)
W_active      = ica_base['W'].copy()
W_pinv_active = ica_base['W_pinv'].copy()
ocular_active = list(ica_base['ocular_components'])

# Classifieurs entraînés lors de la calibration sujet (initialisés à None)
clf_lda_raw  = None   # CSP + LDA sur signal brut
clf_lda_ica  = None   # CSP + LDA sur signal corrigé ICA
clf_svm_raw  = None   # CSP + SVM sur signal brut
clf_svm_ica  = None   # CSP + SVM sur signal corrigé ICA

# ─────────────────────────────────────────────────────────────────────────────
# 4. FILTRES CAUSAUX
# ─────────────────────────────────────────────────────────────────────────────
sos_hp = butter(4, 1.0,         btype='highpass', fs=FS, output='sos')
sos_bp = butter(4, [8.0, 30.0], btype='bandpass', fs=FS, output='sos')
sos_mu = butter(4, 40.0,        btype='highpass', fs=FS, output='sos')

# État causal du filtre HP (conservé entre epochs pour continuité)
zi_hp = np.zeros((sos_hp.shape[0], 2, N_CHANNELS))

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUFFER EEG & WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────
eeg_queue = queue.Queue()

def parse_cykit(msg: str):
    try:
        p = msg.strip().split(',')
        if len(p) >= 16:
            vals = (np.array(p[2:16], dtype=float) - 4200) * 1e-6
            return vals[ACTIVE_IDX]   # (12,)
    except (ValueError, IndexError):
        pass
    return None

def on_ws_message(ws, msg):
    s = parse_cykit(msg)
    if s is not None:
        eeg_queue.put(s)
        # Enregistrement continu pendant la session de test
        if _recording:
            _record_raw.append(s.copy())

def connect_websocket():
    ws = websocket.WebSocketApp(WS_URL, on_message=on_ws_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()
    print("Connexion WebSocket CyKit...")
    t0 = time.time()
    while eeg_queue.empty():
        if time.time() - t0 > 10:
            print("[ERREUR] Aucune donnée reçue. CyKit est-il lancé ?")
            sys.exit(1)
        time.sleep(0.1)
    print(f"  ✓ Stream actif — canaux : {ACTIVE_CH}")

def collect_epoch(duration: float) -> np.ndarray:
    """Collecte `duration` secondes → (n_channels, n_samples)."""
    n = int(duration * FS)
    buf = []
    while len(buf) < n:
        try:
            buf.append(eeg_queue.get(timeout=2.0))
        except queue.Empty:
            print("[WARN] Signal interrompu.")
            break
    return np.array(buf[:n]).T

def flush_queue():
    while not eeg_queue.empty():
        eeg_queue.get_nowait()

# ─────────────────────────────────────────────────────────────────────────────
# 6. PREPROCESSING ONLINE
# ─────────────────────────────────────────────────────────────────────────────
def causal_hp(data: np.ndarray) -> np.ndarray:
    """Filtre passe-haut causal 1Hz avec conservation de l'état inter-epochs."""
    global zi_hp
    out = np.zeros_like(data)
    for ch in range(N_CHANNELS):
        out[ch], zi_hp[:, :, ch] = sosfilt(sos_hp, data[ch], zi=zi_hp[:, :, ch])
    return out

def is_muscle(data: np.ndarray) -> bool:
    """True si variance bande >40Hz dépasse le seuil → epoch rejetée."""
    hf = sosfilt(sos_mu, data, axis=1)
    return float(np.var(hf, axis=1).mean()) > MUSCLE_THRESHOLD

def apply_ica(data: np.ndarray) -> np.ndarray:
    """Démixage ICA + suppression composantes oculaires + reconstruction."""
    sources = W_active @ data
    sources[ocular_active, :] = 0
    return W_pinv_active @ sources

def prepare(data: np.ndarray) -> np.ndarray:
    """Crop [0.5→2.0s] + filtre 8-30Hz → (1, n_ch, n_samples)."""
    ep = data[:, CLF_START:CLF_END]
    ep = sosfilt(sos_bp, ep, axis=1)
    return ep[np.newaxis, :, :]

# ─────────────────────────────────────────────────────────────────────────────
# 7. CALIBRATION ICA OCULAIRE (30s de clignements)
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_ica_ocular(ui_update):
    global W_active, W_pinv_active, ocular_active
    ui_update("CALIBRATION ICA\n\nClignez des yeux régulièrement\npendant 30 secondes", "#FFA500")
    time.sleep(3)
    flush_queue()
    ui_update("Clignez !", "#FFA500")
    data = collect_epoch(30.0)
    ui_update("Calcul ICA en cours...", "#FFA500")
    data_hp = sosfilt(sos_hp, data, axis=1)
    try:
        n_ch    = data_hp.shape[0]
        ica_cal = FastICA(n_components=n_ch, random_state=42, max_iter=500, tol=0.01)
        src     = ica_cal.fit_transform(data_hp.T)
        W_cal   = ica_cal.components_
        frontal = data_hp[FRONTAL_IDX, :].mean(axis=0)
        corr    = np.array([abs(pearsonr(src[:, k], frontal)[0]) for k in range(n_ch)])
        ocu     = np.argsort(corr)[-2:].tolist()
        W_active      = W_cal
        W_pinv_active = np.linalg.pinv(W_cal)
        ocular_active = ocu
        ui_update(f"Calibration ICA ✓\nComposantes oculaires : {ocu}", "#00FF00")
        print(f"  ICA calibration OK — composantes : {ocu}")
    except Exception as e:
        ui_update("Matrice ICA Cho2017 conservée (fallback)", "#FF8800")
        print(f"  [WARN] ICA calibration échouée ({e}). Fallback Cho2017.")
    time.sleep(2)

# ─────────────────────────────────────────────────────────────────────────────
# 8. CALIBRATION SUJET : collecte N_CALIB_TRIALS epochs et entraîne les 4 clf
# ─────────────────────────────────────────────────────────────────────────────
def run_calibration_subject(ui: 'BCI_UI'):
    """
    Collecte N_CALIB_TRIALS epochs annotées (GAUCHE/DROITE),
    entraîne CSP+LDA et CSP+SVM sur signal brut ET signal corrigé ICA.
    C'est cette étape qui personnalise les classifieurs au sujet du jour.
    """
    global clf_lda_raw, clf_lda_ica, clf_svm_raw, clf_svm_ica

    trials = [0] * (N_CALIB_TRIALS // 2) + [1] * (N_CALIB_TRIALS // 2)
    random.shuffle(trials)

    X_raw_list, X_ica_list, y_list = [], [], []
    rejected = 0

    ui.update("CALIBRATION SUJET\n\nSuivez les instructions\n(pas de retour de prédiction)", "#00AAFF")
    time.sleep(3)

    for idx, label in enumerate(trials):
        if not ui.running:
            return False

        cue_txt   = "← GAUCHE" if label == 0 else "DROITE →"
        cue_color = "#4488FF"  if label == 0 else "#FF4444"
        prog      = f"Calibration {idx+1}/{N_CALIB_TRIALS}"

        ui.update("+", "white", "", prog)
        flush_queue()
        time.sleep(T_FIXATION)

        ui.update(cue_txt, cue_color, "Imaginez le mouvement...", prog)
        time.sleep(T_CUE)

        ui.update("...", cue_color, "Collecte...", prog)
        raw = collect_epoch(T_IMAGINE)
        hp  = causal_hp(raw)

        if is_muscle(hp):
            ui.update("⚠ Artefact musculaire — essai ignoré", "#FF8800", "", prog)
            rejected += 1
            time.sleep(1.5)
            continue

        X_raw_list.append(prepare(hp)[0])        # (n_ch, n_samples)
        X_ica_list.append(prepare(apply_ica(hp))[0])
        y_list.append(label)

        ui.update(cue_txt, cue_color, "✓", prog)
        time.sleep(T_PAUSE_CALIB)

    n_valid  = len(y_list)
    n_gauche = int(np.sum(np.array(y_list) == 0))
    n_droite = int(np.sum(np.array(y_list) == 1))
    print(f"  Calibration : {n_valid} epochs valides / {N_CALIB_TRIALS} "
          f"({rejected} rejetées) — GAUCHE:{n_gauche} DROITE:{n_droite}")

    if n_valid < 10:
        ui.update(f"Trop peu d'epochs valides ({n_valid}).\nRelancez la calibration.", "#FF0000")
        time.sleep(4)
        return False

    # Vérification d'équilibre des classes
    # Si une classe a moins de 3 epochs, le modèle sera dégénéré
    if n_gauche < 3 or n_droite < 3:
        ui.update(
            f"Déséquilibre trop fort (G:{n_gauche} D:{n_droite}).\n"
            "Réduisez les mouvements parasites et relancez.", "#FF0000"
        )
        time.sleep(4)
        return False

    # Équilibrage par under-sampling de la classe majoritaire
    # → garantit que le modèle ne se bloque pas sur une seule classe
    y_arr = np.array(y_list)
    n_min = min(n_gauche, n_droite)
    idx_g = np.where(y_arr == 0)[0][:n_min]
    idx_d = np.where(y_arr == 1)[0][:n_min]
    idx   = np.concatenate([idx_g, idx_d])
    np.random.shuffle(idx)

    X_raw_list = [X_raw_list[i] for i in idx]
    X_ica_list = [X_ica_list[i] for i in idx]
    y_list     = [y_list[i]     for i in idx]
    print(f"  Après équilibrage : {len(y_list)} epochs (G:{n_min} D:{n_min})")

    X_raw = np.array(X_raw_list)
    X_ica = np.array(X_ica_list)
    y     = np.array(y_list)

    ui.update("Entraînement des classifieurs...", "#FFA500")

    # ── CSP + LDA ────────────────────────────────────────────────────────────
    from sklearn.pipeline import make_pipeline
    clf_lda_raw = make_pipeline(
        CSP(n_components=4, reg='shrinkage', log=True, norm_trace=False), LDA()
    )
    clf_lda_ica = make_pipeline(
        CSP(n_components=4, reg='shrinkage', log=True, norm_trace=False), LDA()
    )
    clf_lda_raw.fit(X_raw, y)
    clf_lda_ica.fit(X_ica, y)

    # ── CSP + SVM (GridSearch rapide sur C) ──────────────────────────────────
    # Inner 3-fold uniquement si on a assez d'epochs
    n_splits_inner = min(3, n_valid // 2)

    def make_svm_clf(X, y):
        csp = CSP(n_components=4, reg='shrinkage', log=True, norm_trace=False)
        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=0)
        gs = GridSearchCV(
            Pipeline([('svc', SVC(kernel='rbf', probability=True))]),
            param_grid={'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 'auto']},
            cv=inner_cv, scoring='accuracy', n_jobs=-1, refit=True
        )
        csp.fit(X, y)
        gs.fit(csp.transform(X), y)
        return CSP_SVM_Wrapper(csp, gs)

    clf_svm_raw = make_svm_clf(X_raw, y)
    clf_svm_ica = make_svm_clf(X_ica, y)

    print(f"  Classifieurs entraînés sur {n_valid} epochs.")



    ui.update("Classifieurs prêts ✓\nSession de test dans 3s", "#00FF00")
    time.sleep(3)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# 9. SESSION DE TEST
# ─────────────────────────────────────────────────────────────────────────────
def run_test_session(ui: 'BCI_UI') -> list:
    """
    N_TEST_TRIALS essais avec prédiction en temps réel par les 4 classifieurs.
    Retourne l'historique des essais pour export CSV.
    """
    global _recording, _record_raw, _record_ica
    _record_raw.clear()
    _record_ica.clear()
    _recording = True   # démarre l'enregistrement continu

    trials = [0] * (N_TEST_TRIALS // 2) + [1] * (N_TEST_TRIALS // 2)
    random.shuffle(trials)
    history = []

    for idx, true_label in enumerate(trials):
        if not ui.running:
            break

        cue_txt   = "← GAUCHE" if true_label == 0 else "DROITE →"
        cue_color = "#4488FF"  if true_label == 0 else "#FF4444"
        prog      = f"Test {idx+1}/{N_TEST_TRIALS}"

        ui.update("+", "white", "", prog)
        flush_queue()
        time.sleep(T_FIXATION)

        ui.update(cue_txt, cue_color, "Imaginez le mouvement...", prog)
        time.sleep(T_CUE)

        ui.update("...", cue_color, "Collecte...", prog)
        raw = collect_epoch(T_IMAGINE)
        hp  = causal_hp(raw)

        if is_muscle(hp):
            ui.update("⚠ Artefact musculaire — essai ignoré", "#FF8800", "", prog)
            time.sleep(2)
            continue

        X_raw = prepare(hp)
        X_ica = prepare(apply_ica(hp))

        try:
            p_lda_b = int(clf_lda_raw.predict(X_raw)[0])
            p_lda_a = int(clf_lda_ica.predict(X_ica)[0])
            p_svm_b = int(clf_svm_raw.predict(X_raw)[0])
            p_svm_a = int(clf_svm_ica.predict(X_ica)[0])

            # Scores de confiance pour l'AUC
            s_lda_b = float(clf_lda_raw.decision_function(X_raw)[0])
            s_lda_a = float(clf_lda_ica.decision_function(X_ica)[0])
            s_svm_b = float(clf_svm_raw.predict_proba(X_raw)[0][1])
            s_svm_a = float(clf_svm_ica.predict_proba(X_ica)[0][1])

        except Exception as e:
            print(f"  [WARN] Prédiction essai {idx+1} : {e}")
            continue

        history.append({
            'essai':     idx + 1,
            'vrai':      true_label,
            'lda_avant': p_lda_b, 'lda_apres': p_lda_a,
            'svm_avant': p_svm_b, 'svm_apres': p_svm_a,
            '_s_lda_b':  s_lda_b, '_s_lda_a': s_lda_a,
            '_s_svm_b':  s_svm_b, '_s_svm_a': s_svm_a,
        })

        # (enregistrement continu géré par le thread WebSocket — voir _record_raw)

        def sym(p, t): return "✓" if p == t else "✗"
        result_str = (
            f"Vrai : {cue_txt}\n\n"
            f"LDA avant ICA : {'← GAUCHE' if p_lda_b==0 else 'DROITE →'} {sym(p_lda_b,true_label)}    "
            f"LDA après ICA : {'← GAUCHE' if p_lda_a==0 else 'DROITE →'} {sym(p_lda_a,true_label)}\n"
            f"SVM avant ICA : {'← GAUCHE' if p_svm_b==0 else 'DROITE →'} {sym(p_svm_b,true_label)}    "
            f"SVM après ICA : {'← GAUCHE' if p_svm_a==0 else 'DROITE →'} {sym(p_svm_a,true_label)}"
        )
        ui.update(cue_txt, cue_color, result_str, prog)
        time.sleep(T_PAUSE_TEST)

    _recording = False  # arrête l'enregistrement continu
    return history

# ─────────────────────────────────────────────────────────────────────────────
# 10. RÉSUMÉ ET EXPORT CSV
# ─────────────────────────────────────────────────────────────────────────────
def permutation_test(y_true: np.ndarray, y_pred: np.ndarray,
                     n_permutations: int = 1000) -> float:
    """
    Test de permutation : estime la p-value empirique de l'accuracy observée.
    H0 : le classifieur ne fait pas mieux que le hasard.
    Retourne p = proportion de permutations >= accuracy observée.
    """
    obs = np.mean(y_pred == y_true)
    rng = np.random.default_rng(0)
    count = 0
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_true)
        if np.mean(y_perm == y_pred) >= obs:
            count += 1
    return count / n_permutations





def save_epochs_fif(ts: str):
    """
    Sauvegarde le signal brut continu enregistré pendant la session de test,
    ainsi que sa version corrigée par ICA, en fichiers .fif Raw.
    Ces fichiers peuvent être ouverts avec MNE pour comparaison visuelle
    avant/après correction (équivalent Figure 19 du rapport, en conditions online).
    """
    if not _record_raw:
        print("  [WARN] Aucun signal enregistré.")
        return

    info = mne.create_info(ch_names=ACTIVE_CH, sfreq=FS, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')

    # Signal brut continu : (n_samples, n_channels) → (n_channels, n_samples)
    data_raw = np.array(_record_raw).T   # (12, n_samples)

    # Version corrigée ICA du signal continu
    W       = W_active
    n_comp  = W.shape[0]
    sources = W @ data_raw                       # (n_comp, n_samples)
    sources[ocular_active] = 0                   # ocular_active = liste globale
    data_ica = np.linalg.pinv(W) @ sources       # (12, n_samples)

    raw_mne = mne.io.RawArray(data_raw, info, verbose='ERROR')
    ica_mne = mne.io.RawArray(data_ica, info, verbose='ERROR')

    path_raw = os.path.join(RESULTS_DIR, f"online_signal_brut_{ts}_raw.fif")
    path_ica = os.path.join(RESULTS_DIR, f"online_signal_ica_{ts}_raw.fif")
    raw_mne.save(path_raw, overwrite=True, verbose='ERROR')
    ica_mne.save(path_ica, overwrite=True, verbose='ERROR')
    print(f"  Signal brut continu → {path_raw}")
    print(f"  Signal ICA continu  → {path_ica}")


def save_and_summarize(history: list, ui: 'BCI_UI'):
    if not history:
        ui.update("Aucun essai valide.", "#FF8800")
        return

    y_true = np.array([r['vrai'] for r in history])
    n_valid = len(history)

    def acc(key):
        return np.mean([r[key] == r['vrai'] for r in history]) * 100

    def auc(skey):
        scores = np.array([r[skey] for r in history])
        try:
            return roc_auc_score(y_true, scores)
        except Exception:
            return float('nan')

    acc_lb = acc('lda_avant');  auc_lb = auc('_s_lda_b')
    acc_la = acc('lda_apres');  auc_la = auc('_s_lda_a')
    acc_sb = acc('svm_avant');  auc_sb = auc('_s_svm_b')
    acc_sa = acc('svm_apres');  auc_sa = auc('_s_svm_a')

    # ── Tests de permutation (1000 tirages) ──────────────────────────────────
    print("\n  Tests de permutation (1000 tirages)...")
    y_pred_lb = np.array([r['lda_avant'] for r in history])
    y_pred_la = np.array([r['lda_apres'] for r in history])
    y_pred_sb = np.array([r['svm_avant'] for r in history])
    y_pred_sa = np.array([r['svm_apres'] for r in history])

    p_lb = permutation_test(y_true, y_pred_lb)
    p_la = permutation_test(y_true, y_pred_la)
    p_sb = permutation_test(y_true, y_pred_sb)
    p_sa = permutation_test(y_true, y_pred_sa)

    def sig(p): return "✓ sig." if p < 0.05 else "n.s."

    summary = (
        "SESSION TERMINÉE\n\n"
        f"LDA avant ICA : {acc_lb:.0f}%  (AUC {auc_lb:.3f})  p={p_lb:.3f} {sig(p_lb)}\n"
        f"LDA après ICA : {acc_la:.0f}%  (AUC {auc_la:.3f})  p={p_la:.3f} {sig(p_la)}\n"
        f"SVM avant ICA : {acc_sb:.0f}%  (AUC {auc_sb:.3f})  p={p_sb:.3f} {sig(p_sb)}\n"
        f"SVM après ICA : {acc_sa:.0f}%  (AUC {auc_sa:.3f})  p={p_sa:.3f} {sig(p_sa)}\n\n"
        f"({n_valid} essais valides / {N_TEST_TRIALS} total)"
    )
    ui.update(summary, "#00FF00")

    print(f"\n{'─'*50}")
    print(f"LDA avant ICA : {acc_lb:.0f}%  AUC {auc_lb:.3f}  p={p_lb:.3f} {sig(p_lb)}")
    print(f"LDA après ICA : {acc_la:.0f}%  AUC {auc_la:.3f}  p={p_la:.3f} {sig(p_la)}")
    print(f"SVM avant ICA : {acc_sb:.0f}%  AUC {auc_sb:.3f}  p={p_sb:.3f} {sig(p_sb)}")
    print(f"SVM après ICA : {acc_sa:.0f}%  AUC {auc_sa:.3f}  p={p_sa:.3f} {sig(p_sa)}")
    print(f"{'─'*50}")

    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV : prédictions par essai + résumé ──────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, f"online_results_{ts}.csv")
    lbl = {0: "GAUCHE", 1: "DROITE"}

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        f.write("Essai;Vrai;LDA avant ICA;LDA après ICA;SVM avant ICA;SVM après ICA\n")
        for r in history:
            f.write(
                f"{r['essai']};{lbl[r['vrai']]};"
                f"{lbl[r['lda_avant']]};{lbl[r['lda_apres']]};"
                f"{lbl[r['svm_avant']]};{lbl[r['svm_apres']]}\n"
            )
        f.write("\n")
        f.write(f"Essais valides;{n_valid}/{N_TEST_TRIALS}\n")
        f.write(";;Accuracy;AUC;p-value (permutation);Significatif\n")
        f.write(f"LDA avant ICA;;{acc_lb:.1f}%;{auc_lb:.3f};{p_lb:.3f};{sig(p_lb)}\n")
        f.write(f"LDA après ICA;;{acc_la:.1f}%;{auc_la:.3f};{p_la:.3f};{sig(p_la)}\n")
        f.write(f"SVM avant ICA;;{acc_sb:.1f}%;{auc_sb:.3f};{p_sb:.3f};{sig(p_sb)}\n")
        f.write(f"SVM après ICA;;{acc_sa:.1f}%;{auc_sa:.3f};{p_sa:.3f};{sig(p_sa)}\n")
        f.write("\nNote : p-value par test de permutation (1000 tirages). "
                "Seuil de significativité : p < 0.05\n")

    print(f"  Résultats CSV → {csv_path}")

    # ── Sauvegarde des epochs .fif ────────────────────────────────────────────
    try:
        save_epochs_fif(ts)
    except Exception as e:
        print(f"  [WARN] Sauvegarde .fif échouée : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. INTERFACE GRAPHIQUE
# ─────────────────────────────────────────────────────────────────────────────
class BCI_UI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BCI Online — EPOC+ (12 canaux)")
        self.root.geometry("1300x750")
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.running = True

        self.lbl_stim = tk.Label(self.root, text="", font=("Arial", 65, "bold"),
                                  bg="black", fg="white", wraplength=1200, justify="center")
        self.lbl_stim.pack(expand=True)

        self.lbl_res = tk.Label(self.root, text="", font=("Arial", 19),
                                 bg="black", fg="#CCCCCC", wraplength=1200, justify="center")
        self.lbl_res.pack(pady=10)

        self.lbl_prog = tk.Label(self.root, text="", font=("Arial", 13),
                                  bg="black", fg="#555555")
        self.lbl_prog.pack(pady=5)

    def update(self, stim: str, color: str = "white", result: str = "", prog: str = ""):
        def _do():
            self.lbl_stim.config(text=stim, fg=color)
            self.lbl_res.config(text=result)
            self.lbl_prog.config(text=prog)
        self.root.after(0, _do)

    def _close(self):
        self.running = False
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()

# ─────────────────────────────────────────────────────────────────────────────
# 12. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    connect_websocket()
    ui = BCI_UI()

    def experiment():
        # Phase 1 : calibration ICA oculaire
        calibrate_ica_ocular(lambda t, c: ui.update(t, c))

        # Phase 2 : calibration sujet (entraîne les classifieurs)
        ok = run_calibration_subject(ui)
        if not ok:
            return

        # Phase 3 : session de test
        history = run_test_session(ui)

        # Phase 4 : résumé + export
        save_and_summarize(history, ui)

    threading.Thread(target=experiment, daemon=True).start()
    ui.mainloop()

if __name__ == "__main__":
    main()
