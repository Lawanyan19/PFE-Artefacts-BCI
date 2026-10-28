# -*- coding: utf-8 -*-
"""
Prépare les matrices X / y pour la tâche Gauche vs Droite.
Filtre 8-30 Hz, crop sur la fenêtre d'imagination motrice (0.5 → 2.0 s).
Sauvegarde : data_before.pkl  /  data_after.pkl
"""

import mne
import numpy as np
import os
import pickle

base_path = "data/epochs"


# ─────────────────────────────────────────────────────────────────────────────
# 1. FONCTION DE PRÉPARATION
# ─────────────────────────────────────────────────────────────────────────────
def prepare_X_y(file_left: str, file_right: str):
    """
    Charge les epochs gauche et droite, filtre en bande motrice,
    crop sur la fenêtre d'imagination et retourne X, y.

    Retourne
    --------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)   0 = gauche, 1 = droite
    """
    epochs_left  = mne.read_epochs(file_left,  preload=True, verbose='ERROR')
    epochs_right = mne.read_epochs(file_right, preload=True, verbose='ERROR')

    epochs_full = mne.concatenate_epochs([epochs_left, epochs_right])

    # Bande motrice (mu + bêta)
    epochs_full.filter(l_freq=8.0, h_freq=30.0, method='iir', verbose='ERROR')

    # Fenêtre d'imagination motrice uniquement
    epochs_full.crop(tmin=0.5, tmax=2.0)

    X = epochs_full.get_data()
    y = np.concatenate([
        np.zeros(len(epochs_left)),   # 0 → gauche
        np.ones(len(epochs_right))    # 1 → droite
    ])

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. BOUCLE SUR LES SUJETS
# ─────────────────────────────────────────────────────────────────────────────
data_before: dict = {}
data_after:  dict = {}

print("Début de l'extraction Gauche vs Droite...")

for i in range(1, 21):
    print(f"\n--- Traitement du Sujet {i:02d} ---")

    try:
        # ── AVANT correction ──────────────────────────────────────────────────
        f_bl = os.path.join(base_path, f"patient{i:02d}_epochs_before_correction_left.fif")
        f_br = os.path.join(base_path, f"patient{i:02d}_epochs_before_correction_right.fif")
        X_b, y_b = prepare_X_y(f_bl, f_br)
        data_before[i] = {'X': X_b, 'y': y_b}
        print(f"  AVANT  — X: {X_b.shape} | y: {y_b.shape}")

        # ── APRÈS correction ──────────────────────────────────────────────────
        f_al = os.path.join(base_path, f"patient{i:02d}_epochs_after_correction_left.fif")
        f_ar = os.path.join(base_path, f"patient{i:02d}_epochs_after_correction_right.fif")
        X_a, y_a = prepare_X_y(f_al, f_ar)
        data_after[i] = {'X': X_a, 'y': y_a}
        print(f"  APRÈS  — X: {X_a.shape} | y: {y_a.shape}")

    except FileNotFoundError as e:
        print(f"  [!] Fichier introuvable : {e.filename}. Passage au suivant.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
print("\nSauvegarde des matrices Gauche/Droite...")

with open(os.path.join(base_path, 'data_before.pkl'), 'wb') as f:
    pickle.dump(data_before, f)

with open(os.path.join(base_path, 'data_after.pkl'), 'wb') as f:
    pickle.dump(data_after, f)

print("Terminé !")
