# -*- coding: utf-8 -*-

import mne
import matplotlib
matplotlib.use('Qt5Agg')

from moabb.datasets import Cho2017


for i in range (7, 11):
    # 1. CHARGEMENT DU DATASET
    dataset = Cho2017()
    data = dataset.get_data(subjects=[i])
    raw = data[i]["0"]["0"]
    print(f"Signal chargé: {len(raw.ch_names)} canaux, fs={raw.info['sfreq']} Hz")
    
    
    # 2. NETTOYAGE PRELIMINAIRE (CANAUX ET PASSE-HAUT)
    
    # A. Suppression des canaux inutiles
    channels_to_drop = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'Stim', 'P10', 'TP10', 'FT10', 'P9', 'TP9', 'FT9']
    raw.drop_channels(channels_to_drop, on_missing='ignore')
    print(f"Canaux supprimés : {channels_to_drop}")
    
    raw.reorder_channels(['Fpz',
            'AF7', 'AF3', 'AF4', 'AF8', 
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'Oz', 'O2'])
    
    # B. Filtre PASSE-HAUT
    raw.filter(l_freq=1.0, h_freq=None, method='fir', phase='zero', verbose='info')
    print("Filtre Passe-Haut (1 Hz) appliqué.")
    
    # C. Interpolation
    print("\n--- MARQUAGE MANUEL DES CANAUX ---")
    raw.plot(
        n_channels=29,              # Affiche 29 canaux d'un coup
        scalings={'eeg': 1450e-6},  # Règle l'échelle sur 2200 µV
        block=True, 
        title=f"Sujet {i} - Marquage Bad Channels pour interpolation"
    )
    
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True)
        print(f"Interpolation effectuée sur : {raw.info['bads']}")
    else:
        print("Aucun canal interpolé.")

    raw.plot(
        n_channels=29,              # Affiche 29 canaux d'un coup
        scalings={'eeg': 1450e-6},  # Règle l'échelle sur 2200 µV
        block=True, 
        title=f"Sujet {i} - Visualisation après interpolation des canaux"
    )

    # 3. EPOCHING ET REJET DES ARTEFACTS DE MUSCLES
    print("\n--- CREATION DES EPOCHS ET REJET DES ARTEFACTS DE MUSCLES ---")
    
    # Création des events
    events, event_id = mne.events_from_annotations(raw)
    
    # On découpe les epochs
    epochs_raw = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=-2.0, tmax=2.0, baseline=(-2.0, -0.5),
        preload=True, reject_by_annotation=True
    )
    
    # On Nettoie les epochs musclés
    epochs_raw.plot(
        n_epochs=5,                 # Affiche seulement 5 epochs à la fois
        n_channels=29,              # Affiche les 29 canaux
        scalings={'eeg': 1450e-6},  # Echelle fixée à 2900 µV
        block=True, 
        title=f"Sujet {i} - Rejet Muscles (Clic pour supprimer)"
    )
    
    print(f"Suppression des epochs marquées")
    epochs_raw.drop_bad()

    # Séparation en gauche droite pour le nettoyage
    epochs_left_raw = epochs_raw['left_hand'].copy()
    epochs_right_raw = epochs_raw['right_hand'].copy()

    # Sauvegarde mes raws séparés
    epochs_left_raw.save(f"patient{i:02d}_epochs_before_correction_left.fif", overwrite=True)
    epochs_right_raw.save(f"patient{i:02d}_epochs_before_correction_right.fif", overwrite=True)
    
    # Fit ICA using FastICA algorithm
    ica = mne.preprocessing.ICA(n_components=0.99, method='fastica', random_state=42)
    ica.fit(epochs_raw)
    
    # Interactive plots for manual inspection
    ica.plot_sources(epochs_raw)
    ica.plot_components()
    
    
    epochs_corr = ica.apply(epochs_raw.copy())
    
    epochs_raw.plot(
        n_epochs=5, n_channels=29, scalings={'eeg': 1450e-6}, 
        block=False, title=f"Sujet {i} - AVANT Correction"
    )
    epochs_corr.plot(
        n_epochs=5, n_channels=29, scalings={'eeg': 1450e-6}, 
        block=True, title=f"Sujet {i} - APRÈS Correction"
    )
    
    # Séparation en gauche droite pour le nettoyage
    epochs_left_corr = epochs_corr['left_hand'].copy()
    epochs_right_corr = epochs_corr['right_hand'].copy()
    
    epochs_left_corr.save(f"patient{i:02d}_epochs_after_correction_left.fif", overwrite=True)
    epochs_right_corr.save(f"patient{i:02d}_epochs_after_correction_right.fif", overwrite=True)