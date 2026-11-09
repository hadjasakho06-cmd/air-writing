import cv2                  # OpenCV : capture vidéo, affichage, dessin
import mediapipe as mp      # MediaPipe : détection des landmarks de la main
import numpy as np          # manipulation de tableaux numériques (non utilisé directement ici mais utile en général)
import os                   # gestion des fichiers et dossiers
import json                 # sauvegarde des trajectoires au format JSON

# ── Configuration ──────────────────────────────────────────
# Liste des classes à collecter : 26 lettres (A-Z) + 10 chiffres (0-9) = 36 classes
CLASSES = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    [str(i) for i in range(10)]
)

SAMPLES_PER_CLASS = 30        # nombre d'échantillons à collecter par classe
SAVE_DIR          = "dataset2"  # dossier de sauvegarde (nouveau, pour ne pas écraser l'ancien dataset)
os.makedirs(SAVE_DIR, exist_ok=True)  # crée le dossier s'il n'existe pas déjà

# ── Initialisation MediaPipe ───────────────────────────────
mp_hands   = mp.solutions.hands           # module de détection de la main
mp_drawing = mp.solutions.drawing_utils   # module pour dessiner les landmarks sur l'image
hands      = mp_hands.Hands(
    static_image_mode=False,        # mode vidéo (suivi continu, pas image statique)
    max_num_hands=1,                # une seule main détectée à la fois
    min_detection_confidence=0.7,   # seuil de confiance minimal pour détecter une main
    min_tracking_confidence=0.7     # seuil de confiance minimal pour suivre la main d'une frame à l'autre
)

def get_index_tip(hand_landmarks, w, h):
    """Extrait les coordonnées pixel du bout de l'index (landmark 8)"""
    lm = hand_landmarks.landmark[8]              # landmark n°8 = bout de l'index
    return (int(lm.x * w), int(lm.y * h))        # conversion des coordonnées normalisées (0-1) en pixels

def collect_data():
    # Ouverture de la webcam (CAP_DSHOW = driver Windows, plus stable)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERREUR : Webcam non détectée.")
        exit()

    print("=== COLLECTE DE DONNÉES AIR WRITING ===")
    print("ESPACE = démarrer/arrêter le tracé | Q = quitter")

    # ── Boucle sur chaque classe à collecter (A, B, ..., Z, 0, ..., 9) ──
    for label in CLASSES:
        label_dir = os.path.join(SAVE_DIR, label)   # dossier spécifique à cette classe
        os.makedirs(label_dir, exist_ok=True)

        # Vérifie combien d'échantillons existent déjà pour cette classe
        existing = len(os.listdir(label_dir))
        if existing >= SAMPLES_PER_CLASS:
            print(f"[SKIP] {label} : {existing} échantillons déjà collectés")
            continue   # passe à la classe suivante si déjà complète

        print(f"\n>>> Classe : {label} | Restants : {SAMPLES_PER_CLASS - existing}")

        sample_count = existing   # reprend la collecte là où elle s'était arrêtée

        # ── Boucle pour collecter les échantillons de cette classe ──
        while sample_count < SAMPLES_PER_CLASS:
            sequence  = []        # liste des coordonnées (x, y) du tracé en cours
            recording = False     # indique si on est en train d'enregistrer un geste

            print(f"  Echantillon {sample_count+1}/{SAMPLES_PER_CLASS} "
                  f"— Appuyez sur ESPACE pour commencer")

            # ── Boucle de capture vidéo en temps réel ──
            while True:
                ret, frame = cap.read()       # lecture d'une frame de la webcam
                if not ret or frame is None:
                    continue                  # ignore les frames invalides

                frame   = cv2.flip(frame, 1)              # effet miroir (plus naturel pour l'utilisateur)
                h, w, _ = frame.shape                      # dimensions de l'image
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe attend du RGB, OpenCV fournit du BGR
                results = hands.process(rgb)               # détection de la main sur la frame

                # ── Bandeau d'instruction en haut de l'écran ──
                if not recording:
                    # Bandeau vert : en attente du démarrage
                    cv2.rectangle(frame, (0, 0), (w, 60), (0, 100, 0), -1)
                    cv2.putText(frame,
                                f"Classe: {label} | Echantillon: {sample_count+1}/{SAMPLES_PER_CLASS} | ESPACE pour commencer",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Bandeau rouge : enregistrement en cours
                    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 180), -1)
                    cv2.putText(frame,
                                f"Tracez {label}... ({len(sequence)} pts) | ESPACE pour terminer",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # ── Si une main est détectée dans l'image ──
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        # Dessine les 21 landmarks et leurs connexions sur l'image (retour visuel)
                        mp_drawing.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                        if recording:
                            # Récupère la position du bout de l'index et la normalise (0-1)
                            tip    = get_index_tip(hand_lm, w, h)
                            x_norm = tip[0] / w
                            y_norm = tip[1] / h
                            sequence.append([x_norm, y_norm])   # ajoute le point à la trajectoire

                            # Dessine la trajectoire déjà tracée (ligne rouge reliant les points)
                            if len(sequence) > 1:
                                for i in range(1, len(sequence)):
                                    p1 = (int(sequence[i-1][0] * w),
                                          int(sequence[i-1][1] * h))
                                    p2 = (int(sequence[i][0] * w),
                                          int(sequence[i][1] * h))
                                    cv2.line(frame, p1, p2, (0, 0, 255), 3)

                cv2.imshow("Air Writing - Collecte", frame)   # affiche la fenêtre vidéo
                key = cv2.waitKey(1) & 0xFF                    # capture la touche pressée

                # ── ESPACE : démarrer ou arrêter l'enregistrement ──
                if key == ord(' '):
                    if not recording:
                        # Démarrer un nouvel enregistrement
                        recording = True
                        sequence  = []
                        print("  → Enregistrement démarré !")
                    else:
                        # Arrêter l'enregistrement et sauvegarder le résultat
                        recording = False
                        if len(sequence) >= 5:   # ignore les tracés trop courts (probablement accidentels)
                            filename = os.path.join(
                                label_dir,
                                f"{label}_{sample_count:03d}.json"
                            )
                            # Sauvegarde la trajectoire et son label au format JSON
                            with open(filename, "w") as f:
                                json.dump({
                                    "label": label,
                                    "sequence": sequence
                                }, f)
                            print(f"  ✓ Sauvegardé : {filename} ({len(sequence)} pts)")
                            sample_count += 1
                            cv2.waitKey(500)   # petite pause avant le prochain échantillon
                            break              # sort de la boucle de capture pour passer à l'échantillon suivant
                        else:
                            print("  ⚠ Séquence trop courte, recommencez")

                # ── Q : quitter le programme à tout moment ──
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    # ── Fin de la collecte : libère la webcam et ferme les fenêtres ──
    cap.release()
    cv2.destroyAllWindows()
    print("\n=== COLLECTE TERMINÉE ===")
    print(f"Dataset sauvegardé dans : {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    collect_data()