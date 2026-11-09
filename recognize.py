import cv2                            # OpenCV : capture vidéo, affichage, dessin
import mediapipe as mp                # MediaPipe : détection des landmarks de la main
import numpy as np                    # manipulation de tableaux numériques
import pickle                         # chargement de l'encodeur de classes sauvegardé
from scipy.interpolate import interp1d        # interpolation pour normaliser les séquences
from tensorflow.keras.models import load_model  # chargement du modèle LSTM entraîné

# ── Chargement du modèle et de l'encodeur ─────────────────
model = load_model('model_airwriting.h5')   # charge le modèle LSTM pré-entraîné
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)                # charge l'encodeur (correspondance indice → symbole)

print("Modèle chargé !")
print(f"Classes : {list(encoder.classes_)}")   # affiche les 36 classes reconnues (A-Z, 0-9)

# ── Initialisation MediaPipe ───────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,        # mode vidéo (suivi continu)
    max_num_hands=1,                # une seule main à la fois
    min_detection_confidence=0.7,   # seuil de confiance pour détecter la main
    min_tracking_confidence=0.7     # seuil de confiance pour suivre la main d'une frame à l'autre
)

SEQUENCE_LENGTH = 30   # longueur fixe attendue par le modèle (comme à l'entraînement)

def normaliser_sequence(sequence, longueur_cible=30):
    """Ramène une séquence de longueur variable à une longueur fixe par interpolation"""
    sequence = np.array(sequence)
    n = len(sequence)
    if n < 2:
        return None   # séquence trop courte pour être interpolée

    # Crée des indices régulièrement espacés entre 0 et 1, pour la séquence d'origine et la cible
    indices_originaux = np.linspace(0, 1, n)
    indices_cibles    = np.linspace(0, 1, longueur_cible)

    # Interpole x et y séparément pour obtenir 'longueur_cible' points
    interp_x = interp1d(indices_originaux, sequence[:, 0])
    interp_y = interp1d(indices_originaux, sequence[:, 1])
    return np.column_stack([interp_x(indices_cibles),
                            interp_y(indices_cibles)])

def get_index_tip(hand_landmarks, w, h):
    """Extrait les coordonnées pixel du bout de l'index (landmark 8)"""
    lm = hand_landmarks.landmark[8]
    return (int(lm.x * w), int(lm.y * h))

def reconnaissance():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # ouverture de la webcam (driver Windows)

    if not cap.isOpened():
        print("ERREUR : Webcam non détectée.")
        exit()

    sequence   = []      # trajectoire en cours de capture
    recording  = False   # indique si l'enregistrement est actif
    resultat   = ""       # dernier symbole reconnu
    confidence = 0.0      # score de confiance de la dernière prédiction

    # Crée la fenêtre à l'avance et lui donne une taille fixe (pour bien capter le focus clavier)
    cv2.namedWindow("Air Writing - Reconnaissance", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Writing - Reconnaissance", 800, 600)

    print("\n=== RECONNAISSANCE EN TEMPS RÉEL ===")
    print("ESPACE = démarrer/arrêter | Q = quitter")
    print("⚠️  Cliquez sur la fenêtre vidéo avant d'appuyer sur ESPACE !")

    while True:
        ret, frame = cap.read()          # lecture d'une frame de la webcam
        if not ret or frame is None:
            continue                     # ignore les frames invalides

        frame   = cv2.flip(frame, 1)                       # effet miroir
        h, w, _ = frame.shape                               # dimensions de l'image
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # conversion BGR → RGB pour MediaPipe
        results = hands.process(rgb)                        # détection de la main sur la frame

        # ── Zone d'instruction en haut de l'écran ──────────
        if not recording:
            # Bandeau vert : en attente de démarrage
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 100, 0), -1)
            cv2.putText(frame, "Cliquez ici puis ESPACE pour commencer",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Bandeau rouge : enregistrement en cours
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 180), -1)
            cv2.putText(frame, f"Tracez {len(sequence)} pts | ESPACE pour terminer",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ── Détection de la main et suivi de l'index ───────
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS)  # dessine les 21 landmarks

                if recording:
                    # Récupère et normalise la position du bout de l'index
                    tip    = get_index_tip(hand_lm, w, h)
                    x_norm = tip[0] / w
                    y_norm = tip[1] / h
                    sequence.append([x_norm, y_norm])

                    # Dessine la trajectoire tracée jusqu'à présent (ligne rouge)
                    if len(sequence) > 1:
                        for i in range(1, len(sequence)):
                            p1 = (int(sequence[i-1][0] * w),
                                  int(sequence[i-1][1] * h))
                            p2 = (int(sequence[i][0] * w),
                                  int(sequence[i][1] * h))
                            cv2.line(frame, p1, p2, (0, 0, 255), 3)

        # ── Affichage du résultat de la dernière reconnaissance ──
        if resultat:
            # Vert si confiance suffisante, orange si confiance faible
            couleur = (0, 200, 0) if confidence >= 70 else (0, 165, 255)
            cv2.rectangle(frame, (0, h-130), (w, h), (30, 30, 30), -1)   # bandeau de fond en bas
            cv2.putText(frame, f"Symbole : {resultat}",
                        (20, h-80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, couleur, 3)
            cv2.putText(frame, f"Confiance : {confidence:.1f}%",
                        (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, couleur, 2)

            # Message d'avertissement si la confiance est trop faible
            if confidence < 70:
                cv2.putText(frame, "Confiance faible - Retracez le geste",
                            (20, h-160), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

        cv2.imshow("Air Writing - Reconnaissance", frame)   # affiche la fenêtre vidéo

        # ── Gestion des touches clavier ─────────────────────
        # waitKey(30) au lieu de waitKey(1) pour capturer les touches plus fiablement
        key = cv2.waitKey(30) & 0xFF

        if key == 32:  # 32 = code ASCII de la touche ESPACE
            if not recording:
                # Démarrer un nouvel enregistrement
                recording  = True
                sequence   = []
                resultat   = ""
                confidence = 0.0
                print("→ Enregistrement démarré ! Tracez votre geste...")
            else:
                # Arrêter l'enregistrement et lancer la reconnaissance
                recording = False
                print(f"→ Enregistrement arrêté ({len(sequence)} points)")

                if len(sequence) >= 5:   # ignore les tracés trop courts
                    seq_norm = normaliser_sequence(sequence, SEQUENCE_LENGTH)
                    if seq_norm is not None:
                        seq_input  = np.expand_dims(seq_norm, axis=0)   # ajoute la dimension batch (1, 30, 2)
                        prediction = model.predict(seq_input, verbose=0) # prédiction du modèle LSTM
                        classe_idx = np.argmax(prediction)                # classe avec la plus forte probabilité
                        confidence = prediction[0][classe_idx] * 100      # score de confiance en %
                        resultat   = encoder.classes_[classe_idx]         # symbole correspondant (ex: 'A')
                        print(f"→ Résultat : {resultat} ({confidence:.1f}%)")
                else:
                    print("→ Séquence trop courte, retracez le geste")

        # ── Q : quitter le programme ─────────────────────────
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=== FIN DE LA RECONNAISSANCE ===")

if __name__ == "__main__":
    reconnaissance()