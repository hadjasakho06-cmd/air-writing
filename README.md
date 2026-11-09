# ✍️ Air Writing — Reconnaissance de gestes d'écriture dans l'air

Système de **vision par ordinateur** qui reconnaît, en temps réel et avec une simple webcam, les **lettres (A–Z) et les chiffres (0–9)** tracés dans l'air avec le doigt.

> Projet universitaire réalisé dans le cadre du Master Intelligence Artificielle et Technologies Intelligentes (Université Iba Der Thiam de Thiès).

## 🎯 Principe

L'utilisateur trace un caractère dans l'air avec son index devant la caméra. Le système suit la position du bout du doigt, puis un réseau de neurones récurrent (LSTM) classe la trajectoire parmi **36 classes**.

```
Webcam ──► MediaPipe Hands ──► Trajectoire (x, y) de l'index
                                      │
                         Normalisation à 30 points
                                      │
                               Réseau LSTM ──► Symbole reconnu + score de confiance
```

## 🧠 Méthodologie

1. **Collecte** (`collect_data.py`) : enregistrement de la trajectoire du bout de l'index (landmark n°8 de MediaPipe Hands), 30 échantillons par classe, sauvegardés au format JSON.
2. **Prétraitement** : normalisation de chaque trajectoire à une longueur fixe de 30 points par interpolation linéaire.
3. **Augmentation des données** : bruit gaussien, mise à l'échelle, translation et légère rotation, soit un jeu de données multiplié par 6 (1 078 → 6 468 séquences).
4. **Modèle** (LSTM) :
   - LSTM (64 unités) → Dropout (0,2)
   - LSTM (128 unités) → Dropout (0,2)
   - Dense (64, ReLU) → Dense (36, Softmax)
   - 126 564 paramètres ; optimiseur Adam ; perte *categorical crossentropy* ; arrêt anticipé (*early stopping*).
5. **Reconnaissance en temps réel** (`recognize.py`) : affichage du symbole prédit et de son score de confiance, avec un avertissement si la confiance est inférieure à 70 %.

## 📊 Résultats

| Élément | Valeur |
|---|---|
| Classes | 36 (26 lettres + 10 chiffres) |
| Séquences d'origine | 1 078 |
| Séquences après augmentation | 6 468 |
| Répartition entraînement / test | 80 % / 20 % |
| Exactitude sur le jeu de test | **99,85 %** |

<p align="center">
  <img src="images/courbes_apprentissage.png" width="48%" alt="Courbes d'apprentissage">
  <img src="images/matrice_confusion.png" width="48%" alt="Matrice de confusion">
</p>

## 🎬 Démonstration

<p align="center">
  <img src="images/demo.png" width="70%" alt="Démonstration de la reconnaissance en temps réel">
</p>

## 📁 Structure du projet

```
air-writing/
├── collect_data.py          # Collecte des trajectoires via la webcam
├── recognize.py             # Reconnaissance en temps réel
├── train_model.ipynb        # Entraînement et évaluation du modèle (Google Colab)
├── model_airwriting.h5      # Modèle LSTM entraîné
├── label_encoder.pkl        # Encodeur des classes (indice → symbole)
├── dataset2.zip             # Jeu de données (trajectoires JSON)
├── images/                  # Captures et graphiques
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/hadjasakho06-cmd/air-writing.git
cd air-writing
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS : source venv/bin/activate)
pip install -r requirements.txt
```

## ▶️ Utilisation

**1. Reconnaître des gestes (modèle déjà entraîné)**

```bash
python recognize.py
```

Cliquez sur la fenêtre vidéo, puis **ESPACE** pour démarrer le tracé, **ESPACE** à nouveau pour terminer et obtenir la prédiction, **Q** pour quitter.

**2. Collecter votre propre jeu de données**

```bash
python collect_data.py
```

Les trajectoires sont enregistrées dans le dossier `dataset2/`.

**3. Réentraîner le modèle**

Décompressez `dataset2.zip`, puis ouvrez `train_model.ipynb` (conçu pour Google Colab) et adaptez la variable `DATASET_PATH` à l'emplacement de vos données.

## ⚠️ Limites et pistes d'amélioration

- **Évaluation** : l'augmentation de données est appliquée *avant* la séparation entraînement/test. Des versions modifiées d'une même trajectoire peuvent donc se retrouver des deux côtés, ce qui tend à surestimer l'exactitude. Une évaluation plus rigoureuse consiste à séparer les trajectoires d'origine avant d'augmenter, idéalement avec des tracés de personnes absentes de l'entraînement.
- **Données** : 30 échantillons par classe, un jeu de données modeste.
- **Entrée** : seule la trajectoire (x, y) de l'index est utilisée ; la forme complète de la main ne l'est pas.
- **Perspectives** : reconnaissance de mots, élargissement du jeu de données, adaptation à la reconnaissance de signes.

## 🛠️ Technologies

Python · OpenCV · MediaPipe Hands · TensorFlow / Keras (LSTM) · scikit-learn · NumPy · SciPy · Matplotlib

## 👩‍💻 Auteure

**Racky Sakho SOW** — Étudiante en Master 2 Intelligence Artificielle et Technologies Intelligentes
