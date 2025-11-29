import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


# Pfade zu den Bildern
no = "C:/Users/peich/Documents/TH-Köln/Projekt/Hinrtumor - LLM/Gehirntumor/no"
yes = "C:/Users/peich/Documents/TH-Köln/Projekt/Hinrtumor - LLM/Gehirntumor/yes"

images = [] #leere listen für die Bilddaten und die dazugehörende Labels
labels = []

# Tumor-Bilder
for img in os.listdir(yes): #Liste aller Dateien im Ordner
    img_path = os.path.join(yes, img) #Baut den vollständigen Pfad zum Bild
    img_array = load_img(img_path, target_size=(150, 150)) #Bild auf 150x150 Pixel skalieren
    img_array = img_to_array(img_array)/255.0 #Bild in NumPy-Array umwandeln
    images.append(img_array) #Bild speichern
    labels.append(1) #Label Tumor hinzufügen

# Kein-Tumor-Bilder
for img in os.listdir(no):
    img_path = os.path.join(no, img)
    img_array = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img_array)/255.0
    images.append(img_array)
    labels.append(0) #Label ohne Tumor hinzufügen

#Wandelt die Listen in NumPy-Arrays um, damit sie vom Keras-Modell verarbeitet werden können.
X = np.array(images)
y = np.array(labels)

# Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #
print('Trainingsdaten:', X_train.shape)
print('Testdaten:', X_test.shape)

# Functional API Modell
inputs = layers.Input(shape=(150, 150, 3)) #Eingangsschicht für Bilder 150x150 Pixel mit 3 Farbkanälen (RGB)
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs) #Erste Convolutional Layer: 32 Filter, 3x3 Kernel, ReLU-Aktivierung
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x) #Zweite Convolutional Layer mit 64 Filtern
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_2")(x) #Dritte Convolutional Layer, 128 Filter, Name für Grad-CAM
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x) #2D Merkmalskarten → 1D-Vektor
x = layers.Dense(256, activation='relu')(x) #Dense Layer (vollständig verbunden) mit 256 Neuronen
outputs = layers.Dense(1, activation='sigmoid')(x) #gibt Wahrscheinlichkeit für Tumor (0–1)

model = Model(inputs, outputs) #Modell zusammenbauen

#Kompilieren des Modells
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    X_train, y_train, #Trainingsdaten
    validation_split=0.2, #20% Trainingsdaten für Validierung
    shuffle=True, #Daten vor jedem Epoch mischen
    epochs=20 #20 Durchläufe
)

# Grad-CAM vorbereiten
last_conv_layer = model.get_layer("conv2d_2") #Wählt die letzte Convolutional Layer für Grad-CAM
grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output]) #Erstellt ein neues Modell, das Layer-Ausgabe + Vorhersage liefert

# Bild für Grad-CAM
img_path = "C:/Users\peich\Documents\TH-Köln\Projekt\Hinrtumor - LLM\Gehirntumor/yes/Y1.jpg" #Lädt Testbild, normalisiert, fügt Batch-Dimension hinzu ((1,150,150,3))
img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)
x_uint8 = np.uint8(x[0] * 255)
gray = cv2.cvtColor(x_uint8, cv2.COLOR_RGB2GRAY)

# Schwellenwert für Helligkeit (anpassen je nach Bild)
_, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Optionale Glättung
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Farbmarkierung: Tumorbereich wird rot
highlight = x_uint8.copy()
highlight[mask > 0] = [255, 0, 0]  # Rot in RGB

# Transparente Überlagerung
overlay = cv2.addWeighted(x_uint8, 0.6, highlight, 0.4, 0)

# Anzeige
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# Vorhersage für das Bild
prediction = model.predict(x)  #Modell sagt Wahrscheinlichkeit für Tumor
pred_value = prediction[0][0]

print("Hallo")

# Ausgabe
print(f"Wahrscheinlichkeit für Tumor: {pred_value:.2f}")
if pred_value > 0.5:
    print("Das Bild zeigt einen Tumor")
else:
    print("Das Bild zeigt keinen Tumor")


# Testgenauigkeit
test_loss, test_acc = model.evaluate(X_test, y_test) #Bewertet das Modell auf ungesehenen Testdaten
print('Testgenauigkeit:', test_acc)
