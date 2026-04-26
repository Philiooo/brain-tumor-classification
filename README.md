# brain-tumor-classification  
Deep-Learning-Modell zur Erkennung von Tumoren in Hirnscans mit Python, Keras und Grad-CAM.  

# 📌 Overview  

Dieses Repository enthält ein Deep-Learning-Projekt zur Klassifikation von Gehirntumoren anhand von MRT-Bildern.  

<br>

### Es kombiniert:  
1. Bildvorverarbeitung (Skalierung, Normalisierung)  
2. CNN-Modellierung mit Keras (Functional API)  
3. Training, Validierung & Testen des Modells  
4. Bildvorhersagen für neue MRT-Scans  
5. Optionale Heatmap-/Highlight-Visualisierung  

Das Ziel ist es, ein einfaches, verständliches und reproduzierbares Modell zu erstellen, das zeigt,  
wie Deep Learning zur medizinischen Bildanalyse eingesetzt werden kann.  

<br>

# 🧼 Datasets & Preprocessing  
### 🧩 1. Image Preprocessing  

### Schritte:  
- Laden und Skalieren der Bilder (150×150 px)  
- Normalisieren auf Wertebereich [0, 1]  
- Erstellen von NumPy-Arrays für Training/Test  
- Train/Test-Split (80/20)  

<br>

# 🧠 CNN Model Architecture  

Das Modell wurde mit der Keras Functional API gebaut:  
- Conv2D (32 Filter) → MaxPooling  
- Conv2D (64 Filter) → MaxPooling  
- Conv2D (128 Filter) → MaxPooling  
- Flatten  
- Dense (256 Neuronen)  
- Dense (1, Sigmoid)  
Loss: Binary Crossentropy  
Optimizer: Adam  
Metrics: Accuracy  

<br>  

# 📈 Training & Evaluation  

Das Modell wird auf den vorbereiteten Daten trainiert:  
- epochs = 20  
- automatische Validierung (20%)  
- Testgenauigkeit nach Trainingsende  
- Plot von Trainings- und Validierungsverlauf (optional)  

<br>  

# 🔍 Prediction & Visualization  

### Das Skript kann:  
✔ Einzelbilder laden  
✔ Tumorwahrscheinlichkeit berechnen  
✔ Ergebnisse ausgeben:  
Wahrscheinlichkeit für Tumor: 0.97  
Das Bild zeigt einen Tumor  
Testgenauigkeit: 0.7647058963775635  

<br>  

# ⚠️ Datenhinweis  

Dieses Projekt dient ausschließlich zu Lernzwecken.  
Keine realen medizinischen Bilddaten werden im Repository gespeichert.  

<br>  

# 🛡 Ethical Notice  

Dieses Projekt dient ausschließlich Bildungszwecken  
und ist nicht für klinische Entscheidungen gedacht.  

