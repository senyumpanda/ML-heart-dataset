# Import Modules
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("heart.csv")

# Karena pengklasifikasi KNN memprediksi kelas pengamatan tes yang diberikan dengan mengidentifikasi
# pengamatan yang paling dekat dengannya, skala variabel penting. Variabel apa pun yang berada dalam skala
# besar akan memiliki efek yang jauh lebih besar pada jarak antara pengamatan, dan karenanya pada
# pengklasifikasi KNN, daripada variabel yang berada pada skala kecil.

# Preprocessing Dataset
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(X)

# Training dan Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Model -> KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Melakukan Prediksi dan mendapat Hasilnya
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)
print(f"Hasil prediksi : {akurasi*100:.2f}%")