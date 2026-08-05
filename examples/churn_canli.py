"""Egitim sirasinda canli izleme ornegi (musteri kaybi / churn modeli).

Calistirmak icin:
    PYTHONPATH=src python examples/churn_canli.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np

import nn3d

FEATURES = [
    "Tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod",
    "InternetService", "OnlineSecurity", "TechSupport", "PaperlessBilling",
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "StreamingTV", "StreamingMovies",
]

rng = np.random.default_rng(0)
X = rng.normal(size=(2000, len(FEATURES))).astype("float32")
# Ogrenilebilir bir sinyal koyalim ki metrikler gercekten iyilessin.
y = (X[:, 0] * 1.3 - X[:, 3] * 0.9 + X[:, 7] * 0.6 > 0).astype("float32")

model = keras.Sequential([
    keras.layers.Input(shape=(len(FEATURES),)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
], name="churn")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

monitor = nn3d.Monitor(
    X[:1],
    every=5,
    input_labels=FEATURES,
    output_labels=["Kayip Olasiligi"],
)
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2,
          callbacks=[monitor], verbose=2)

nn3d.wait()   # tarayici sekmesi acik kalsin
