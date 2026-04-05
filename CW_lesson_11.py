<<<<<<< HEAD
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("data/figures.csv")

encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"]) 

X = df[["area", "perimeter", "corners"]]
y = df["label_enc"]

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

history = model.fit(X, y, epochs=10000, verbose=0)

test = np.array([[16, 16, 0]])
pred = model.predict(test)


predicted_class = encoder.inverse_transform([np.argmax(pred)])
print(f"Prediction probabilities: {pred}")
print(f"Predicted class: {predicted_class[0]}")

plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["accuracy"], label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Model Learning")
plt.legend()
plt.show()

=======
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("data/figures.csv")

encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"]) 

X = df[["area", "perimeter", "corners"]]
y = df["label_enc"]

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

history = model.fit(X, y, epochs=10000, verbose=0)

test = np.array([[16, 16, 0]])
pred = model.predict(test)


predicted_class = encoder.inverse_transform([np.argmax(pred)])
print(f"Prediction probabilities: {pred}")
print(f"Predicted class: {predicted_class[0]}")

plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["accuracy"], label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Model Learning")
plt.legend()
plt.show()

>>>>>>> 4dbd617551f81dec361b34588c4f3b75ca6513a9
