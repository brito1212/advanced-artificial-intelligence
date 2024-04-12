import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical

# Carregar dados de treinamento
train_audio_path = "path_to_your_audio_files"
labels = os.listdir(train_audio_path)

# Converter áudio para vetores de recursos
train = []
for label in labels:
    waves = [
        f for f in os.listdir(train_audio_path + "/" + label) if f.endswith(".wav")
    ]
    for wav in waves:
        samples, sample_rate = librosa.load(
            train_audio_path + "/" + label + "/" + wav, sr=16000
        )
        train.append(samples)

# Codificar rótulos
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y, num_classes=len(labels))

# Reshape para 2D
train = np.array(train).reshape(-1, 8000, 1)

# Criar modelo
model = Sequential()
model.add(
    Conv1D(8, 13, padding="valid", activation="relu", strides=1, input_shape=(8000, 1))
)
model.add(MaxPooling1D(3))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(len(labels), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinar modelo
model.fit(train, y, epochs=30, batch_size=100)

# Salvar modelo
model.save("path_to_save_your_pretrained_model.h5")
