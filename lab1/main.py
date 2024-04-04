import os
import pydub
from pydub import AudioSegment
from pathlib import Path
import numpy as np
from sklearn.covariance import EllipticEnvelope

pydub.AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"


# Função para converter arquivo MP3 em array numpy
def mp3_to_np(file_path):
    audio = AudioSegment.from_mp3(file_path)
    audio_data = np.array(audio.get_array_of_samples())
    return audio_data


# Lista para armazenar os arrays numpy de cada arquivo de áudio
audio_arrays = []

# Diretório onde estão os arquivos MP3
input_dir = Path(__file__).parent.joinpath("tmp", "audio_files")

# Loop pelos arquivos no diretório
for file_name in os.listdir(input_dir):
    if file_name.endswith(".mp3"):
        file_path = os.path.join(input_dir, file_name)
        audio_data = mp3_to_np(file_path)
        audio_arrays.append(audio_data)

# Converter a lista de arrays numpy em uma matriz numpy
audio_matrix = np.vstack(audio_arrays)

# Criar e ajustar o modelo EllipticEnvelope
envelope = EllipticEnvelope(
    contamination=0.1
)  # Defina a porcentagem esperada de anomalias
envelope.fit(audio_matrix)

# Loop pelos arquivos novamente para detectar anomalias e filtrar
useful_files = []
for file_name, audio_data in zip(os.listdir(input_dir), audio_arrays):
    if envelope.predict([audio_data]) == 1:  # Se não for uma anomalia
        useful_files.append(file_name)

# Imprimir arquivos úteis
print("Arquivos úteis:", useful_files)
