import os
import pydub
from pydub import AudioSegment
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest

pydub.AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"


def move_useful_files(file):
    try:
        Path(file).rename(output_dir / Path(file).name)
        chart = input_images_dir / Path(file).name.replace(".mp3", ".png")
        Path(chart).rename(charts_dir / Path(chart).name)
    except FileNotFoundError:
        print(f"Arquivo {file} não encontrado")


def equalize_array_sizes(arrays: tuple[Path, np.ndarray]):
    equalize_array = []
    max_size = max(arr.size for file, arr in arrays)
    for file, arr in arrays:
        arr = np.pad(arr, (0, max_size - arr.size))
        equalize_array.append((file, arr))
    return equalize_array


def get_avarege_duration(input_dir: Path):
    audios = []
    durations = []
    for file in input_dir.iterdir():
        if file.name.endswith(".mp3"):
            audio = AudioSegment.from_mp3(file)
            durations.append(len(audio))
            audios.append((file, audio))

    average_duration = int(np.mean(durations))
    return average_duration / 2, audios


def adjust_audio_arrays(average_duration, audios):
    adjusted_audio_arrays = []
    for file, audio in audios:
        if len(audio) < average_duration:

            last_second = audio[-1000:]  # -1000 milliseconds = -1 second
            while len(audio) < average_duration:
                audio += last_second
        else:
            # Cut the audio to the average duration
            audio = audio[:average_duration]
        audio_data = np.array(audio.get_array_of_samples())
        adjusted_audio_arrays.append((file, audio_data))

    adjusted_audio_arrays = equalize_array_sizes(adjusted_audio_arrays)
    return adjusted_audio_arrays


# Lista para armazenar os arrays numpy de cada arquivo de áudio

input_dir = Path(__file__).parent.joinpath("tmp", "audio_files")
input_images_dir = Path(__file__).parent.joinpath("tmp", "Charts")
output_dir = Path(__file__).parent.joinpath("tmp", "useful_audios")
output_dir.mkdir(parents=True, exist_ok=True)
charts_dir = Path(__file__).parent.joinpath("tmp", "useful_charts")
charts_dir.mkdir(parents=True, exist_ok=True)

avrg_duration, audios = get_avarege_duration(input_dir)

adjusted_audio_arrays = adjust_audio_arrays(avrg_duration, audios)

audio_matrix = np.vstack([array for file, array in adjusted_audio_arrays])

envelope = IsolationForest(contamination=0.25)
envelope.fit(audio_matrix)


useful_files = []
for file, audio_data in adjusted_audio_arrays:
    if envelope.predict([audio_data]) == 1:
        useful_files.append(file)
        move_useful_files(file)

# Imprimir arquivos úteis
print("Numero de Arquivos úteis:", len(useful_files))
