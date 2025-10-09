from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np

preload_models()
print("Bark models loaded")

def dialogue_to_wav_bark(json_list, output_file="conversation.wav"):
    audio_chunks = []

    for turn in json_list:
        text = turn["content"]

        # Assign speaker based on person
        if turn["speaker"] == "Person 1":
            history_prompt = "v2/en_speaker_7"
        elif turn["speaker"] == "Person 2":
            history_prompt = "v2/en_speaker_6"
        else:
            history_prompt = None  

        audio_array = generate_audio(text, history_prompt=history_prompt)
        audio_chunks.append(audio_array)

    full_audio = np.concatenate(audio_chunks)

    full_audio_int16 = np.int16(full_audio * 32767)

    write_wav(output_file, SAMPLE_RATE, full_audio_int16)
    print(f"Saved combined audio to {output_file}")

import json
with open("all_dia.json","r",encoding = "utf-8") as f:
  json_dialogue = json.load(f)

dialogue_to_wav_bark(json_dialogue)