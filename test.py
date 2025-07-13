import os 
audio_path = "output_audio"
contents = os.listdir(audio_path)
for content in contents:
        full_path = os.path.join(audio_path,content)
        os.remove(full_path)

os.rmdir(audio_path)