import subprocess
import json
import os

# Load scenes
with open("scenes.json", "r", encoding="utf-8") as f:
    scenes = json.load(f)

def generate_all_audio(folder_name, scenes):
    os.makedirs(folder_name, exist_ok=True)

    for scene in scenes:
        content = scene["narration_script"]
        order = scene["order"]
        output_file = os.path.join(folder_name, f"{folder_name}_{order:03d}.wav")

        # Build command
        command = [
            "piper",
            "-m", "models/en_US-lessac-medium.onnx",
            "-f", output_file, content
        ]

        print(f"Generating: {output_file}")
        subprocess.run(command, check=True)

generate_all_audio("output_audio", scenes)
