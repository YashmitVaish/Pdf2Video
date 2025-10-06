import subprocess
import json
import os
import concurrent.futures



def generate_audio(folder_name, scene):
    os.makedirs(folder_name, exist_ok=True)

    content = scene["narration_script"]
    order = scene["order"]
    output_file = os.path.join(folder_name, f"{folder_name}_{order:03d}.wav")

    command = [
            "piper",
            "-m", "models/en_US-lessac-medium.onnx",
            "-f", output_file, content
        ]

    print(f"Generating: {output_file}")
    subprocess.run(command, check=True)

def generate_all_audio(folder_name,scenes):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for scene in scenes:
            futures.append(executor.submit(generate_audio,folder_name,scene))

        concurrent.futures.wait(futures)


if __name__ == "__main__":
    with open("scenes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        f.close()
    generate_all_audio("output_audio",data)
