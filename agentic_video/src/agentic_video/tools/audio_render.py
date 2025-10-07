import os
import subprocess
import concurrent.futures
from typing import List, Dict, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class AudioRenderInput(BaseModel):
    scenes: List[Dict] = Field(..., description="List of scene dicts with narration_script and order.")
    folder_name: str = Field(default="outputs/audio", description="Folder to store generated audio files.")


class AudioRenderTool(BaseTool):
    name: str = "audio_renderer"
    description: str = "Generates audio narration for all scenes using Piper TTS."
    args_schema: Type[BaseModel] = AudioRenderInput

    def _generate_audio(self, folder_name: str, scene: Dict):
        os.makedirs(folder_name, exist_ok=True)

        content = scene.get("narration_script", "")
        order = scene.get("order", 0)
        output_file = os.path.join(folder_name, f"scene_{order:03d}.wav")

        if not content.strip():
            print(f"[Scene {order}] Empty narration, skipping.")
            return None

        try:
            command = [
                "piper",
                "-m", "util/models/en_US-lessac-medium.onnx",
                "-f", output_file,
                content
            ]
            print(f"[Scene {order}] Generating audio → {output_file}")
            subprocess.run(command, check=True)
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"[Scene {order}] Piper error: {e}")
            return None
        except Exception as e:
            print(f"[Scene {order}] Unknown error: {e}")
            return None

    def _run(self, scenes: List[Dict], folder_name: str) -> List[Dict]:
        os.makedirs(folder_name, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_scene = {
                executor.submit(self._generate_audio, folder_name, scene): scene for scene in scenes
            }

            for future in concurrent.futures.as_completed(future_to_scene):
                scene = future_to_scene[future]
                order = scene.get("order", 0)
                try:
                    output_path = future.result()
                    scene["audio_path"] = output_path
                    print(f"[Scene {order}] Audio complete.")
                except Exception as e:
                    print(f"[Scene {order}] Thread error: {e}")
                    scene["audio_path"] = None

        return scenes

