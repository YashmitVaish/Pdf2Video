import json
import os
from moviepy import TextClip, AudioFileClip, CompositeVideoClip, ColorClip

with open("scenes.json", encoding="utf-8") as f:
    scenes = json.load(f)

def render_scene(scene, audio_dir="output_audio", video_dir="video_output"):
    os.makedirs(video_dir, exist_ok=True)

    order = scene["order"]
    visual_text = scene["visual_text"]
    audio_filename = f"{audio_dir}_{order:03d}.wav" 
    audio_path = os.path.join(audio_dir, audio_filename)
    output_video_path = os.path.join(video_dir, f"scene_{order:03d}.mp4")

    audio = AudioFileClip(audio_path)
    duration = audio.duration

    bg = ColorClip(size=(1280, 720), color=(20, 20, 20), duration=duration)

    txt_clip = (
        TextClip(
            text=visual_text,
            font_size=55,
            color="white",
            method="label",
            size=(1100, 600)
        )
        .with_position("center")
        .with_duration(duration)
    )

    video = CompositeVideoClip([bg, txt_clip]).with_audio(audio)

    video.write_videofile(
        output_video_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )

    print(f"done: {output_video_path}")

for scene in scenes:
    render_scene(scene)
    



