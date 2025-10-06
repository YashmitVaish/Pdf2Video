from moviepy import *
from moviepy.video.tools.drawing import color_gradient
import textwrap
import json 
import os 
import concurrent.futures

def render_scene(scene,audio_dir ,video_dir , resolution = (1280,720)):
    background_path = "blackboard.png"
    
    os.makedirs(video_dir,exist_ok=True)

    order = scene["order"]

    title = scene["scene_title"]
    explanation = scene["screen_text"]
    audio_filename = f"{audio_dir}_{order:03d}.wav"
    audio_path = os.path.join(audio_dir,audio_filename)
    output_video_path = os.path.join(video_dir, f"scene_{order:03d}.mp4")
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    background = ImageClip(background_path).resized(resolution).with_duration(duration)

    title_txt = TextClip(
            text=title,
            font_size=60,
            color='white',
            font="dejavu-fonts-ttf-2.37/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
            method="caption",
            size=(resolution[0] - 100, 200)
        ).with_position(("center", 40)).with_duration(duration)

    wrapped = textwrap.fill(explanation, width=70)
    body_txt = TextClip(
            text=wrapped,
            font_size=40,
            color='white',
            font="dejavu-fonts-ttf-2.37/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
            method="caption",
            size=(resolution[0]-100 ,600)
        ).with_position(("center", "center")).with_duration(duration)
        
    video = CompositeVideoClip([background, title_txt,body_txt]).with_audio(audio)
    video.write_videofile(
            output_video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )

    print(f"done: {output_video_path}")

    


def render_all_vedio(scenes,audio_dir ,video_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for scene in scenes:
            futures.append(executor.submit(render_scene,scene,audio_dir,video_dir))
        concurrent.futures.wait(futures)




