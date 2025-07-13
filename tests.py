from moviepy import *
from moviepy.video.tools.drawing import color_gradient
import textwrap

def render_scene_blackboard(scene, background_path="blackboard.png", duration=6, resolution=(1280, 720)):
    title = scene["scene_title"]
    explanation = scene["visual_text"]
    output_file = f"scene_{scene['order']:03d}.mp4"

    background = ImageClip(background_path).resized(resolution).with_duration(duration)

    # Title styling
    title_txt = TextClip(
        text=title,
        font_size=60,
        color='white',
        font="dejavu-fonts-ttf-2.37/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
        method="caption",
        size=(resolution[0] - 100, None)
    ).with_position(("center", 100)).with_duration(duration)

    # Explanation styling
    wrapped = textwrap.fill(explanation, width=80)
    body_txt = TextClip(
        text=wrapped,
        font_size=40,
        color='white',
        font="dejavu-fonts-ttf-2.37/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
        method="caption",
        size=(resolution[0] - 100, 400)
    ).with_position(("center", "center")).with_duration(duration)

    # Combine
    video = CompositeVideoClip([background, title_txt, body_txt])
    video.write_videofile(output_file, fps=24)

    return output_file


sample_scene = {
    "scene_title": "Understanding Electricity",
    "visual_text": "Electricity is a controllable form of energy used widely in homes, schools, and industries.",
    "order": 1
}

render_scene_blackboard(sample_scene)
