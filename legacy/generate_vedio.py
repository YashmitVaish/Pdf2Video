from manim import *
from manim.utils.color import ManimColor
import json, os, subprocess, textwrap
from typing import Dict


with open("test_json_scene.json","r",encoding="utf-8") as f:

    SCENES_JSON = json.load(f)


def safe_color(value, fallback=BLUE):
    """Safely parse a color name or hex."""
    if not value:
        return fallback
    try:
        return ManimColor(value)
    except Exception:
        try:
            return ManimColor(value.upper())
        except Exception:
            return fallback


def safe_top(mob):
    """Return the top Y coordinate safely."""
    try:
        return mob.get_top()[1]
    except Exception:
        return 0


def safe_bottom(mob):
    """Return the bottom Y coordinate safely."""
    try:
        return mob.get_bottom()[1]
    except Exception:
        return 0


def wrap_and_fit_text(text: str, font_size: int, max_width: float, max_height: float):
    """Wrap long text and scale to fit within frame limits."""
    chars_per_line = max(int((max_width * 10) / max((font_size / 2), 1)), 20)
    wrapped = textwrap.fill(text, width=chars_per_line)
    txt = Text(wrapped, font_size=font_size)
    scale_w = (max_width / txt.width) if txt.width > 0 else 1.0
    scale_h = ((max_height - 0.1) / txt.height) if txt.height > 0 else 1.0
    scale_factor = min(1.0, scale_w, scale_h)
    if scale_factor < 1.0:
        txt.scale(scale_factor)
    return txt


def create_graph_element(elem: Dict):
    """Create and return an Axes-based graph."""
    points = elem.get("points", [])
    if not points:
        return Text("No data", font_size=24)

    xs, ys = zip(*points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    x_step = (x_max - x_min) / 5 or 1
    y_step = (y_max - y_min) / 5 or 1

    ax = Axes(
        x_range=[x_min, x_max, x_step],
        y_range=[y_min, y_max, y_step],
        x_length=6,
        y_length=3,
        axis_config={"color": WHITE},
    )

    graph = ax.plot_line_graph(xs, ys, line_color=YELLOW, add_vertex_dots=True)
    xlab = Text(elem.get("x_label", ""), font_size=20).next_to(ax.x_axis, DOWN)
    ylab = Text(elem.get("y_label", ""), font_size=20).next_to(ax.y_axis, LEFT)
    title_color = safe_color(elem.get("style", {}).get("title_color", "#7ED8F7"))
    title = Text(elem.get("title", ""), font_size=22, color=title_color).next_to(ax, UP)
    return VGroup(ax, graph, xlab, ylab, title)


def create_textbox(elem: Dict):
    """Create a styled textbox based on JSON definition."""
    style = elem.get("style", {})
    font_size = style.get("font_size", 30)
    box = style.get("box", False)
    box_color = safe_color(style.get("box_color", "BLUE"), fallback=BLUE)

    top_margin = 1.5
    bottom_margin = 1.0
    max_w = config.frame_width - 1.0
    max_h = config.frame_height - top_margin - bottom_margin

    txt = wrap_and_fit_text(elem.get("text", ""), font_size, max_w, max_h)
    if box:
        rect = SurroundingRectangle(txt, color=box_color, buff=0.3, stroke_width=3)
        group = VGroup(rect, txt)
    else:
        group = txt

    pos = elem.get("position", "center")
    if pos == "center":
        group.move_to(ORIGIN)
    elif pos == "bottom":
        group.to_edge(DOWN, buff=0.8)
        if safe_top(group) > config.frame_height / 4:
            group.shift(UP * 0.5)
    elif pos == "top":
        group.to_edge(UP, buff=1.2)
    elif pos == "left":
        group.to_edge(LEFT, buff=0.8)
    elif pos == "right":
        group.to_edge(RIGHT, buff=0.8)
    else:
        group.move_to(ORIGIN)

    # Scale safety check
    if group.width > config.frame_width - 0.5:
        group.scale((config.frame_width - 0.5) / group.width)
    if group.height > config.frame_height - 1.0:
        group.scale((config.frame_height - 1.0) / group.height)

    return group


# ---------- Base JSON-driven Scene ----------
class JSONScene(Scene):
    def __init__(self, scene_data: Dict, audio_dir: str = "test_audio_output", **kwargs):
        self.scene_data = scene_data
        self.audio_dir = audio_dir
        super().__init__(**kwargs)

    def construct(self):
        order = self.scene_data.get("order", 0)
        title_text = self.scene_data.get("scene_title", "")
        elements = self.scene_data.get("elements", [])
        audio_path = os.path.join(self.audio_dir, f"scene_{order:03d}.wav")

        title = Text(title_text, font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(title), run_time=0.9)
        self.wait(0.25)


        # Screen text (explanation from JSON)
        screen_text_str = self.scene_data.get("screen_text", "")
        if screen_text_str:
            screen_txt = wrap_and_fit_text(screen_text_str, font_size=32,
                                        max_width=config.frame_width - 1,
                                        max_height=2)
            screen_txt.next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(screen_txt, shift=UP*0.3), run_time=0.8)
            self.wait(0.5)


        if os.path.exists(audio_path):
            try:
                self.add_sound(audio_path)
            except Exception as e:
                print("Warning: couldn't play audio:", e)

        for elem in elements:
            etype = elem.get("type", "textbox")
            if etype == "textbox":
                mob = create_textbox(elem)
                self.play(FadeIn(mob, shift=UP * 0.4), run_time=0.9)
                self.wait(0.8)
            elif etype == "graph":
                mob = create_graph_element(elem)
                if elem.get("position", "") == "bottom":
                    mob.next_to(ORIGIN, DOWN, buff=0.6)
                else:
                    mob.move_to(ORIGIN)
                # ensure it fits
                if mob.width > config.frame_width - 0.5:
                    mob.scale((config.frame_width - 0.5) / mob.width)
                if mob.height > config.frame_height - 2.0:
                    mob.scale((config.frame_height - 2.0) / mob.height)
                self.play(Create(mob), run_time=1.6)
                self.wait(1.0)
            else:
                fallback = create_textbox({
                    "text": str(elem),
                    "style": {"font_size": 24, "box": False},
                    "position": "center"
                })
                self.play(FadeIn(fallback), run_time=0.8)
                self.wait(0.8)

        if os.path.exists(audio_path):
            from pydub import AudioSegment
            audio_length = AudioSegment.from_file(audio_path).duration_seconds
            # Subtract the time already waited for animations
            current_time = self.renderer.time
            remaining_time = max(0, audio_length - current_time)
            self.wait(remaining_time)
        else:
            self.wait(1)

        # Now fade out gracefully after narration ends
        self.play(FadeOut(*self.mobjects), run_time=1.5)
        self.wait(0.5)


# ---------- Scene Factory ----------
def make_scene_class(scene_data):
    class_name = f"Scene_{scene_data.get('order', 0):03d}"
    def __init__(self, **kwargs):
        JSONScene.__init__(self, scene_data, **kwargs)
    return type(class_name, (JSONScene,), {"__init__": __init__})


for sc in SCENES_JSON:
    cname = f"Scene_{sc.get('order', 0):03d}"
    globals()[cname] = make_scene_class(sc)


# ---------- Optional CLI Automation ----------
MANIM_FLAGS = ["-pqh"]  # preview, low quality
def render_all_via_manim_cli():
    this_file = os.path.abspath(__file__)
    for sc in SCENES_JSON:
        scene_name = f"Scene_{sc.get('order', 0):03d}"
        print(f"Rendering {scene_name} ...")
        cmd = ["manim", *MANIM_FLAGS, this_file, scene_name]
        subprocess.run(cmd, check=False)

if __name__ == "__main__":
    render_all_via_manim_cli()