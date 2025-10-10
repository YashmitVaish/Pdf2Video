from audio_render import AudioRenderTool
from merger import MergeAllTool
from pdf_extractor import PDFExtractorTool
from scene_generator import SceneGeneratorTool
from video_render import RenderVideoTool

audio_tool = AudioRenderTool()
merge_tool = MergeAllTool()
pdf_tool = PDFExtractorTool()
scene_tool = SceneGeneratorTool()
video_tool = RenderVideoTool()

def pipeline(pdf_path : str, audio_path : str, video_path: str, final_name : str):

    chunks = pdf_tool._run(pdf_path)

    scenes = scene_tool._run(chunks)

    audio_tool._run(scenes,audio_path)

    video_tool._run(scenes,audio_path,video_path)

    merge_tool._run(video_path,final_name)


pipeline("Newtons_Laws_of_Motion.pdf","tes_aud","tes_vid","final.mp4")

