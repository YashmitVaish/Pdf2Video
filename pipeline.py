from pdf_extract import extract_from_pdf
from generate_audio import generate_all_audio
from generate_vedio import render_all_vedio
from frame_generation import generate_scenes
from stitch_vedio_and_clean import stich_and_clean

def pdftovideo(path_to_pdf):
    chunks = extract_from_pdf(path_to_pdf)
    scenes = generate_scenes(chunks)
    generate_all_audio("temp",scenes)
    render_all_vedio(scenes,audio_dir="temp",video_dir="tempv")
    stich_and_clean("tempv","temp")
    
