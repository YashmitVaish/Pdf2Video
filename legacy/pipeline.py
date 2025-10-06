from pdf_extract import process_pdf
from generate_audio import generate_all_audio
from generate_vedio import render_all_vedio
from frame_generation import generate_scenes
from stitch_vedio_and_clean import stich_and_clean

def pdftovideo(path_to_pdf):
    print("processing start")
    scenes = generate_scenes(process_pdf(path_to_pdf))
    print("scenes done")
    generate_all_audio("temp",scenes)
    print("audio made")
    render_all_vedio(scenes,audio_dir="temp",video_dir="tempv")
    print("videos made")
    stich_and_clean("tempv","temp")
    print("done and dusted")
    
if __name__ == "__main__":
    pdftovideo("testpdf.pdf")

