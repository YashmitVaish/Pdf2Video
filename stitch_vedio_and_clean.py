from moviepy import VideoFileClip, concatenate_videoclips
import os

def stich_and_clean(videos_path:str,output_file = "final_vedio.mp4"):
    contents = os.listdir(videos_path)
    clips = []

    for content in contents:
        if(content.endswith(".mp4")):
            full_path = os.path.join(videos_path,content)
            clip = VideoFileClip(full_path)
            clips.append(clip)

    if not clips:
        raise FileNotFoundError
        return
    
    final = concatenate_videoclips(clips,"compose")
    final.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=24)

    print("done")
    for content in contents:
        full_path = os.path.join(videos_path,content)
        os.remove(full_path)

    os.rmdir(videos_path)

stich_and_clean("video_output")


