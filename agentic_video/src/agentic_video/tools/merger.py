from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from moviepy import VideoFileClip, concatenate_videoclips
import os
from typing import Type


class MergerInput(BaseModel):
    """Input schema for video merge agent"""
    video_path: str = Field(..., description="Path to the directory containing video scenes.")
    output_file_name: str = Field("final_video.mp4", description="Name of the merged output file.")


class MergeAllTool(BaseTool):
    name :str = "merge_all_videos"
    description : str = "Merges all rendered scenes into a single video file with transitions."
    args_schema: Type[BaseModel] = MergerInput

    def _run(self, video_path: str, output_file_name: str = "final_video.mp4"):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")

        clips = []
        for content in sorted(os.listdir(video_path)):
            if content.endswith(".mp4"):
                full_path = os.path.join(video_path, content)
                clip = VideoFileClip(full_path)
                clips.append(clip)

        if not clips:
            raise FileNotFoundError("No video files found to merge.")

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_file_name, codec="libx264", audio_codec="aac", fps=24)

        for clip in clips:
            clip.close()

        return {"final_video_path": output_file_name}

if __name__ == "__main__":
    merger = MergeAllTool()
    print(merger._run(video_path="video", output_file_name="final.mp4"))
