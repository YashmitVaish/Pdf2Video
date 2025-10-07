from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from tools.pdf_extractor import PDFExtractorTool
from tools.scene_generator import SceneGeneratorTool
from tools.audio_render import AudioRenderTool
from tools.video_render import RenderVideoTool
from tools.merger import MergeAllTool


@CrewBase
class AgenticVideo:

    agents: List[BaseAgent]
    tasks: List[Task]

    # ---- Agents ----
    @agent
    def extractor_agent(self) -> Agent:
        return Agent(
            role="PDF Extractor",
            goal="Extract structured scene chunks from a PDF.",
            backstory="Parses PDFs and outputs ordered text chunks and metadata.",
            tools=[PDFExtractorTool()],
            verbose=True,
        )

    @agent
    def scene_agent(self) -> Agent:
        return Agent(
            role="Scene Generator",
            goal="Convert text chunks into structured scene JSONs.",
            backstory="Produces slide_title, narration_script, elements and order for each scene.",
            tools=[SceneGeneratorTool()],
            verbose=True,
        )

    @agent
    def audio_agent(self) -> Agent:
        return Agent(
            role="Audio Renderer",
            goal="Generate narration audio files from scene narration scripts.",
            backstory="Uses Piper (or configured TTS) to produce .wav files per scene.",
            tools=[AudioRenderTool()],
            verbose=True,
        )

    @agent
    def render_agent(self) -> Agent:
        return Agent(
            role="Video Renderer",
            goal="Render scenes to MP4 using Manim and sync audio.",
            backstory="Produces per-scene MP4 files based on scene JSON and audio.",
            tools=[RenderVideoTool()],
            verbose=True,
        )

    @agent
    def merge_agent(self) -> Agent:
        return Agent(
            role="Video Merger",
            goal="Concatenate per-scene MP4s into a single final video.",
            backstory="Final assembly and optional cleanup of intermediate files.",
            tools=[MergeAllTool()],
            verbose=True,
        )

    # ---- Tasks ----
    @task
    def extract_task(self) -> Task:
        return Task(
            description="Extract scenes from PDF",
            agent=self.extractor_agent(),          # <- Agent instance, not a string
            expected_output="chunks",
            output_file="intermediate/scenes.json",
        )

    @task
    def generate_scenes_task(self) -> Task:
        return Task(
            description="Generate scene JSON from chunks",
            agent=self.scene_agent(),
            expected_output="a list of scenes",
        )

    @task
    def audio_task(self) -> Task:
        return Task(
            description="Render audio for each scene",
            agent=self.audio_agent(),
            expected_output="populated dir",
        )

    @task
    def render_task(self) -> Task:
        return Task(
            description="Render videos for each scene",
            agent=self.render_agent(),
            expected_output="video_output/",
        )

    @task
    def merge_task(self) -> Task:
        return Task(
            description="Merge all scene videos",
            agent=self.merge_agent(),
            expected_output="final_video.mp4",
        )


if __name__ == "__main__":
    # Example run. Replace paths in the task/tool implementations or pass inputs to kickoff.
    crew = AgenticVideo().crew()  # type: ignore[attr-defined]
    # Pass inputs the tools expect via kickoff inputs (adjust keys to your tools)
    kickoff_inputs = {
        "pdf_path": "testpdf-6-9.pdf"
    }
    result = crew.kickoff(inputs=kickoff_inputs)
    print("Pipeline finished. Result:", result)
