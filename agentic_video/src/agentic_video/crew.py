from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from tools.pdf_extractor import PDFExtractorTool
from tools.scene_generator import SceneGeneratorTool
from tools.audio_render import AudioRenderTool
from tools.video_render import RenderVideoTool
from tools.merger import MergeAllTool

import time
import threading
from litellm import completion

class RateLimitedLLM:
    def __init__(self, model: str, tpm_limit: int = 6000, buffer: int = 500, refill_rate: int = 60):
        """
        model: model name e.g. 'groq/llama-3.1-8b-instant'
        tpm_limit: token per minute cap (from provider)
        buffer: safety margin before hitting the limit
        refill_rate: seconds to reset (usually 60)
        """
        self.model = model
        self.tpm_limit = tpm_limit - buffer
        self.refill_rate = refill_rate
        self.tokens_used = 0
        self.lock = threading.Lock()
        self.last_reset = time.time()

    def _refill(self):
        if time.time() - self.last_reset > self.refill_rate:
            self.tokens_used = 0
            self.last_reset = time.time()

    def _wait_if_needed(self, tokens):
        with self.lock:
            self._refill()
            if self.tokens_used + tokens > self.tpm_limit:
                wait_time = self.refill_rate - (time.time() - self.last_reset)
                print(f"Rate limit near! Waiting {wait_time:.2f}s...")
                time.sleep(max(0, wait_time))
                self._refill()
            self.tokens_used += tokens

    def __call__(self, messages, **kwargs):
        # estimate tokens (roughly)
        tokens = sum(len(m["content"].split()) for m in messages) * 1.3
        self._wait_if_needed(int(tokens))
        return completion(model=self.model, messages=messages, **kwargs)

rate_limited_llm = RateLimitedLLM("groq/llama-3.1-8b-instant")


@CrewBase
class AgenticVideo:
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.extractor_agent(),
                self.scene_agent(),
                self.audio_agent(),
                self.render_agent(),
                self.merge_agent(),
            ],
            tasks=[
                self.extract_task(),
                self.generate_scenes_task(),
                self.audio_task(),
                self.render_task(),
                self.merge_task(),
            ],
            process=Process.sequential,
            verbose=True,
            chat_llm= rate_limited_llm,
            planning_llm=rate_limited_llm,
            manager_llm=rate_limited_llm,
            function_calling_llm= rate_limited_llm,
    )


    agents: List[BaseAgent]
    tasks: List[Task]

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

    @task
    def extract_task(self) -> Task:
        return Task(
            description="Extract scenes from a PDF file.",
            agent=self.extractor_agent(),
            expected_output="chunks",
            output_file="intermediate/scenes.json",
            inputs={"pdf_path": "{{pdf_path}}"}
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
            expected_output="populated dir",
        )

    @task
    def merge_task(self) -> Task:
        return Task(
            description="Merge all scene videos",
            agent=self.merge_agent(),
            expected_output="final_video.mp4",
        )


if __name__ == "__main__":
    AgenticVideo().crew().kickoff(inputs={"pdf_path": "testpdf-6-9.pdf"})
    print("Pipeline finished. Result:")
