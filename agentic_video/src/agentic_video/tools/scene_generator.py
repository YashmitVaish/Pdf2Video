from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import json

load_dotenv(".env")

def extract_json_block(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

class SceneGeneratorInput(BaseModel):
    """Input schema for the scene generator tool."""
    chunks: List[Dict] = Field(..., description="List of extracted PDF chunks containing section_title, content, and tables.")
    
prompt = PromptTemplate.from_template(
    """
        You are an educational content generation assistant that converts textbook or conceptual material into structured video scene data compatible with automated rendering systems such as Pdf2Video or Manim.

        Your goal is to generate detailed JSON objects representing educational video scenes.  
        Each scene must include:
        1. A **clear, friendly narration script** (for voiceover).
        2. A **short, readable screen text or title** (for display).
        3. A **structured elements block** that defines visual components used for rendering.

        ---
        ### OUTPUT RULES

        - You must output **only valid JSON**, no text or explanations.
        - The output must follow this **exact JSON schema**:

        {{
        "scene_title": "string",
        "narration_script": "string",
        "screen_text": "string",
        "elements": [
            {{
            "type": "textbox" | "equation" | "graph",
            "text": "string (for textbox or equation)",
            "title": "string (optional, for graph)",
            "x_label": "string (only for graph)",
            "y_label": "string (only for graph)",
            "points": [[number, number], ...] (only for graph),
            "position": "center" | "top" | "bottom" | "left" | "right",
            "style": {{
                "font_size": number,
                "box": boolean,
                "box_color": "string (e.g. LIGHTBLUE, BLUE, YELLOW, GREEN, RED)"
            }}
            }}
        ],
        "order": number
        }}

        ---

        ### FIELD DEFINITIONS AND RULES

        **scene_title**
        - Short, descriptive title of the concept.
        - ≤ 10 words.

        **narration_script**
        - Clear and conversational.
        - Target audience: middle school to early college students.
        - Simplify complex terms.
        - Merge or rephrase repetitive text.
        - Maintain factual and mathematical correctness.

        **screen_text**
        - Key phrase, heading, or equation.
        - ≤ 20 words.
        - Summarizes the main idea of the narration.

        **elements**
        - Array of one or more renderable components.
        - Each element must be one of:
        - `"textbox"` → Displays short textual content.
        - `"equation"` → Displays formatted math expression.
        - `"graph"` → Displays data as coordinate points with labels.
        - Allowed keys:
        - `"text"` — required for `textbox` or `equation`.
        - `"title"`, `"x_label"`, `"y_label"`, `"points"` — required for `graph`.
        - `"position"` — one of `center`, `top`, `bottom`, `left`, `right`.
        - `"style"` — object defining:
            - `"font_size"` (integer)
            - `"box"` (boolean)
            - `"box_color"` (string; allowed: LIGHTBLUE, BLUE, YELLOW, GREEN, RED)
        - Each element must be syntactically complete and visually meaningful.


        ---

        ### CONTENT CORRECTION RULES

        - Correct any math or physics notation (e.g., format “I_1 / I_2 = V_1 / V_2” properly).
        - Fix or standardize typographical inconsistencies.
        - Remove redundancy in narration.
        - Ensure consistent tone and readability.

        ---

        ### OUTPUT ENFORCEMENT

        - Output **one valid JSON object only**.
        - Do **not** use markdown formatting, code blocks, or natural language commentary.
        - Any non-JSON text will invalidate the output.

        ---

        ### INPUT FORMAT EXAMPLE

        Input:
        Topic Title: Resistance and Its Factors  
        Topic Content: Resistance depends on material, length, cross-sectional area, and temperature. Longer wires or thinner wires increase resistance.  
        Tables (if applicable): None  

        ---

        ### OUTPUT FORMAT EXAMPLE

        {{
        "scene_title": "Factors Affecting Resistance",
        "narration_script": "Resistance depends on a material's type, its length, cross-sectional area, and temperature. Longer or thinner wires increase resistance.",
        "screen_text": "Resistance depends on material, length, area, and temperature",
        "elements": [
            {{
            "type": "textbox",
            "text": "Material, Length, Area, Temperature",
            "position": "center",
            "style": {{
                "font_size": 36,
                "box": true,
                "box_color": "LIGHTBLUE"
            }}
            }}
        ],
        }}

        ---

        ### TASK SUMMARY

        When given:
        - `Topic Title`
        - `Topic Content`
        - (Optional) `Tables`

        You must:
        1. Parse the educational concept.
        2. Generate one complete JSON object strictly matching the schema.
        3. Ensure narration, screen text, and visual elements are coherent and educationally aligned.
        4. Output only valid JSON, fully parseable, with no commentary.

        Here is the input data 
        Topic Title: {section_title} Topic Content: {content} Tables(if applicable) : {tables}
"""
)

class SceneGeneratorTool(BaseTool):
    name: str = "scene_generator"
    description: str = "Generates structured scenes from extracted PDF chunks using ChatGroq LLM."
    args_schema: Type[BaseModel] = SceneGeneratorInput

    def _run(self, chunks: List[Dict]) -> List[Dict]:
        llm = ChatGroq(model = "llama-3.1-8b-instant",temperature=0.3)
        scenes = []

        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("section_title", "")
            text = chunk.get("content", "")
            tables = chunk.get("tables", "")

            query = prompt.format(section_title=title, content=text, tables=tables)

            try:
                response = llm.invoke(query)
                raw_output = response.content.strip()

                start, end = raw_output.find("{"), raw_output.rfind("}")
                if start == -1 or end == -1:
                    print(f"[Scene {i}] Invalid JSON output, skipping.")
                    continue

                scene_data = json.loads(raw_output[start:end+1])
                scene_data["order"] = i
                scenes.append(scene_data)
                print(f"[Scene {i}] Generated successfully.")

            except Exception as e:
                print(f"[Scene {i}] Generation failed: {e}")
                continue

        return scenes


