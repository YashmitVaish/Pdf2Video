import json
from langchain.prompts import PromptTemplate
import ollama
import re

def extract_json_block(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}



with open("cleaned_chunks.json", "r", encoding= "utf-8") as f:
    data = json.load(f)
    f.close()


prompt = PromptTemplate.from_template(
    """
        You are an educational content assistant helping to generate video scenes from textbook material.

        Your task is to create:
        1. A simple, easy-to-understand narration script suitable for voiceover.
        2. A clear and concise visual title or phrase to display on the screen.

        Follow these instructions carefully:
        - Keep the narration friendly, conversational, and suitable for school or college students.
        - Break down complex terms into simpler language.
        - The visual text should be short and readable, like a heading, key phrase, or equation.
        - Do NOT include explanations in the visual text.
        - Output only valid JSON.
        - You can summarize/fix the text if its repititive in nature
        - You are allowed to fix mathamatical expressions
        - Return only the response in JSON with no LLM responses like (sure here is... among others)


        Example Output Format:

        {{
        "scene_title": "Ohm's Law",
        "narration_script": "Ohm's Law explains the relationship between voltage, current, and resistance. When the resistance stays constant, increasing the voltage will increase the current flowing through a circuit.",
        "visual_text": "Ohm's Law: V = I × R"
        }}

        Input:

        Topic Title: {section_title}
        Topic Content: {content}
        Tables(if applicable) : {tables}
    """
)


def generate_scenes(chunks):
    scenes = []
    i = 1
    for chunk in chunks:
        title = chunk["section_title"]
        text = chunk["content"]
        tables = chunk["tables"]

        query = prompt.format(section_title = title, content = text, tables = tables,order = i)
        response = ollama.chat(model="mistral:instruct", messages=[{"role": "user", "content": query}])
        ans = extract_json_block(response["message"]["content"])
        ans["order"] = i
        scenes.append(ans)
        print(i)
        i = i+1
    return scenes
        
scenes = (generate_scenes(data))

with open("scenes.json","w",encoding="utf-8") as f:
    json.dump(scenes,f)
    f.close()