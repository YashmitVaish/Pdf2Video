from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from concurrent.futures import ThreadPoolExecutor
from rouge_score import rouge_scorer

from elasticsearch import Elasticsearch
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv
import asyncio
import httpx
import random
import json
import hashlib
import logging
import string
import math
import uuid
import re
import os

# =========== Configurations ==========#

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()
logger = logging.getLogger("uvicorn")
executor = ThreadPoolExecutor(max_workers=5)

load_dotenv(".env")
openai_api_key = os.getenv("GROQ_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
print(openai_api_key)
print(neo4j_uri)
print(neo4j_username)
print(neo4j_password)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,

    check_every_n_seconds=0.1,

    max_bucket_size=30
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, rate_limiter=rate_limiter)

parser = JsonOutputParser()

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "changeme"),
    verify_certs=False  
)

INDEX_NAME = "semantic_vectors"
VECTOR_DIM = 384


if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    )

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

graph = Neo4jGraph(
    url=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password,
    refresh_schema=False,
)

#================Helper=================#

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def ocr_page(page):
    """Run Tesseract OCR (with math support) on a PyMuPDF page."""
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang="eng+equ")

def extract_pdf_text(path):
    text = ""
    try:
        doc = fitz.open(path)
        fallback_pages = []

        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                fallback_pages.append(i)

        if not text.strip():
            raise ValueError("No text found with PyMuPDF. Switching to OCR.")

        for i in fallback_pages:
            ocr_text = ocr_page(doc[i])
            text += f"\n[OCR Page {i+1}]\n{ocr_text}\n"

        return text.strip()

    except Exception as e:
        print(f"[ERROR] Failed to extract text: {e}")
        return ""

# ========= Api Schema =================#


class ChunkRequest(BaseModel):
    text: str
    pdf_name: str


class SummaryRequest(BaseModel):
    topics: List
    length: int
    target_lang: str
    style: str
    stream: bool = False


class GoldenNotesRequest(BaseModel):
    number_of_questions: int
    pdf_name: str
    target_lang: str
    stream: bool = False


class QuizRequest(BaseModel):
    number_of_questions: int
    pdf_name: str
    target_lang: str
    stream: bool = False


class FlashcardRequest(BaseModel):
    number_of_flashcards: int
    pdf_name: str
    target_lang: str
    stream: bool = False


class ExamRequest(BaseModel):
    number_of_questions: int
    pdf_name: str
    previous_questions: Optional[List[str]] = []
    target_lang: str
    stream: bool = False


class MindmapRequest(BaseModel):
    pdf_name: str
    target_lang: str
    stream: bool = False


class SpecificQuizRequest(BaseModel):
    number_of_mcq_questions: int
    number_of_subjective_questions: int
    pdf_name: str
    difficulty: str
    target_lang: str


class SpecificNoteRequest(BaseModel):
    topics: List


# ========= Prompt =============#


latex_prompt = """
LaTeX FORMATTING RULES (for JSON-safe responses):

1. Enclose all **inline math** in dollar signs: `$...$`
   - Example: `$x^2 + y^2 = z^2$`
   - DO NOT use `\\(...\\)` or `\\[...\\]` for math in JSON.

2. **Double all backslashes** in LaTeX commands to ensure valid JSON:
   - Use `\\\\frac{{a}}{{b}}`, not `\\frac{{a}}{{b}}`
   - Use `\\\\vec{{F}}`, `\\\\mu_0`, `\\\\nabla`, etc.

3. **Escape newlines** in long answers as `\\n`
   - DO NOT use raw line breaks or unescaped `\n`
   - Instead: `"This is line one.\\nThis is line two."`

4. **Do not escape dollar signs**. Just wrap math with them using `$...$`, never `\\$`.

5. Wrap all **keys and values** in double quotes.  
   - Escape any **double quotes (`"`) inside values** using `\\\"`  
   - Do **not** escape apostrophes (`'`).

6. NEVER return `\\$`. If a dollar symbol is required in plain text, write `"dollar"` or `"USD"`.

7. Return only the **JSON object**, without surrounding markdown (no ```json, no explanation, no trailing commas).

---
Example of a valid JSON output:
{{
  "type": "mathematical",
  "question": "What is the value of $\\\\int_{{0}}^{{1}} x^2 dx$?",
  "answer": "$\\\\frac{{1}}{{3}}$"
}}
"""


summary_prompt_template_kid = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a simplification engine. Your task is to explain structured topics as if you are talking to a 5-year-old child.\n\n"
            "The explanation must use very simple words, short sentences, and easy comparisons.\n"
            "Avoid technical jargon, complex phrases, or long explanations.\n"
            "Each topic should feel like a friendly teacher telling a story.\n"
            "You must strictly adhere to the specified word limit (±10%).\n\n"
            "Return the final output as a plain string using HTML tags:\n"
            "- Each topic starts with an <h2> tag for the topic title.\n"
            "- Each explanation goes inside a <p> tag.\n"
            "- Do not wrap the result in JSON or any other structure.\n"
            "- Do not include explanations, comments, or any non-HTML output.\n"
            "- The final output must only be valid HTML string with topic-wise explanations.\n"
            "- Here's an example:\n"
            "  <h2>The Sun</h2>\n"
            "  <p>The Sun is like a big hot ball in the sky. It gives us light and keeps us warm, just like a big lamp for the Earth.</p>\n"
            "  <h2>Rain</h2>\n"
            "  <p>Rain is water falling from the clouds. It is like the sky taking a shower to give water to plants and animals.</p>\n"
            "  (and so on)\n"
            f"{latex_prompt}",
        ),
        (
            "user",
            "Explain the following content in approximately {length} words total, using very simple words a 5-year-old can understand.\n\n"
            "### Topics and Subtopics:\n{topics}\n\n"
            "### Output:\nReturn the result as an HTML string with topic-wise explanations:",
        ),
    ]
)

summary_prompt_template_keypoints = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a summarization engine. Your task is to generate clear, concise summaries in the form of key points based on structured topic outlines.\n\n"
            "The summary must strictly adhere to the specified word limit (±10%).\n"
            "Each topic should be summarized using short, direct bullet points instead of long paragraphs.\n"
            "Use simple academic language, avoiding unnecessary detail, while still covering the core ideas.\n\n"
            "Return the final output as a plain string using HTML tags:\n"
            "- Each topic starts with an <h2> tag for the topic title.\n"
            "- Each explanation goes inside a <ul> list with multiple <li> items for key points.\n"
            "- Do not wrap the result in JSON or any other structure.\n"
            "- Do not include explanations, comments, or any non-HTML output.\n"
            "- The final output must only be valid HTML string with topic-wise key point summaries.\n"
            "- Here's an example:\n"
            "  <h2>Photosynthesis</h2>\n"
            "  <ul>\n"
            "    <li>Plants make food using sunlight, water, and air.</li>\n"
            "    <li>The process happens in the green parts of plants.</li>\n"
            "    <li>It gives out oxygen, which humans and animals need to breathe.</li>\n"
            "  </ul>\n"
            "  <h2>Water Cycle</h2>\n"
            "  <ul>\n"
            "    <li>Water from rivers and seas goes up as vapor.</li>\n"
            "    <li>Clouds are formed from vapor.</li>\n"
            "    <li>Rain brings water back to the ground.</li>\n"
            "  </ul>\n"
            "  (and so on)\n"
            f"{latex_prompt}",
        ),
        (
            "user",
            "Summarize the following content in approximately {length} words total, using key points instead of paragraphs.\n\n"
            "### Topics and Subtopics:\n{topics}\n\n"
            "### Output:\nReturn the result as an HTML string with topic-wise bullet point summaries:",
        ),
    ]
)

summary_prompt_template_analogy = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an explanation engine. Your task is to generate topic-wise summaries using simple everyday analogies and comparisons.\n\n"
            "The summary must strictly adhere to the specified word limit (±10%).\n"
            "Each topic should be explained in academic but accessible prose, where complex ideas are simplified by linking them to daily experiences.\n"
            "The analogies must make the concept intuitive, but not childish. Use comparisons like cooking, traveling, sports, games, nature, or tools.\n\n"
            "Return the final output as a plain string using HTML tags:\n"
            "- Each topic starts with an <h2> tag for the topic title.\n"
            "- Each explanation goes inside a <p> tag.\n"
            "- Do not wrap the result in JSON or any other structure.\n"
            "- Do not include explanations, comments, or any non-HTML output.\n"
            "- The final output must only be valid HTML string with topic-wise summaries.\n"
            "- Here's an example:\n"
            "  <h2>Electric Current</h2>\n"
            "  <p>Electric current is like water flowing through a pipe. Just as water moves from one place to another through pipes, electricity moves through wires, carrying energy where it is needed.</p>\n"
            "  <h2>Memory in Computers</h2>\n"
            "  <p>Computer memory is like a schoolbag. The bag can hold books you need for class, and the memory holds data the computer needs to work on tasks.</p>\n"
            "  (and so on)\n"
            f"{latex_prompt}",
        ),
        (
            "user",
            "Summarize the following content in approximately {length} words total, using everyday analogies to explain each concept.\n\n"
            "### Topics and Subtopics:\n{topics}\n\n"
            "### Output:\nReturn the result as an HTML string with topic-wise analogy-based summaries:",
        ),
    ]
)


mcq_prompt = """You are a skilled question setter for educational exams and assessments. "
            "Your only task is to generate exactly ONE multiple-choice question (MCQ) "
            "based on the provided topic and subtopics. You MUST strictly follow the format below.\n\n"
            "== FORMAT REQUIREMENTS ==\n"
            "- Only generate an MCQ of the given difficulty.\n"
            "- Each MCQ must have exactly four answer options.\n"
            "- Use the following JSON structure with all fields filled:\n"
            "  {{\n"
            '    "type": "mcq",\n'
            '    "question": "<escaped LaTeX text>",\n'
            '    "options": ["<text>", "<text>", "<text>", "<text>"],\n'
            '    "answer_index": "<0|1|2|3>",\n'
            '    "answer_explanation": "<a very detailed explanation of why this is the correct answer>",\n'
            '    "difficulty": "{format}"\n'
            "  }}\n"
            "- All strings must be properly escaped for JSON.\n"
            "- Do not return any markdown, commentary, or explanations outside the JSON.\n"
            "- Do not return null, undefined, or empty strings for any field.\n"
            "- Do not invent extra fields or change field names.\n"
            "- Do not include anything outside the JSON object.\n\n"
            "== LATEX USAGE IN QUESTIONS ==\n"
            "{latex_prompt}\n"
            "IMPORTANT:\n"
            "- Return ONLY the valid, escaped JSON object — nothing else.\n"
            "- Do not wrap it in Markdown backticks or add headings.\n"
            "- If any requirement cannot be fulfilled, return an error message: {{\"error\": \"Requirement not met.\"}}"""

sub_prompt = """You are a skilled exam question setter. Generate one **subjective academic** question based on the topic and subtopics.\n"
     "The question should be descriptive and require a detailed written answer (no MCQ or one-word types).\n\n"
     "{latex_prompt}"
     "Return a valid JSON and NO LLM JUNK:\n"
     "{{\n"
     '  "type": "subjective",\n'
     '  "question": "<escaped LaTeX text>",\n'
     '  "answer": "<model answer>",\n'
     '  "answer_explanation": "<a detailed explanation of the concept and why this is the answer>",\n'
     '  "difficulty": "Easy | Medium | Hard"\n'
     "}}"""

basic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a summarization engine. Your task is to generate academic summaries based on structured topic outlines.\n\n"
            "You must strictly adhere to the specified word limit (±10%). Going significantly over or under the target is unacceptable.\n"
            "The summary should be divided clearly by topic, with each topic forming a separate section in the output.\n"
            "Each topic's summary must be a well-written paragraph in academic prose, suitable for textbooks or educational narration.\n"
            f"{latex_prompt}"
            "Return the final output as a plain string using HTML tags:\n"
            "- Each topic starts with an <h2> tag for the topic title.\n"
            "- Each explanation goes inside a <p> tag.\n"
            "- Do not wrap the result in JSON or any other structure.\n"
            "- Do not include explanations, comments, or any non-HTML output.\n"
            "- The final output must only be valid HTML string with topic-wise summaries.\n"
            "- Here's an example:\n"
            "  <h2>Introduction to Trigonometry</h2>\n"
            "  <p>Trigonometry is the branch of mathematics that deals with the study of relationships between the angles and sides of triangles...</p>\n"
            "  <h2>Formulas</h2>\n"
            "  <p>Key trigonometric identities include: $\\sin^2(A) + \\cos^2(A) = 1$...</p>\n"
            "  (and so on)\n",
        ),
        (
            "user",
            "Summarize the following content in approximately {length} words total. \n\n"
            "### Topics and Subtopics:\n{topics}\n\n"
            "### Output:\nReturn the result as an HTML string with topic-wise summaries:",
        ),
    ]
)

summary_prompt_template_narrative = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a summarization engine. Your task is to generate complete narrative-style summaries based on structured topic outlines.\n\n"
            "The explanation should flow like a connected story rather than isolated points. Each section should feel like part of a continuous narrative, with smooth transitions and natural explanations.\n"
            "Use academic prose that is clear, coherent, and engaging, as if writing a textbook or chapter narration.\n"
            "Do not list points or use analogies unless necessary—focus on a fluid storytelling tone that makes the reader feel guided through the concepts.\n"
            "You must strictly adhere to the specified word limit (±10%).\n\n"
            "Return the final output as a plain string using HTML tags:\n"
            "- Each topic starts with an <h2> tag for the topic title.\n"
            "- Each explanation goes inside a <p> tag, written as a narrative passage.\n"
            "- Do not wrap the result in JSON or any other structure.\n"
            "- Do not include explanations, comments, or any non-HTML output.\n"
            "- The final output must only be valid HTML string with topic-wise narrative summaries.\n"
            "- Here's an example:\n"
            "  <h2>Industrial Revolution</h2>\n"
            "  <p>The Industrial Revolution marked a turning point in human history. Beginning in the late 18th century, it transformed societies through rapid advancements in technology, manufacturing, and transportation. Factories replaced traditional workshops, and steam power opened new possibilities for production and trade. This era not only reshaped economies but also altered the way people lived and worked.</p>\n"
            "  <h2>Impact on Society</h2>\n"
            "  <p>The changes brought by the Industrial Revolution were profound. Cities grew rapidly as workers moved from rural areas to industrial centers. While opportunities increased, so did challenges such as overcrowding and poor working conditions. At the same time, education, science, and communication advanced, laying the foundation for modern society.</p>\n"
            "  (and so on)\n"
            f"{latex_prompt}",
        ),
        (
            "user",
            "Summarize the following content in approximately {length} words total, as a continuous narrative for each topic.\n\n"
            "### Topics and Subtopics:\n{topics}\n\n"
            "### Output:\nReturn the result as an HTML string with topic-wise narrative summaries:",
        ),
    ]
)


# ========= Core Functions =============#


def clean_chunk(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[\u2580-\u259F\uFB00-\uFB4F]", "", text)
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    stripped = text.replace(" ", "")

    return text


def clean_chunks_list(chunks: list[str]) -> list[str]:
    cleaned = [clean_chunk(c) for c in chunks]
    return [c for c in cleaned if c]


def chunk_generator(text: str, chunk_size=1700, chunk_overlap=200) -> list[str]:

    logger.info("generating chunks")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", ".", " "],
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
    )

    chunks = splitter.split_text(text=text)

    clean_chunks = clean_chunks_list(chunks=chunks)

    return clean_chunks


SEMAPHORE = asyncio.Semaphore(10)

_LATEX_PLACEHOLDER_FMT = "LATEX_PLACEHOLDER_{id}"  # ascii-only base

_LATEX_RE = re.compile(
    r"""(
        \$\$.*?\$\$                     |   # display math $$...$$
        \$[^$].*?\$                     |   # inline $...$
        \\\[.*?\\\]                     |   # \[ ... \]
        \\\(.*?\\\)                     |   # \( ... \)
        \\begin\{(?P<env>[^\}]+)\}.*?\\end\{(?P=env)\}  # \begin{env} ... \end{env}
    )""",
    re.DOTALL | re.VERBOSE,
)


def protect_latex(text: str):
    """
    Replace LaTeX fragments with placeholders.
    Returns (protected_text, placeholders) where placeholders[i] is the original LaTeX for LATEX_PLACEHOLDER_i.
    """
    placeholders: List[str] = []

    def _repl(m: re.Match) -> str:
        idx = len(placeholders)
        placeholders.append(m.group(0))
        uid = uuid.uuid4().hex[:8]
        return f"{_LATEX_PLACEHOLDER_FMT.format(id=idx)}_{uid}"

    protected = _LATEX_RE.sub(_repl, text)
    return protected, placeholders


def _normalize_latex_for_json_fragment(latex_fragment: str) -> str:

    s = latex_fragment

    s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"$\1$", s, flags=re.DOTALL)

    s = s.replace("\r\n", "\n").replace("\n", "\\n")

    s = s.replace("\\", "\\\\")

    return s


def restore_latex(
    text: str, placeholders: List[str], apply_json_rules: bool = True
) -> str:

    ph_re = re.compile(r"(?i)latex_placeholder_(\d+)_[0-9a-fA-F]{8}")

    def _repl(m: re.Match) -> str:
        i = int(m.group(1))
        try:
            frag = placeholders[i]
            if apply_json_rules:
                frag = _normalize_latex_for_json_fragment(frag)
                frag = frag.replace("\\\\", "\\")
            return frag
        except Exception:
            return m.group(0)

    return ph_re.sub(_repl, text)


async def translate_chunk_google(
    client: httpx.AsyncClient, text: str, source: str, target: str
):

    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": source,
        "tl": target,
        "dt": "t",
        "q": text,
    }

    async with SEMAPHORE:
        response = await client.get(url, params=params)
        response.raise_for_status()

    data = response.json()
    translated_parts = [seg[0] for seg in data[0] if seg and seg[0]]
    return "".join(translated_parts)


async def google_translate(text: str, target: str, source: str = "auto"):

    protected_text, placeholders = protect_latex(text)

    chunks = list(chunk_generator(protected_text, 4000, 200))

    timeout = httpx.Timeout(10.0, read=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            translate_chunk_google(client, chunk, source, target) for chunk in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    translated_parts = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            # keep the protected chunk as-is (or indicate error)
            translated_parts.append(f"[ERROR in chunk {i}: {res}]")
        else:
            translated_parts.append(res.strip())

    translated_joined = " ".join(translated_parts)

    restored = restore_latex(translated_joined, placeholders, apply_json_rules=True)

    return restored


def extract_unique_topic_dict(data_list):
    topic_list = []

    seen_topics = {}

    for data in data_list:
        for entry in data.get("topics", []):
            topic = entry.get("topic")
            subtopics = entry.get("subtopics", [])

            if topic not in seen_topics:
                seen_topics[topic] = {
                    "id": str(uuid.uuid4()),
                    "topic_name": topic,
                    "subtopics": [],
                }

            existing_sub_names = {
                s["subtopic_name"] for s in seen_topics[topic]["subtopics"]
            }

            for sub in subtopics:
                if sub not in existing_sub_names:
                    seen_topics[topic]["subtopics"].append(
                        {"id": str(uuid.uuid4()), "subtopic_name": sub}
                    )

    topic_list = list(seen_topics.values())
    return topic_list


async def sync_generate_topics_and_subtopics(chunks: list[str]):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in analyzing and organizing educational content. "
                "Your task is to extract the most relevant high-level topics and their corresponding subtopics from the given text. "
                "Always output a clean and well-structured JSON object as specified.",
            ),
            (
                "human",
                """Analyze the following text and extract a structured hierarchy of topics and subtopics. 
        Each topic should represent a major concept or theme in the text. Subtopics should be specific ideas or components related to that topic.

        Ensure the output follows this strict JSON format:
        {{
        "topics": [
            {{
            "topic": "<main_topic_name>",
            "subtopics": ["<subtopic_1>", "<subtopic_2>", "..."]
            }},
            ...
        ]
        }}

    Avoid unnecessary explanation. Return only the JSON output. 
    Here is the text:
    {input_text}""",
            ),
        ]
    )

    chain = prompt | llm | parser

    # Process chunks in parallel using asyncio.gather
    tasks = [chain.ainvoke({"input_text": chunk}) for chunk in chunks]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions gracefully
    clean_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(f"Chunk {i} failed with exception: {response}")
        else:
            clean_responses.append(response)

    unique_topics_data = extract_unique_topic_dict(clean_responses)

    return unique_topics_data


def find_semantic_links(structured_topics: List[Dict], threshold=0.65):

    topic_texts = {
        topic["id"]: topic["topic_name"]
        + " "
        + " ".join(sub["subtopic_name"] for sub in topic.get("subtopics", []))
        for topic in structured_topics
    }

    topic_embeddings = {
        tid: model.encode(text, convert_to_tensor=True)
        for tid, text in topic_texts.items()
    }

    linked_topics = []
    topic_ids = list(topic_embeddings.keys())

    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            tid1, tid2 = topic_ids[i], topic_ids[j]
            sim_score = util.cos_sim(
                topic_embeddings[tid1], topic_embeddings[tid2]
            ).item()

            if sim_score >= threshold:
                linked_topics.append((tid1, tid2, round(sim_score, 3)))

    return linked_topics


def push_topics_to_neo4j(structured_topics: List[Dict], pdf_name: str):
    nodes: Dict[str, Node] = {}
    relationships: List[Relationship] = []

    for topic in structured_topics:
        topic_name = topic["topic_name"]
        topic_uuid = topic["id"]

        topic_node = Node(
            id=topic_name,
            type="Topic",
            properties={"uuid": topic_uuid, "source_pdf": pdf_name},
        )
        nodes[topic_name] = topic_node

        for sub in topic.get("subtopics", []):
            sub_name = sub["subtopic_name"]
            sub_uuid = sub["id"]

            sub_node = Node(
                id=sub_name,
                type="Subtopic",
                properties={"uuid": sub_uuid, "source_pdf": pdf_name},
            )
            nodes[sub_name] = sub_node

            relationships.append(
                Relationship(
                    source=topic_node,
                    target=sub_node,
                    type="HAS_SUBTOPIC",
                    properties={"source_pdf": pdf_name},
                )
            )

    semantic_links = find_semantic_links(structured_topics)

    for src_uuid, dst_uuid, score in semantic_links:
        src_node = next(
            (
                node
                for node in nodes.values()
                if node.properties.get("uuid") == src_uuid
            ),
            None,
        )
        dst_node = next(
            (
                node
                for node in nodes.values()
                if node.properties.get("uuid") == dst_uuid
            ),
            None,
        )

        if src_node and dst_node:
            relationships.append(
                Relationship(
                    source=src_node,
                    target=dst_node,
                    type="SEMANTICALLY_RELATED",
                    properties={"score": score, "source_pdf": pdf_name},
                )
            )

    graph_doc = GraphDocument(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=Document(
            page_content="Auto-generated mindmap graph", metadata={"id": pdf_name}
        ),
    )

    graph.add_graph_documents([graph_doc], include_source=False)
    logger.info("Graph successfully pushed to Neo4j.")


def upsert_chunk(chunks: list[str], pdf_name: str):
    results = []
    try:
        for chunk in chunks:
            vector = model.encode(chunk).tolist()
            doc_id = hashlib.sha256((pdf_name + chunk).encode("utf-8")).hexdigest()
            doc = {"text": chunk, "vector": vector, "pdf_name": pdf_name}
            es.index(index=INDEX_NAME, id=doc_id, document=doc)
            results.append({"result": "upserted", "id": doc_id})
        return results
    except Exception as e:
        raise ValueError(f"Cannot upsert: {e}")


def extract_context(search_results):
    return "\n\n".join(result["text"] for result in search_results if "text" in result)


def search_chunk(query: str, k: int = 3):
    try:
        vector = model.encode(query).tolist()

        body = {
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": k,
                "num_candidates": 100,
            }
        }

        res = es.search(index=INDEX_NAME, body=body)
        results = [
            {"text": hit["_source"]["text"], "score": hit["_score"]}
            for hit in res["hits"]["hits"]
        ]

        return results

    except Exception as e:
        raise ValueError(f"Search failed: {e}")


def mapping(ques_str: dict):
    if ques_str.get("type") == "mcq":
        ques_str["type"] = "1"
    elif ques_str.get("type") == "subjective":
        ques_str["type"] = "2"

    difficulty_map = {"easy": "1", "medium": "2", "hard": "3", "very hard": "4"}
    ques_str["difficulty"] = difficulty_map.get(
        (ques_str.get("difficulty")).lower(), "2"
    )

    return ques_str


def format_topics(data_list):
    lines = []
    for topic_obj in data_list:
        topic = topic_obj.get("topic_name", "").lower()
        subtopics = [
            sub.get("subtopic_name", "").lower()
            for sub in topic_obj.get("subtopics", [])
        ]
        line = topic + " " + " ".join(subtopics)
        lines.append(line)
    return "\n".join(lines)



def get_related_concepts(topic_name: str) -> List[str]:

    query = f"""
    MATCH (t {{id: "{topic_name}"}})-[*1..2]-(related)
    RETURN DISTINCT related.id AS related_id
    LIMIT 5
    """

    results = graph.query(query)
    return [record["related_id"] for record in results if "related_id" in record]


def get_top_concepts(n: int, pdf_name: str) -> list[str]:
    query = f"""
    MATCH (n)-[r]-()
    WHERE n.source_pdf = '{pdf_name}' AND r.source_pdf = '{pdf_name}'
    RETURN n.id AS node, COUNT(r) AS connections
    ORDER BY connections DESC
    LIMIT {n}
    """
    results = graph.query(query)
    top_nodes = [record["node"] for record in results]
    return top_nodes


def generate_mm_json(pdf_name: str) -> list[dict]:
    topics = get_top_concepts(20, pdf_name)
    jj = []
    for topic in topics:
        ss = []
        subtopics = get_related_concepts(topic)
        for subtopic in subtopics:
            ss.append({"subtopic_name": subtopic})
        jj.append({"topic_name": topic, "subtopics": ss})

    return jj


def find_semantic_links_from_raw_dict(
    data: List[Dict], threshold: float = 0.65
) -> List[tuple[str, str, float]]:

    topic_texts = {}

    for i, topic in enumerate(data):
        topic_id = f"topic_{i}"
        topic_name = topic["topic_name"]
        subtopics = " ".join(sub["subtopic_name"] for sub in topic.get("subtopics", []))
        combined_text = f"{topic_name} {subtopics}"
        topic_texts[topic_id] = combined_text

    topic_embeddings = {
        tid: model.encode(text, convert_to_tensor=True)
        for tid, text in topic_texts.items()
    }

    linked_topics = []
    topic_ids = list(topic_embeddings.keys())

    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            tid1, tid2 = topic_ids[i], topic_ids[j]
            sim_score = util.cos_sim(
                topic_embeddings[tid1], topic_embeddings[tid2]
            ).item()

            if sim_score >= threshold:
                linked_topics.append((tid1, tid2, round(sim_score, 3)))

    return linked_topics


async def translate_mindmap(data: dict, target_lang: str) -> dict:
    """
    Recursively translate the 'label' and 'level' fields in the mindmap
    without touching 'id' or structure.
    Uses asyncio.gather for parallel translation of children.
    """
    if not isinstance(data, dict):
        return data
    print("=" * 20 + "translate mm")
    new_node = {}

    tasks = []

    for key, value in data.items():
        if key in ("label", "level") and isinstance(value, str):
            new_node[key] = await google_translate(value, target_lang)

        elif key == "children" and isinstance(value, list):
            tasks = [translate_mindmap(child, target_lang) for child in value]
            new_node[key] = await asyncio.gather(*tasks)

        else:
            new_node[key] = value

    return new_node


async def generate_mindmap(pdf_name: str, target_lang):

    topics = generate_mm_json(pdf_name)

    links = find_semantic_links_from_raw_dict(topics)

    import uuid
    from langchain.prompts import ChatPromptTemplate

    mindmap_id = str(uuid.uuid4())

    mm_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that formats knowledge into structured mind maps use normal quotes, and return a valid json.",
            ),
            (
                "human",
                """
        Given the following topic clusters and similarity scores, generate a structured mind map in the following JSON format:

        ```json
        {{
        "id": "root",
        "level": "<main-topic(generate a main heading for the content)>",
        "children": [
            {{
            "id": "chapter1",
            "label": "<chapter-label>",
            "children": [
                {{
                "id": "concept-a",
                "label": "<concept-label>",
                "children": [
                    {{ "id": "detail-1", "label": "<detail-label>" }}
                ]
                }}
            ]
            }}
        ]
        }}
        
        here is the input {input_json}""",
            ),
        ]
    )
    chain = mm_prompt | llm | parser

    input_data = {"topics": topics, "similarities": links}

    input_json = json.dumps(input_data, indent=2)

    attempts = 0

    response = None

    while attempts < 3:
        try:
            response = chain.invoke({"input_json": input_json})
            break

        except Exception as e:
            logger.debug(f"Retrying mindmap : Error {e}")
            attempts += 1

    if response:

        print("=" * 20 + "sent to translation")

        response["id"] = str(uuid.uuid4())

        if target_lang!= "en":

            final = await translate_mindmap(response, target_lang)

        else: 
            final = response

        return final

    else:
        return {"no mindmap generated"}


# ============ Async Wrapper =========#


async def process_chunk(chunk: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in analyzing and organizing educational content. "
                "Your task is to extract the most relevant high-level topics and their corresponding subtopics from the given text. "
                "Always output a clean and well-structured JSON object as specified.",
            ),
            (
                "human",
                """Analyze the following text and extract a structured hierarchy of topics and subtopics. 
        Each topic should represent a major concept or theme in the text. Subtopics should be specific ideas or components related to that topic.

        Ensure the output follows this strict JSON format:
        {{
        "topics": [
            {{
            "topic": "<main_topic_name>",
            "subtopics": ["<subtopic_1>", "<subtopic_2>", "..."]
            }},
            ...
        ]
        }}

    Avoid unnecessary explanation. Return only the JSON output. 
    Here is the text:
    {input_text}""",
            ),
        ]
    )

    chain = prompt | llm | parser

    try:
        return await chain.ainvoke({"input_text": chunk})
    except Exception as e:
        logger.warning(f"Failed to process chunk: {e}")
        return None


async def generate_topics_and_subtopics(chunks: list[str]) -> list[dict]:
    tasks = [process_chunk(chunk) for chunk in chunks]
    responses = await asyncio.gather(*tasks)

    clean_responses = [r for r in responses if r is not None]

    unique_topics_data = extract_unique_topic_dict(clean_responses)
    return unique_topics_data


async def chunk_savetodb_and_topics(text: str, pdf_name: str):
    chunks = chunk_generator(text)
    upsert_chunk(chunks, pdf_name)
    topics = await sync_generate_topics_and_subtopics(chunks)
    push_topics_to_neo4j(topics, pdf_name)
    return topics


async def create_golden_notes(
    number_of_questions: int, pdf_name: str, target_lang: str
):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a highly skilled educator and science communicator. "
                "Your task is to write a concise, memorable, and insightful golden note (50–100 words) that captures the essence of the given topic.\n\n"
                "Use the retrieved source content as context. Rephrase clearly and precisely; do not copy text directly.\n"
                "Maintain an engaging, crisp tone suitable for a science notebook.\n\n"
                f"{latex_prompt}"
                "- Avoid newlines or unnecessary special characters in the JSON output.\n\n"
                '- if you return string in "" then make sure to escape all double quotes in it and similarly for single quotes'
                "Include all relevant subtopics in your response. Avoid filler or generic statements.\n\n"
                "Return only a single valid JSON object in this format:\n"
                "{{\n"
                '  "title": "Heading for the note",\n'
                '  "description": "Explanation"\n'
                "}}",
            ),
            (
                "user",
                "Topic: {topic}\n\nSubtopic: {subtopic}\n\nRetrieved Chunks:\n{chunks}",
            ),
        ]
    )

    chain = prompt | llm | parser
    notes = []

    priority_topics = get_top_concepts(number_of_questions, pdf_name)

    for topic in priority_topics:
        concepts = get_related_concepts(topic)
        all_concepts = " ".join(concepts)

        query = topic + all_concepts
        search_result = search_chunk(query=query)
        context = extract_context(search_result)

        response = await chain.ainvoke(
            {"topic": topic, "subtopic": all_concepts, "chunks": context}
        )

        response["title"] = await google_translate(response["title"], target_lang)
        response["description"] = await google_translate(
            response["description"], target_lang
        )

        response["id"] = str(uuid.uuid4())
        notes.append(response)

    return notes


# =========== Test =============#

if __name__ == "__main__":

    gn = asyncio.run(generate_mindmap("agentic_hack","en"))

    with open("intermediate/MINDMAP.json","w",encoding="utf-8") as f:
        json.dump(gn,f, indent= 2, ensure_ascii= False)

       

    
