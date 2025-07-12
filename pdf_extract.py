import pdfplumber
import spacy
import re
from langchain.prompts import PromptTemplate
import ollama

nlp = spacy.load("en_core_web_sm")

def extract_from_pdf(pdf_path : str) -> list[dict]:
    content = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num , page in enumerate(pdf.pages,start=1):
            page_data = {
                "page_number": page_num,
                "text" : "",
                "tables" : []
            }

            page_data["text"] = page.extract_text() or None

            tables = page.extract_tables()
            for table in tables:
                page_data["tables"].append(table)
            
            content.append(page_data)
        
    return content

def remove_ocr_junk(text:str)-> str:

    text = re.sub(r'([A-Za-z])\1{2,}', r'\1', text)
    text = re.sub(r'[0-9]{4,}|[\.]{4,}|[A-Z]{5,}|\s{2,}', '', text)
    return text
            
def format_lines(text:str) -> str:
    lines = text.split("\n")
    formatted = []
    sentence = ""

    for line in lines: 
        line = line.strip()

        if not line:
            if sentence:
                formatted.append(sentence)
                sentence = ""
            continue

        if line.endswith((".","?",":","!")) or re.match(r'^[A-Z][^a-z]+$', line):
            sentence += " "+ line
            formatted.append(sentence.strip())
            sentence = " "
        else:
            sentence += " "+ line
        
    if sentence:
        formatted.append(sentence.strip())

    return "\n".join(formatted)

def is_heading(line):

    if not line.strip():
        return False
    
    doc = nlp(line.strip())

    if line.strip().isupper() or line.strip().istitle():
        return True

    # Heuristic 2: Mostly nouns and proper nouns
    noun_ratio = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"]) / (len(doc) or 1)
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")

    if noun_ratio > 0.6 and verb_count == 0:
        return True

    return False

def split_topics(text):
    lines = text.split("\n")
    chunks = []
    current_title = " "
    order = 1
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_heading(line):
            if buffer:
                chunks.append({
                    "chunk_id": f"slide_{order:03d}",
                    "section_title": current_title,
                    "content": buffer.strip(),
                    "tables": [],
                    "order": order
                })
                order += 1
                buffer = ""

            current_title = line
        else:
            buffer += " " + line

    if buffer:
        chunks.append({
            "chunk_id": f"slide_{order:03d}",
            "section_title": current_title,
            "content": buffer.strip(),
            "tables": [],
            "order": order
        })

    return chunks

def process_raw_chunks(raw_chunks):
    all_chunks = []
    order = 1

    for page in raw_chunks:
        raw_text = page.get("text", "")
        cleaned = remove_ocr_junk(raw_text)
        merged = format_lines(cleaned)

        topics = split_topics(merged)

        # Attach any tables from the page to the first topic
        tables = page.get("tables", [])
        if tables and topics:
            topics[0]["tables"] = tables

        # Assign sequential order numbers
        for topic in topics:
            topic["page_number"] = page.get("page_number", -1)
            topic["order"] = order
            topic["chunk_id"] = f"slide_{order:03d}"
            order += 1

        all_chunks.extend(topics)
  

def process_pdf(pdf_path):

    raw_chunks = extract_from_pdf(pdf_path)

    final_chunks = process_raw_chunks(raw_chunks)

    return final_chunks







    








