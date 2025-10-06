from base import Agent
from data_types import Page
from typing import List
import pdfplumber
import spacy
import re

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

    noun_ratio = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"]) / (len(doc) or 1)
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")

    if noun_ratio > 0.6 and verb_count == 0:
        return True

    return False

def split_topics(text):
    lines = text.split("\n")
    chunks = []
    current_title = " "
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_heading(line):
            if buffer:
                chunks.append({
                    "section_title": current_title,
                    "text": buffer.strip(),
                    "tables": [],
                })
                buffer = ""

            current_title = line
        else:
            buffer += " " + line

    if buffer:
        chunks.append({
            "section_title": current_title,
            "text": buffer.strip(),
            "tables": [],
        })

    return chunks

def process_raw_chunks(raw_chunks) -> List[Page]:
    pages: List[Page] = []

    for page in raw_chunks:
        raw_text = page.get("text", "")
        cleaned = remove_ocr_junk(raw_text)
        merged = format_lines(cleaned)

        topics = split_topics(merged)

        tables = page.get("tables", [])
        page_number = page.get("page_number", -1)

        # Build Page objects from topics
        for idx, topic in enumerate(topics):
            section_title = topic.get("title", f"Section {idx+1}")
            text = topic.get("content", merged)

            # If this is the first topic on the page, attach tables
            topic_tables = tables if idx == 0 else []

            page_obj = Page(
                page_number=page_number,
                section_title=section_title,
                text=text,
                tables=topic_tables,
            )
            pages.append(page_obj)

    return pages
  

def process_pdf(pdf_path):

    raw_chunks = extract_from_pdf(pdf_path)

    final_chunks = process_raw_chunks(raw_chunks)

    return final_chunks

#=================== Main Class=======================#

class PDFExtractor(Agent):
    def run(self, pdf_path:str)-> List[Page]:

        raw_chunks = extract_from_pdf(pdf_path)

        final_chunks = process_raw_chunks(raw_chunks)

        return final_chunks


        

