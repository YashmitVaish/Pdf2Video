import pdfplumber
import json

def extract_text_and_tables(pdf_path):
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_data = {"page_number": page_num, "text": "", "tables": []}

            # Extract text
            page_data["text"] = page.extract_text() or ""

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                page_data["tables"].append(table)

            result.append(page_data)

    return result

if __name__ == "__main__":
    pdf_path = "jesc111.pdf"
    data = extract_text_and_tables(pdf_path)

    # Pretty-print as JSON
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # Save to file
    with open("extracted_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


import re
import json

def remove_junk(text):
    # Collapse repeated characters like "AAAAA" to "A"
    text = re.sub(r'([A-Za-z])\1{2,}', r'\1', text)
    # Remove lines of mostly numbers, dots, or gibberish patterns
    text = re.sub(r'[0-9]{4,}|[\.]{4,}|[A-Z]{5,}|\s{2,}', '', text)
    return text

def merge_lines(text):
    lines = text.split('\n')
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            continue

        # End buffer if the line seems like a full sentence
        if line.endswith(('.', '?', ':')) or re.match(r'^[A-Z][^a-z]+$', line):
            buffer += ' ' + line
            merged.append(buffer.strip())
            buffer = ""
        else:
            buffer += ' ' + line

    if buffer:
        merged.append(buffer.strip())

    return '\n'.join(merged)

import spacy

nlp = spacy.load("en_core_web_sm")

def is_heading_line(line):
    doc = nlp(line.strip())
    if not line.strip():
        return False

    # Heuristic 1: All uppercase or Titlecase
    if line.strip().isupper() or line.strip().istitle():
        return True

    # Heuristic 2: Mostly nouns and proper nouns
    noun_ratio = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"]) / (len(doc) or 1)
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")

    if noun_ratio > 0.6 and verb_count == 0:
        return True

    return False

def split_into_topics(text):
    lines = text.split("\n")
    chunks = []
    current_title = "Introduction"
    order = 1
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_heading_line(line):
            # Save the previous buffer as a chunk
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

    # Save the last chunk
    if buffer:
        chunks.append({
            "chunk_id": f"slide_{order:03d}",
            "section_title": current_title,
            "content": buffer.strip(),
            "tables": [],
            "order": order
        })

    return chunks

def process_extracted_data(raw_pages):
    all_chunks = []
    order = 1

    for page in raw_pages:
        raw_text = page.get("text", "")
        cleaned = remove_junk(raw_text)
        merged = merge_lines(cleaned)

        topics = split_into_topics(merged)

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

    return all_chunks

if __name__ == "__main__":
    # Example input (replace with your pdfplumber-extracted data)
    with open("extracted_output.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    final_chunks = process_extracted_data(raw_data)

    # Output the result
    print(json.dumps(final_chunks, indent=2, ensure_ascii=False))

    # Save to file
    with open("cleaned_chunks.json", "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)

    print("\nCleaned & split data saved to 'cleaned_chunks.json'")
