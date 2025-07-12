import pdfplumber 
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_from_pdf(pdf_path: str):
    content = []

    with pdfplumber.open(pdf_path) as pdf:
        