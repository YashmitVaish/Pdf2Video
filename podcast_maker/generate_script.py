import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import string
import re
from dotenv import load_dotenv

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

rate_limiter = InMemoryRateLimiter(
    requests_per_second=80,        
    check_every_n_seconds=0.1,    
    max_bucket_size=200           
)

load_dotenv(".env")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.65
)

def clean_json_block(text: str) -> str:
    """
    Removes Markdown-style ```json ... ``` or ``` ... ``` wrappers
    and returns just the inner JSON.
    """
    # Remove triple backtick blocks with optional 'json'
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL | re.MULTILINE)
    return cleaned.strip()

def clean_chunk(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[\u2580-\u259F\uFB00-\uFB4F]', '', text)
    text = re.sub(r'[' + re.escape(string.punctuation) + r']{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stripped = text.replace(" ", "")
    non_alpha_ratio = sum(1 for c in stripped if not c.isalnum()) / (len(stripped) + 1e-5)

    if (
        len(text) < 30 or               
        text.count(' ') < 5 or          
        non_alpha_ratio > 0.5           
    ):
        return ""

    return text

def clean_chunks_list(chunks: list[str]) -> list[str]:
    cleaned = [clean_chunk(c) for c in chunks]
    return [c for c in cleaned if c] 

def chunk_generator(text:str, chunk_size = 1700, chunk_overlap = 200) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        separators= ["\n","\n\n","."," "],
        chunk_overlap = chunk_overlap,
        chunk_size = chunk_size
    )

    chunks = splitter.split_text(text = text)

    clean_chunks = clean_chunks_list(chunks = chunks)

    return clean_chunks


last_10_dialogues = ["Start of the Conversation"]


punctuation_guide = """
Punctuation & Prosody Guide:

1. Period (.)
   - Signals a neutral sentence ending.
   - Causes a brief pause (~250ms) and a falling pitch contour.
   - Example: "I am going home."

2. Comma (,)
   - Signals a short pause (~150ms) with a slight pitch drop or continuation tone.
   - Good for breaking long sentences into natural chunks.
   - Example: "Well, I think that’s fine."

3. Exclamation Mark (!)
   - Signals excitement, emphasis, or urgency.
   - Increases pitch range and stress on preceding words.
   - Example: "Watch out!"

4. Question Mark (?)
   - Signals inquiry or curiosity.
   - Produces a rising pitch contour on the last stressed syllable.
   - Example: "Are you serious?"

5. Colons (:) and Semicolons (;)
   - Slight pause, with a forward-flowing intonation.
   - Used for lists or elaboration.
   - Example: "We need three things: food, water, and shelter."

6. Extra Spaces
   - Adding two or more spaces between words forces a perceptible pause.
   - Useful for dramatic timing.
   - Example: "This is important…  very important."

7. Capitalization
   - Doesn’t change pitch directly, but can emphasize words when combined with punctuation.
   - Example: "This is VERY important!"

8. Phonetic Emphasis
   - You can elongate vowels with extra letters for drawn-out delivery.
   - Example: "Nooo!" → longer "o" sound.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", 
 """You are a dialogue generator. 
You will be given:
1. A topic name 
2. A text chunk containing relevant details and examples
3. A list of previous questions to maintain conversation flow

Instructions:
- Create a natural conversation between two speakers (Person 1, Person 2).
- Use the topic_name as the central theme.
- Use details and examples from the chunk to add depth.
- Use the previous_questions to ensure smooth conversation continuity.
- Keep responses conversational and descriptive so the listener stays engaged.
- Add punctuation appropriately for good rendering.
- Satire and light humor are welcome if contextually appropriate.

STRICT RULES:
1. Output MUST be a valid JSON array of objects.
2. Each object must contain ONLY:
   - "speaker" (string: "Person 1" or "Person 2")
   - "content" (string: dialogue text)
3. DO NOT include anything outside the JSON array.
   - No ```json fences
   - No ~ tildes
   - No commentary before or after JSON
   - No LaTeX, TeX, or math mode markers (`$`, `$$`, `\\(`, `\\)`)
4. NEVER use raw math symbols. Always spell them out in plain English:
   - Write "2 Pi radians" instead of `$2\\pi$`
   - Write "degrees" instead of `°`
   - Write "square root of x" instead of `√x`
   - Write "less than or equal to" instead of `≤`
   - Write "greater than or equal to" instead of `≥`
   - Write "times" instead of `×`
5. The output must be plain JSON only, with nothing before or after it.

Example of correct format:
[
  {{"speaker": "Person 1", "content": "So, what’s the deal with this topic?"}},
  {{"speaker": "Person 2", "content": "Well, let me break it down with some examples..."}}
]
"""
),
    ("human", """
        Relevant chunk for which dialogue is to be generated: {chunk}
        previous dialogues = {previous_dialogue}
""")
])

def format_conversation_array(old_array:list[str],generated_convo:dict):
    for dia in generated_convo:
        text = ""
        text = text+ dia["speaker"]+ " : " +dia["content"]
        old_array.append(text)
    new_arr = old_array[-10:]
    return new_arr

with open("cleaned_chunks.json","r",encoding = "utf-8") as f:
    datas = json.load(f)
    f.close()

import time
from langchain_core.output_parsers import StrOutputParser
parser = JsonOutputParser()
strparser = StrOutputParser()

all_text = ""

for data in datas:

    title = data["section_title"]
        
    content = data["content"]

    tables = data ["tables"]

    all_text = all_text+ f"section title : {title}\n content for {title} : {content}, tables(if needed) {tables} "


chunks = chunk_generator(all_text)
aa = len(chunks)


all_dia = []
ti =5
i=1
for chunk in chunks:

    prompt_input = prompt.format(
        format=punctuation_guide,
        chunk=chunk,
        previous_dialogue=last_10_dialogues
    )

    for k in range (3):

        try:
            raw_response = llm.invoke(prompt_input)  

            if hasattr(raw_response, "content"):
                raw_response = raw_response.content

            raw_response = clean_json_block(raw_response)
                                
            reponse = parser.invoke(raw_response)

            for content in reponse:
                all_dia.append(content)
            
            break
        except:
            print("retry")
            time.sleep(ti)
            ti = ti*2

    last_10_dialogues = format_conversation_array(last_10_dialogues, reponse)

    print(f"{i}/{aa}")
        
    with open(f"podcast.json","w",encoding="utf-8") as f:
        json.dump(all_dia,f,indent=2)

    time.sleep(10)
    i+=1