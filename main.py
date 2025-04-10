from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import json
import re
from pathlib import Path

# Download tokenizer
nltk.download('punkt')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load JSON with absolute path
BASE_DIR = Path(__file__).resolve().parent

def load_json_data(filename):
    file_path = BASE_DIR / filename
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load JSON
ipc_data = load_json_data("ipc.json")
crpc_data = load_json_data("crpc.json")

# IT Act (static)
it_act_rules = [
    "Section 43: Penalty for damage to computer, computer system, etc.",
    "Section 66: Hacking with computer system and data theft.",
    "Section 66C: Punishment for identity theft.",
    "Section 66D: Punishment for cheating by impersonation using computer resources.",
    "Section 67: Punishment for publishing or transmitting obscene material in electronic form.",
    "Section 69: Powers to issue directions for interception or monitoring of information.",
    "Section 72: Breach of confidentiality and privacy.",
    "Section 73: Publishing false digital signatures.",
    "Section 74: Publication of forged electronic signatures.",
    "Section 77: Compensation and penalties for failing to protect sensitive personal data.",
    "Section 80: Powers of police officers to enter, search, and arrest without a warrant."
]

section_explanations = {
    "Section 43": "This section imposes penalties for causing damage to computer systems.",
    "Section 66": "This section addresses hacking and data theft.",
    "Section 66C": "This section focuses on identity theft.",
    "Section 66D": "This section targets impersonation using computer resources.",
    "Section 67": "This section deals with obscene electronic material.",
    "Section 69": "This section allows interception of digital communications under law.",
    "Section 72": "This section covers breach of confidentiality and privacy.",
    "Section 73": "This section penalizes false digital signatures.",
    "Section 74": "This section covers forged electronic signatures.",
    "Section 77": "This section outlines compensation for personal data protection failures.",
    "Section 80": "This section empowers police to act without warrant under certain conditions."
}

# Prepare data for BM25
documents = []
source_map = []

# Add IT Act
for rule in it_act_rules:
    documents.append(rule.lower())
    section = rule.split(":")[0].strip()
    description = rule.split(":", 1)[1].strip()
    source_map.append({
        "source": "IT Act",
        "section": section,
        "description": description,
        "explanation": section_explanations.get(section, "")
    })

# Add IPC
for entry in ipc_data:
    try:
        section_code = f"Section {entry['Section']}"
        title = entry.get("section_title", "")
        desc = entry.get("section_desc", "")
        content = f"{section_code}: {title}. {desc}"
        documents.append(content.lower())
        source_map.append({
            "source": "IPC",
            "section": section_code,
            "description": title,
            "explanation": desc
        })
    except KeyError as e:
        print("Missing key in IPC entry:", e)

# Add CrPC
for entry in crpc_data:
    try:
        section_code = f"Section {entry['section']}"
        title = entry.get("section_title", "")
        desc = entry.get("section_desc", "")
        content = f"{section_code}: {title}. {desc}"
        documents.append(content.lower())
        source_map.append({
            "source": "CrPC",
            "section": section_code,
            "description": title,
            "explanation": desc
        })
    except KeyError as e:
        print("Missing key in CrPC entry:", e)

# Tokenize and initialize BM25
tokenized_docs = [word_tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask")
def ask_question(query: str = Query(..., description="Ask a legal question")):
    query = query.strip()
    if not query:
        return {"question": query, "answer": "Please enter a valid legal question."}

    # Direct match for Section
    match = re.search(r'section\s+(\d+[A-Z]*)', query, re.IGNORECASE)
    if match:
        section_code = f"Section {match.group(1).upper()}"
        for entry in source_map:
            if entry["section"] == section_code:
                return {
                    "question": query,
                    "answer": (
                        f"Under {entry['section']} of the {entry['source']}, which deals with {entry['description'].lower()}, "
                        f"the law provides the following explanation: {entry['explanation']}"
                    )
                }

    # BM25 Search
    tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(tokens)
    top_index = np.argmax(scores)
    top_entry = source_map[top_index]

    return {
        "question": query,
        "answer": (
            f"Under {top_entry['section']} of the {top_entry['source']}, which deals with {top_entry['description'].lower()}, "
            f"the law provides the following explanation: {top_entry['explanation']}"
        )
    }
