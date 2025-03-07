from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import KeyedVectors
import os
from pathlib import Path
from fastapi import Request

# Ensure NLTK resources are available
nltk.download('punkt')

app = FastAPI()

# Serve templates
templates = Jinja2Templates(directory="templates")

# Attempt to load the pre-trained law2vec model
if os.path.exists("law2vec.bin"):
    law2vec_model = KeyedVectors.load_word2vec_format("law2vec.bin", binary=True)
else:
    print("Warning: law2vec.bin not found. Law2Vec semantic scoring will be disabled.")
    law2vec_model = None

# Define IT Act, 2000 Sections and Subsections as legal references
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

# Dictionary of custom explanations for each IT Act section
section_explanations = {
    "Section 43": "This section imposes penalties for causing damage to computer systems, covering unauthorized access or harm.",
    "Section 66": "This section addresses hacking and data theft, used when there is unauthorized access to computer systems.",
    "Section 66C": "This section focuses on identity theft, penalizing misuse of another's identity.",
    "Section 66D": "This section targets impersonation and cheating using computer resources, ensuring accountability for digital fraud.",
    "Section 67": "This section deals with the publication or transmission of obscene material electronically.",
    "Section 69": "This section provides authorities with powers to intercept or monitor digital communications under specific conditions.",
    "Section 72": "This section covers breach of confidentiality and privacy, protecting sensitive information from unauthorized disclosure.",
    "Section 73": "This section penalizes the publishing of false digital signatures.",
    "Section 74": "This section addresses the use of forged electronic signatures.",
    "Section 77": "This section outlines compensation and penalties for failing to secure sensitive personal data.",
    "Section 80": "This section empowers police officers to enter, search, and arrest without a warrant under specified circumstances."
}

# Tokenize the legal sections for BM25 search
try:
    tokenized_rules = [word_tokenize(rule.lower()) for rule in it_act_rules]
except Exception as e:
    print(f"Error in tokenization: {e}")
    tokenized_rules = [rule.lower().split() for rule in it_act_rules]

bm25 = BM25Okapi(tokenized_rules)

# Function to compute the average vector for a given text using law2vec
def get_average_vector(text, model):
    tokens = word_tokenize(text.lower())
    valid_tokens = [token for token in tokens if token in model]
    if not valid_tokens:
        vector_size = model.vector_size if model else 300
        return np.zeros(vector_size)
    return np.mean([model[token] for token in valid_tokens], axis=0)

# Pre-compute law2vec embeddings for each IT Act section if law2vec_model is available
if law2vec_model:
    section_embeddings = [get_average_vector(rule, law2vec_model) for rule in it_act_rules]
else:
    section_embeddings = None

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ask")
def ask_question(query: str = Query(..., description="Ask a question about the IT Act, 2000")):
    query = query.strip()
    if not query:
        return {"question": query, "answer": "Please enter a valid legal question."}
    
    # Tokenize the query for BM25 processing
    try:
        query_tokens = word_tokenize(query.lower())
    except Exception as e:
        print(f"Error in query tokenization: {e}")
        query_tokens = query.lower().split()
    
    # BM25 scores for the query against the IT Act sections
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores
    bm25_min = bm25_scores.min()
    bm25_max = bm25_scores.max()
    if bm25_max - bm25_min > 0:
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_norm = bm25_scores

    # Retrieve top match
    top_index = np.argmax(bm25_norm)
    best_match = it_act_rules[top_index]
    
    # Split section details and format response
    section_code, description = best_match.split(':', 1)
    section_code = section_code.strip()
    description = description.strip()
    explanation = section_explanations.get(section_code, "No detailed explanation available.")
    
    # Create descriptive paragraph response
    formatted_answer = (
        f"Under {section_code} of the Information Technology Act, 2000, which deals with {description.lower()}, "
        f"the legislation provides the following provisions: {explanation} "
        f"This legal framework ensures appropriate measures and consequences for related cyber offenses."
    )
    
    return {"question": query, "answer": formatted_answer}