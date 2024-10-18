from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import json
import os
import re

app = FastAPI()

# ====== CORS Middleware Setup ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"],  # Frontend URL; adjust if different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Load or Generate Embeddings ======
model = SentenceTransformer('all-MiniLM-L6-v2')

programs_file = 'programs_with_embeddings.json'
if not os.path.exists(programs_file):
    if not os.path.exists('programs.json'):
        raise FileNotFoundError("The 'programs.json' file is missing.")
    
    with open('programs.json', 'r') as file:
        programs = json.load(file)
    
    for program in programs:
        description = program.get('description', '')
        embedding = model.encode(description)
        program['embedding'] = embedding.tolist()
    
    with open(programs_file, 'w') as file:
        json.dump(programs, file, indent=2)
else:
    with open(programs_file, 'r') as file:
        programs = json.load(file)

# Prepare embeddings matrix for cosine similarity
embeddings_matrix = np.array([program['embedding'] for program in programs]).astype('float32')

embedding_dimension = embeddings_matrix.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(embeddings_matrix)

# ====== Basic Conversational Replies ======
def handle_basic_conversation(query):
    query = query.lower().strip()
    basic_replies = {
        r"\bhi\b": "Hello! How can I help you today?",
        r"\bhello\b": "Hi there! How can I assist you?",
        r"\bwho are you\b": "I'm the ISANS Chatbot, here to help you with program information.",
        r"\byou\b": "I'm here to assist you with information about ISANS programs.",
        r"\bwhat do you do\b": "I provide information on ISANS programs and services.",
        r"\bthanks\b": "You're welcome! Feel free to ask more.",
        r"\bthank you\b": "You're welcome! Feel free to ask more.",
        r"\bhelp\b": "Sure! I can assist with program information and eligibility.",
        r"\bwho created you\b": "I was created to assist clients with information about ISANS programs.",
    }

    for pattern, reply in basic_replies.items():
        if re.search(pattern, query):
            return reply

    return None

# ====== Knowledge-Based Replies (Program Lookup with Enhanced Matching) ======
def get_relevant_programs(query, top_k=5):
    query_words = query.lower().split()
    relevant_programs = []
    seen_programs = set()

    # Stage 1: Match on specific fields (immigrationStatuses, ageGroups, languageLevels)
    for program in programs:
        immigration_statuses = [status.lower() for status in program['eligibilityCriteria'].get('immigrationStatuses', [])]
        language_levels = [level.lower() for level in program['eligibilityCriteria'].get('languageLevels', [])]
        age_groups = [age.lower() for age in program['eligibilityCriteria'].get('ageGroups', [])]

        # Corrected Matching Logic
        if any(word in status for status in immigration_statuses for word in query_words):
            if program['id'] not in seen_programs:
                relevant_programs.append(program)
                seen_programs.add(program['id'])

        elif any(word in language_levels for word in query_words) or \
             any(word in age_groups for word in query_words):
            if program['id'] not in seen_programs:
                relevant_programs.append(program)
                seen_programs.add(program['id'])

    # If relevant programs were found based on eligibility, return them
    if relevant_programs:
        return relevant_programs

    # Stage 2: Fallback to general search in program name and description
    for program in programs:
        program_name = program['name'].lower()
        program_description = program['description'].lower()

        if any(word in program_name for word in query_words) or \
           any(word in program_description for word in query_words):
            if program['id'] not in seen_programs:
                relevant_programs.append(program)
                seen_programs.add(program['id'])

    # Stage 3: Embedding-Based Similarity Search (if no results from eligibility or fallback search)
    if not relevant_programs:
        search_k = min(top_k, len(programs))
        query_embedding = model.encode(query).reshape(1, -1).astype('float32')
        distances, indices = index.search(query_embedding, search_k)

        for idx in indices[0]:
            program = programs[idx]
            if program['id'] not in seen_programs:
                relevant_programs.append(program)
                seen_programs.add(program['id'])

    return relevant_programs


# ====== Generate Chatbot Response ======
def generate_response(query):
    basic_reply = handle_basic_conversation(query)
    if basic_reply:
        return basic_reply

    # Retrieve programs based on the refined search logic
    relevant_programs = get_relevant_programs(query)
    
    if relevant_programs:
        response = "Here are some programs that might be helpful:\n"
        for idx, program in enumerate(relevant_programs, 1):
            response += f"{idx}. {program['name']}: {program['description']}\n"
    else:
        response = "Sorry, I couldn't find any programs matching your query."
    
    return response


# ====== API Endpoint ======
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_query = data.get('query', '').strip()

    if not user_query:
        return JSONResponse(content={"response": "Please enter a valid query."})

    try:
        response_text = generate_response(user_query)
        return JSONResponse(content={"response": response_text})
    except Exception as e:
        return JSONResponse(content={"response": "Sorry, something went wrong."})
