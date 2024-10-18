import json
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load programs
with open('backend/programs.json', 'r') as file:
    programs = json.load(file)

# Generate embeddings
for program in programs:
    description = program['description']
    embedding = model.encode(description)
    program['embedding'] = embedding.tolist()

# Save programs with embeddings
with open('programs_with_embeddings.json', 'w') as file:
    json.dump(programs, file, indent=2)

print("Embeddings generated and saved to programs_with_embeddings.json")
