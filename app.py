from flask import Flask, request, jsonify, render_template
import json
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# Load the first dataset
with open("matoshri_database.json", "r", encoding="utf-8") as f1:
    dataset_1 = json.load(f1)

# Load the second dataset
with open("Dataset.json", "r", encoding="utf-8") as f2:
    dataset_2 = json.load(f2)

# Combine the datasets
combined_dataset = dataset_1 + dataset_2

# Pre-load embeddings model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute embeddings for dataset prompts
prompt_response_map = {entry["prompt"]: entry["completion"] for entry in combined_dataset}
prompts = list(prompt_response_map.keys())
prompt_embeddings = embedding_model.encode(prompts, convert_to_tensor=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Preprocess user input
    user_input_cleaned = user_input.strip().lower()

    # Step 1: Fuzzy Matching
    results = process.extract(user_input_cleaned, prompts, scorer=fuzz.partial_ratio, limit=5)
    similarity_threshold = 75  # Configurable similarity threshold
    for match, score, _ in results:
        if score >= similarity_threshold:
            print(f"Fuzzy Match: {match} with Score: {score}")
            return jsonify({"reply": prompt_response_map[match]})

    # Step 2: Semantic Similarity Matching
    user_embedding = embedding_model.encode(user_input_cleaned, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, prompt_embeddings).squeeze(0)
    best_match_idx = torch.argmax(cosine_scores).item()
    best_match_score = cosine_scores[best_match_idx].item()

    semantic_similarity_threshold = 0.8  # Configurable threshold for semantic similarity
    if best_match_score >= semantic_similarity_threshold:
        best_prompt = prompts[best_match_idx]
        print(f"Semantic Match: {best_prompt} with Score: {best_match_score}")
        return jsonify({"reply": prompt_response_map[best_prompt]})

    # Step 3: Keyword-Based Fallback
    user_words = set(user_input_cleaned.split())
    best_match = None
    max_overlap = 0

    for prompt in prompts:
        prompt_words = set(prompt.lower().split())
        overlap = len(user_words & prompt_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = prompt

    if best_match:
        print(f"Keyword-Based Match: {best_match} with Overlap: {max_overlap}")
        return jsonify({"reply": prompt_response_map[best_match]})

    # Step 4: Default Response
    return jsonify({"reply": "Sorry, I don't have an answer for that yet. Please try asking something else."})

if __name__ == "__main__":
    app.run(port=5000)
