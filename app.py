from flask import Flask, request, jsonify, render_template
import json
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import pyttsx3
import threading
import speech_recognition as sr

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

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Set speaking rate
tts_engine.setProperty('voice', 'english')  # Set voice (adjust as needed)

# Speech recognition function
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_for_input():
    print("Listening for input...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            print("Processing audio...")
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError as e:
            return f"Speech Recognition Error: {e}"

def speak_response(response):
    # Block speech synthesis to avoid thread issues
    tts_engine.say(response)
    tts_engine.runAndWait()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Handle input from microphone
    if request.json.get("use_mic"):
        user_input = listen_for_input()
    else:
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
            response = prompt_response_map[match]
            threading.Thread(target=speak_response, args=(response,)).start()
            return jsonify({"reply": response})

    # Step 2: Semantic Similarity Matching
    user_embedding = embedding_model.encode(user_input_cleaned, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, prompt_embeddings).squeeze(0)
    best_match_idx = torch.argmax(cosine_scores).item()
    best_match_score = cosine_scores[best_match_idx].item()

    semantic_similarity_threshold = 0.8  # Configurable threshold for semantic similarity
    if best_match_score >= semantic_similarity_threshold:
        best_prompt = prompts[best_match_idx]
        response = prompt_response_map[best_prompt]
        threading.Thread(target=speak_response, args=(response,)).start()
        return jsonify({"reply": response})

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
        response = prompt_response_map[best_match]
        threading.Thread(target=speak_response, args=(response,)).start()
        return jsonify({"reply": response})

    # Step 4: Default Response
    response = "Sorry, I don't have an answer for that yet. Please try asking something else."
    threading.Thread(target=speak_response, args=(response,)).start()
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(port=5000)
