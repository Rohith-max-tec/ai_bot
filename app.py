from flask import Flask, render_template, request, jsonify
import faiss
import numpy as np
import pymongo
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://rh0665971:q7DFaWad4RKQRiWg@cluster0.gusg4.mongodb.net/?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client ['swaraksha']
context_collection = db['context']

# Load the QA model
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global FAISS index variable
faiss_index = None
context_texts = []

def embed_text(text):
    """Convert input text into a numerical vector."""
    return embedding_model.encode(text).astype(np.float32)

def load_knowledge_from_mongo():
    """Load knowledge from MongoDB and vectorize it."""
    global faiss_index, context_texts
    context_texts = [doc['text'] for doc in context_collection.find({}, {'_id': 0, 'text': 1})]
    
    if not context_texts:
        faiss_index = None
        return
    
    knowledge_vectors = np.array([embed_text(context) for context in context_texts], dtype=np.float32)
    vector_dim = knowledge_vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(vector_dim)
    faiss_index.add(knowledge_vectors)
    print("FAISS index updated with new knowledge.")

def get_best_context(question):
    """Retrieve the most relevant context using FAISS."""
    if faiss_index is None or not context_texts:
        return None
    
    question_vector = np.array([embed_text(question)], dtype=np.float32)
    D, I = faiss_index.search(question_vector, k=1)
    best_index = I[0][0]
    
    if D[0][0] > 1.0:
        return None  
    
    return context_texts[best_index] if best_index >= 0 else None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """Handle user questions and return responses."""
    data = request.json
    question = data.get("question", "").strip()

    simple_responses = {
        "hi": "hi hello",
        "hello": "hi hello",
        "hey": "hi hello",
        "how are you?": "I'm just a bot, but I'm here to help!"
    }

    if question.lower() in simple_responses:
        return jsonify({"answer": simple_responses[question.lower()]})

    best_context = get_best_context(question)
    if not best_context:
        return jsonify({"answer": "I'm sorry, I don't have info regarding that."})

    answer = qa_pipeline(question=question, context=best_context)
    return jsonify({"answer": answer["answer"] if answer["score"] > 0.3 else "I'm sorry, I don't have info regarding that."})

@app.route("/teach", methods=["POST"])
def teach():
    """Teach the bot new knowledge."""
    data = request.json
    new_context = data.get("context", "").strip()

    if new_context and not context_collection.find_one({"text": new_context}):
        context_collection.insert_one({"text": new_context})
        load_knowledge_from_mongo()
        return jsonify({"message": "Knowledge updated!"})
    
    return jsonify({"message": "⚠️ This information already exists or is empty."})

if __name__ == "__main__":
    load_knowledge_from_mongo()
    app.run(debug=True)
