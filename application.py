from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, TFAutoModel
import faiss
import json
import numpy as np
import tensorflow as tf
from scripts.nlp_translate import preprocess_text as preprocess_text_indonesian
import time
import os

application = Flask(__name__, static_folder="static")

tokenizer = None
model = None
index = None
corpus = None
corpus_embeddings = None

def initialize_components():
    global tokenizer, model, index, corpus, corpus_embeddings

    if tokenizer is None:
        print("Memuat tokenizer dari Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("XzyanQi/flaskpython")

    if model is None:
        print("Memuat model dari Hugging Face...")
        model = TFAutoModel.from_pretrained("XzyanQi/flaskpython")

    if index is None:
        print("Memuat FAISS index...")
        index = faiss.read_index("model/mindfulness_index.faiss")
        print("Index FAISS berhasil dimuat.")

    if corpus is None:
        print("Memuat corpus JSON...")
        with open("model/corpus_final.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)

    if corpus_embeddings is None:
        print("Memuat context_embeddings.npy...")
        corpus_embeddings = np.load("model/context_embeddings.npy")

def get_embedding(text):
    clean_text = preprocess_text_indonesian(text)
    inputs = tokenizer(clean_text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)[0]
    vec = tf.reduce_mean(outputs, axis=1)
    return vec[0].numpy()

@application.route("/search", methods=["POST"])
def search():
    overall_start_time = time.time()
    data = request.get_json()
    query = data.get("text", "")
    top_k = data.get("top_k", 3)

    print(f"\nMenerima query: '{query}'")

    query_vec = get_embedding(query).reshape(1, -1).astype("float32")

    distances, indices = index.search(query_vec, top_k)

    results = []
    if indices.size > 0 and len(indices[0]) > 0:
        for i in indices[0]:
            if 0 <= i < len(corpus):
                doc = corpus[i]
                results.append(doc.get("response_to_display", "Format tidak sesuai."))
            else:
                results.append("Kesalahan mengambil detail dokumen.")
    else:
        results.append("Tidak ada jawaban relevan ditemukan.")

    print(f"Total waktu proses: {(time.time() - overall_start_time) * 1000:.2f} ms")
    return jsonify({"query": query, "results": results})

@application.route("/")
def root():
    return send_from_directory("static", "index.html")

@application.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

# Inisialisasi semua komponen sebelum menerima request
initialize_components()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    application.run(host="0.0.0.0", port=port)
