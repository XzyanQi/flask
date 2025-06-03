from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer
import faiss
import json
import numpy as np
import tensorflow as tf
from scripts.nlp_translate import preprocess_text as preprocess_text_indonesian
import time
import os
import requests

application = Flask(__name__, static_folder="static")

tokenizer = None
model = None
index = None
corpus = None
corpus_embeddings = None

def download_from_gdrive(file_id, dest_path):
    print("Mengunduh model dari Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Unduhan selesai.")

def initialize_components():
    global tokenizer, model, index, corpus, corpus_embeddings

    if tokenizer is None:
        print("Memuat tokenizer dari lokal HuggingFace model...")
        tokenizer = AutoTokenizer.from_pretrained("model/indobert_local/")

    if model is None:
        print("Memuat model TensorFlow dari .h5...")
        model_path = "model/indobert_local/tf_model.h5"
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            download_from_gdrive("1wBD7t1mRV8ksDQNnlApFs28fhpCUIhyY", model_path)
        model = tf.keras.models.load_model(model_path)
        print("Model berhasil dimuat.")

    if index is None:
        print("Memuat FAISS index...")
        index = faiss.read_index("mindfulness_index.faiss")
        print("Index FAISS berhasil dimuat.")

    if corpus is None:
        print("Memuat corpus...")
        with open("model/corpus_final.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)
        print("Corpus berhasil dimuat.")

    if corpus_embeddings is None:
        print("Memuat context_embeddings.npy...")
        corpus_embeddings = np.load("context_embeddings.npy")
        print("Embedding corpus berhasil dimuat.")

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

    print(f"\n[Python /search] Menerima query: '{query}' pukul {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_embedding_call = time.time()
    query_vec = get_embedding(query).reshape(1, -1).astype("float32")
    end_embedding_call = time.time()
    print(f"   [Python /search] Pembuatan query embedding selesai dalam: {(end_embedding_call - start_embedding_call)*1000:.2f} ms")

    start_faiss = time.time()
    distances, indices = index.search(query_vec, top_k)
    end_faiss = time.time()
    print(f"   [Python /search] Pencarian FAISS selesai dalam: {(end_faiss - start_faiss)*1000:.2f} ms")

    results_texts_to_display = []
    if indices.size > 0 and len(indices[0]) > 0:
        for i in indices[0]:
            if 0 <= i < len(corpus):
                document_object = corpus[i]
                if "response_to_display" in document_object:
                    results_texts_to_display.append(document_object["response_to_display"])
                else:
                    results_texts_to_display.append("Maaf, format data respons tidak sesuai.")
            else:
                results_texts_to_display.append("Maaf, terjadi kesalahan saat mengambil detail dokumen.")
    else:
        results_texts_to_display.append("Maaf, saya tidak menemukan jawaban yang relevan saat ini.")

    overall_end_time = time.time()
    print(f"   [Python /search] Total waktu proses di /search: {(overall_end_time - overall_start_time)*1000:.2f} ms")

    return jsonify({"query": query, "results": results_texts_to_display})

@application.route("/")
def root():
    return send_from_directory("static", "index.html")

@application.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

# Inisialisasi sebelum menerima request
initialize_components()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    application.run(host="0.0.0.0", port=port)
