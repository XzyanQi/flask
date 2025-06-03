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

# Global components
tokenizer = None
model = None
index = None
corpus = None
corpus_embeddings = None

def download_model_from_huggingface():
    url = "https://huggingface.co/XzyanQi/flaskpython/resolve/main/tf_model.h5"
    local_path = "model/indobert_local/tf_model.h5"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 100_000:
        print("⬇ Mengunduh model dari Hugging Face...")
        try:
            r = requests.get(url, stream=True)
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(" Unduhan selesai.")
        except Exception as e:
            print(" Gagal mengunduh model:", e)
            raise RuntimeError("Gagal mengunduh model dari Hugging Face.")
    return local_path

def initialize_components():
    global tokenizer, model, index, corpus, corpus_embeddings

    print(" Memuat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("cahya/distilbert-base-indonesian")

    print(" Memuat model .h5 dari Hugging Face...")
    model_path = download_model_from_huggingface()
    try:
        model = tf.keras.models.load_model(model_path)
        print(" Model berhasil dimuat.")
    except Exception as e:
        print(" Gagal memuat model:", e)
        raise RuntimeError("tf_model.h5 tidak valid.")

    print(" Memuat FAISS index...")
    index = faiss.read_index("model/mindfulness_index.faiss")

    print(" Memuat corpus...")
    with open("model/corpus_final.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(" Memuat embeddings...")
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

    print(f"\n Menerima query: '{query}'")

    query_vec = get_embedding(query).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    if indices.size > 0 and len(indices[0]) > 0:
        for i in indices[0]:
            if 0 <= i < len(corpus):
                results.append(corpus[i].get("response_to_display", "Format tidak sesuai."))
            else:
                results.append("Kesalahan mengambil detail dokumen.")
    else:
        results.append("Tidak ada jawaban relevan ditemukan.")

    print(f" Total waktu proses: {(time.time() - overall_start_time) * 1000:.2f} ms")
    return jsonify({"query": query, "results": results})

@application.route("/")
def root():
    return send_from_directory("static", "index.html")

@application.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

# Inisialisasi
initialize_components()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    application.run(host="0.0.0.0", port=port)
