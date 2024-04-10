from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv, find_dotenv
from embeddings import create_embeddings, ask_and_get_answer
from utilities import calculate_embedding_cost
from services.loaders import DocumentLoader
from services.chunkers import Chunker

app = Flask(__name__)
load_dotenv(find_dotenv(), override=True)

# Assuming a simple in-memory storage for demonstration
vector_stores = {}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('./uploads', filename)
        file.save(file_path)

        chunk_size = request.form.get('chunk_size', 512, type=int)
        data = DocumentLoader.load_document(file_path)
        chunks = Chunker.chunk_data(data, chunk_size=chunk_size)
        tokens, embedding_cost = calculate_embedding_cost(chunks)
        vector_store = create_embeddings(chunks)

        # For simplicity, using filename as key
        vector_stores['session_in_memory'] = vector_store
        print(f"file name: {filename}")
        print(vector_stores)

        return jsonify({"message": "File processed successfully", "embedding_cost": embedding_cost}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    # filename = data.get('filename')
    question = data.get('question')
    k = data.get('k', 3)

    vector_store = vector_stores.get('session_in_memory')
    if not vector_store:
        return jsonify({"error": "File not processed or not found"}), 404

    answer = ask_and_get_answer(vector_store, question, k)
    return jsonify({"answer": answer}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
