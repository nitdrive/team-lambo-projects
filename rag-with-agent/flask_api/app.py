from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv, find_dotenv
from embeddings import create_embeddings, ask_and_get_answer, get_db
from utilities import calculate_embedding_cost
from services.loaders import DocumentLoader
from services.chunkers import Chunker
from services.enhancers import MetadataEnhancer
from services.lookup import WebsiteLookUpByFileName
from agent_chain import execute_chain

app = Flask(__name__, static_folder='static')
load_dotenv(find_dotenv(), override=True)

# Assuming a simple in-memory storage for demonstration
vector_stores = {}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search_in_documents')
def search_documents():
    return render_template('index4.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            print("Error: No file part")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            print("Error: No selected file")
            return jsonify({"error": "No selected file"}), 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('./uploads', filename)
            file.save(file_path)

            chunk_size = request.form.get('chunk_size', 512, type=int)
            data = DocumentLoader.load_document(file_path)
            associated_websites = WebsiteLookUpByFileName.get_websites_by_file_name(filename)
            print(f"Associated website: {associated_websites}")
            enhanced_data = MetadataEnhancer.add_metadata_to_documents(
                docs=data,
                metadata={
                    'source_website': associated_websites
                }
            )

            # print(enhanced_data)
            chunks = Chunker.chunk_data(enhanced_data, chunk_size=chunk_size)
            # tokens, embedding_cost = calculate_embedding_cost(chunks)
            create_embeddings(chunks)
            print(f"Uploaded file name: {filename}")

            return jsonify({"message": "File processed successfully"}), 200
    except Exception as e:
        print("Exception occurred")
        print(e)


@app.route('/ask_about_documents', methods=['POST'])
def ask_about_documents():
    data = request.json
    question = data.get('question')
    k = data.get('k', 3)

    print("Asking about Documents uploaded")

    question = f"You are a good assistant, who answers investment related questions and have expertise on Vanguard. Please validate your answers for relevance and correctness before responding'. Question: {question}"
    answer = ask_and_get_answer(get_db(), question, k, 'DocumentSearch')
    return jsonify({"answer": answer}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')

    # question = f"You are a good assistant. You are an expert on stocks, ETFs and Index Funds and company financials. Question: {question}"
    question = f"You are a good assistant. You are an expert on investing. Include any Sources available in your response. Question: {question}"
    answer = execute_chain(question)
    return jsonify({"answer": answer}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
