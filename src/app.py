from flask import Flask, jsonify, request

from src.classifier.version_1 import DocumentClassifier
from src.models import TextClassificationResult
from src.text_extraction.text_extraction import DocumentTextExtractor

app = Flask(__name__)

extractor = DocumentTextExtractor()
classifier = DocumentClassifier()
classifier.load_model("src/classifier/version_1/models/document_classifier.pkl")


@app.route("/classify_file", methods=["POST"])
def classify_file_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not extractor.allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    document_text = extractor.extract_text(file.read(), file.filename)

    if not document_text:
        return jsonify({"error": "Failed to extract text from the file"}), 500

    file_class: TextClassificationResult = classifier.classify(document_text)

    return jsonify({"file_class": file_class.category}), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True)
