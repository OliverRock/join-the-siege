import json
import os

import pandas as pd

from classifier import DocumentClassifier


def read_data() -> pd.DataFrame:
    # Path to the single JSON file
    json_file = "/Users/oliverrock/personal-programming/join-the-siege/src/classifier/data/samples.json"  # noqa

    # Read the JSON file into a DataFrame
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        combined_df = pd.DataFrame(data)

    # Rename 'label' to 'category' if that column exists
    if "label" in combined_df.columns:
        combined_df = combined_df.rename(columns={"label": "category"})

    return combined_df


def train_classifier():
    """Train classifier on industry-specific training data"""
    # Load training data (assuming CSV format)
    training_data = read_data()

    # Extract texts and categories
    texts = training_data["text"].tolist()
    categories = training_data["category"].tolist()

    # Create and train classifier
    classifier = DocumentClassifier()
    classifier.train(texts, categories)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    classifier.save_model("models/document_classifier.pkl")

    print(
        f"Model trained on {len(texts)} documents. {len(set(categories))} categories."
    )


if __name__ == "__main__":
    train_classifier()
