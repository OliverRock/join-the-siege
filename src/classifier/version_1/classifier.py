import os
import pickle
import time
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.models import TextClassificationResult


class DocumentClassifier:
    """Text-based document classifier using TF-IDF and SVM."""

    def __init__(self, model_path: str = None):
        self.model_name = "TF-IDF_SVM_v1"
        # Create classification pipeline with TF-IDF vectorizer and SVM classifier
        self.pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ("classifier", OneVsRestClassifier(LinearSVC())),
            ]
        )

        self.trained = False
        self.categories = []

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(
        self,
        texts: List[str],
        categories: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train the classifier with labeled text data and evaluate its performance

        Args:
        ----
            texts: List of document texts
            categories: List of corresponding categories for each text
            test_size: Fraction of the data to be used for testing (default: 0.2)
            random_state: Random seed for reproducibility

        Returns:
        -------
            Dictionary with training information and evaluation metrics

        """
        if not texts or len(texts) != len(categories):
            raise ValueError("Training data must include matching texts and categories")

        # Track unique categories
        self.categories = sorted(list(set(categories)))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(  # noqa
            texts, categories, test_size=test_size, random_state=random_state
        )

        print(
            f"Training on {len(X_train)} documents, testing on {len(X_test)} documents"
        )

        # Train the model on the training data
        self.pipeline.fit(X_train, y_train)
        self.trained = True

        # Evaluate the model on the test data
        y_pred = self.pipeline.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Generate a detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Count samples per category
        train_category_counts = {cat: y_train.count(cat) for cat in self.categories}
        test_category_counts = {cat: y_test.count(cat) for cat in self.categories}

        # Create a summary of the results
        results = {
            "accuracy": accuracy,
            "num_samples": {
                "total": len(texts),
                "train": len(X_train),
                "test": len(X_test),
            },
            "categories": self.categories,
            "samples_per_category": {
                "train": train_category_counts,
                "test": test_category_counts,
            },
            "classification_report": report,
        }

        # Print a summary of the results
        print("\nTraining Results:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nPerformance by category:")
        for category in self.categories:
            if category in report:
                cat_metrics = report[category]
                print(
                    f"  {category}: Precision={cat_metrics['precision']:.4f}, "
                    f"Recall={cat_metrics['recall']:.4f}, "
                    f"F1-score={cat_metrics['f1-score']:.4f}, "
                    f"Samples={cat_metrics['support']}"
                )

        return results

    def classify(self, text: str, document_id: str = "") -> TextClassificationResult:
        """Classify a document based on its text content,"""
        # Start timing for inference
        start_time = time.time()

        if not self.trained:
            # Handle untrained model with default values
            return TextClassificationResult(
                category="error",
                confidence=0.0,
                all_scores={"error": 1.0},
                model_version=self.model_name,
                inference_time_sec=time.time() - start_time,
                document_length=len(text),
                document_id=document_id,
            )

        if not text.strip():
            # Handle empty text
            return TextClassificationResult(
                category="unknown",
                confidence=0.0,
                all_scores={"unknown": 1.0},
                model_version=self.model_name,
                inference_time_sec=time.time() - start_time,
                document_length=0,
                document_id=document_id,
            )

        # Reshape the text into a list to create a 2D array with a single sample
        text_sample = [
            text
        ]  # This creates a list with one item, which will be transformed correctly

        # Make prediction
        predicted_category = self.pipeline.predict(text_sample)[0]

        # Get confidence scores (decision function values)
        try:
            decision_values = self.pipeline.named_steps["classifier"].decision_function(
                text_sample
            )

            # For multi-class case, we need to find the right score
            if len(decision_values.shape) > 1:
                # Get max confidence value
                category_idx = self.pipeline.classes_.tolist().index(predicted_category)
                confidence = float(decision_values[0][category_idx])

                # Get scores for all categories
                all_scores = {
                    cat: float(score)
                    for cat, score in zip(
                        self.pipeline.classes_, decision_values[0], strict=False
                    )
                }
            else:
                # Binary classification case
                confidence = float(decision_values[0])
                all_scores = {predicted_category: confidence}

        except Exception as e:
            # Fallback if decision_function fails
            print(f"Warning: Could not get decision values: {str(e)}")
            confidence = 1.0  # Default confidence
            all_scores = {predicted_category: confidence}

        # Calculate inference time
        inference_time_sec = time.time() - start_time

        # Return the complete classification result
        return TextClassificationResult(
            category=predicted_category,
            confidence=confidence,
            all_scores=all_scores,
            model_version=self.model_name,
            inference_time_sec=inference_time_sec,
            document_length=len(text),
            document_id=document_id,
        )

    def save_model(self, model_path: str) -> None:
        """Save trained model to disk"""
        if not self.trained:
            raise ValueError("Cannot save untrained model")

        with open(model_path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "categories": self.categories}, f)

    def load_model(self, model_path: str) -> None:
        """Load trained model from disk"""
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.pipeline = data["pipeline"]
            self.categories = data["categories"]
            self.trained = True
