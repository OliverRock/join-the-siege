from typing import Dict

from pydantic import BaseModel


class TextClassificationResult(BaseModel):
    """Pydantic model for classifier prediction results"""

    category: str
    confidence: float
    all_scores: Dict[str, float]
    model_version: str
    inference_time_sec: float
    document_length: int
    document_id: str