"""Explain linear TF-IDF model predictions using influential terms."""

from __future__ import annotations

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def top_tfidf_terms(model_pipeline, text: str, top_n: int = 8) -> list[tuple[str, float]]:
    """Return the terms that most supported the predicted class."""
    vectorizer = model_pipeline.named_steps.get("vectorizer") or model_pipeline.named_steps.get("tfidf")
    classifier = model_pipeline.named_steps.get("model")
    if not isinstance(vectorizer, TfidfVectorizer) or classifier is None or not hasattr(classifier, "coef_"):
        return []

    transformed = vectorizer.transform([text])
    predicted_label = model_pipeline.predict([text])[0]
    classes = list(classifier.classes_)
    class_index = classes.index(predicted_label)
    coefficients = classifier.coef_
    if coefficients.shape[0] == 1 and len(classes) == 2:
        class_weights = coefficients[0] if class_index == 1 else -coefficients[0]
    else:
        class_weights = coefficients[class_index]

    scores = transformed.multiply(class_weights).toarray().ravel()
    if not np.any(scores):
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [
        (str(feature_names[index]), float(scores[index]))
        for index in top_indices
        if scores[index] > 0
    ]
