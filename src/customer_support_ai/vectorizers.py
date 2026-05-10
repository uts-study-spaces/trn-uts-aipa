"""Custom Scikit-learn compatible transformers for NLP embeddings."""

from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
from top2vec import Top2Vec


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Represent documents as the average of their Word2Vec word vectors."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        # Tokenize the input text
        sentences = [str(text).split() for text in X]
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        return self

    def transform(self, X):
        if self.model is None:
            raise ValueError("Word2Vec model has not been fitted yet.")

        X_transformed = []
        for text in X:
            words = str(text).split()
            vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            if vectors:
                X_transformed.append(np.mean(vectors, axis=0))
            else:
                X_transformed.append(np.zeros(self.vector_size))
        return np.array(X_transformed)


class Top2VecTransformer(BaseEstimator, TransformerMixin):
    """Represent documents using Top2Vec embeddings."""

    def __init__(
        self, embedding_model: str = "doc2vec", speed: str = "learn", workers: int = 4
    ):
        self.embedding_model = embedding_model
        self.speed = speed
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):

        # Top2Vec needs a list of documents
        documents = [str(text) for text in X]
        self.model = Top2Vec(
            documents,
            embedding_model=self.embedding_model,
            speed=self.speed,
            workers=self.workers,
        )
        return self

    def transform(self, X):
        if self.model is None:
            raise ValueError("Top2Vec model has not been fitted yet.")

        if self.embedding_model == "doc2vec":
            vectors = [self.model.model.infer_vector(str(text).split()) for text in X]
        else:
            vectors = self.model.embed(list(X))

        return np.array(vectors)
