from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

class QuoteRetriever:
    def __init__(self, model_path, df):
        self.model = SentenceTransformer(model_path)
        self.df = df
        self.index = self.build_index()

    def build_index(self):
        embeddings = self.model.encode(self.df['combined_text'].tolist(), convert_to_tensor=False)
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        self.embeddings = embeddings
        return index

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query)
        scores, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.df.iloc[i]['combined_text'] for i in indices[0]]