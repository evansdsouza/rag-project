from config import get_gemini_client
import numpy as np
import faiss

client = get_gemini_client()

def add_to_faiss(embeddings_list):

    dimensions = len(embeddings_list[0])
    index = faiss.IndexFlatL2(dimensions)
    vectors = np.array(embeddings_list, dtype=np.float32)
    index.add(vectors)
    return index


def search_faiss(index, query_embeddings, k=3):
    query_vector = np.array(query_embeddings, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return distances, indices


