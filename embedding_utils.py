from config import get_gemini_client, EMBEDDING_MODEL

client = get_gemini_client()

def get_embeddings(documents):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents= documents
    )
    embeddings_list = [e.values for e in result.embeddings]
    return embeddings_list

def get_query_embedding(query):
    result_query = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents= query
    )
    embeddings_query = [e.values for e in result_query.embeddings]
    return embeddings_query