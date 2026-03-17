from Get_Chunks import get_chunks
from embedding_utils import get_embeddings, get_query_embedding
from faiss_utils import add_to_faiss, search_faiss
from config import get_gemini_client
from llm_utils import call_llm

def main():
    ''' 
    This is the main function that runs the entire RAG pipeline.
    1. It gets chunks from get_chunks.py
    2. It generates embeddings for those chunks using embedding_utils.py
    3. It adds those embeddings to a FAISS index using faiss_utils.py   
    4. It takes a user query, generates an embedding for it, and searches the FAISS index for similar chunks.
    5. It retrieves the relevant chunks based on the search results and calls the LLM to generate an answer based on the retrieved chunks and the user query.
    '''
    #step 1: get the chunks from knowldege base text file
    chunks = get_chunks()

    #step 2: get the embedding of chunks
    embeddings_list = get_embeddings(chunks)

    #step 3: add the embedding to faiss index
    index = add_to_faiss(embeddings_list)

    #step 4: take user query 
    user_query = input("Please enter your query: ")

    #step 5: get the embeddings of user query
    query_embedding = get_query_embedding(user_query)

    #step 6: search the faiss index for similar chunks
    distances, indices = search_faiss(index, query_embedding)

    #step 7: retrieve the relevant chunks based on search results
    relevant_chunks = [chunks[idx] for idx in indices[0]]

    #step 8: call the LLM to generate response based on RAG
    response = call_llm(relevant_chunks, user_query)

    print("\nResponse from LLM:\n", response)

if __name__ == "__main__":
    main()
    