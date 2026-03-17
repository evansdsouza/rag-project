from config import get_gemini_client, LLM_MODEL

client = get_gemini_client()

def call_llm(context_list,query):

    context = "\n".join(context_list)

    prompt = f"""
    You are a helpful friendly customer service assistant for iNextLabs. You listen to cutomer queries and provide helpful answer or solve their queries. If you don't know the answer, say you don't know. Always be polite and helpful.

    Context: {context}

    query: {query}

    Answer:
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents= prompt
    )

    return response.text

