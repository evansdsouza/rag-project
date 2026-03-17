def get_chunks():
    with open("knowledge_base_rag.txt", "r", encoding="utf-8") as f:
        content = f.read()

    chunks = content.split("###")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks

# print(get_chunks())
# chunks = get_chunks()
# print(f"Total chunks: {len(chunks)}")
# for i in chunks:
#     print(i)
#     print("---")