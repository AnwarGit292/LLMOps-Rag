from config import llm, embeddings

# Test LLM
response = llm.invoke("Explain RAG in simple terms.")
print("LLM Response:", response.content)

# Test Embeddings
vector = embeddings.embed_query("What is Retrieval Augmented Generation?")
print("Embedding dimensions:", len(vector))