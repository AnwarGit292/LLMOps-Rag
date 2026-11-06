from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Gemini Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", 
                             temperature=0.2, max_output_tokens=1024)

embeder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

loader = DirectoryLoader(path=Path("./processed/").as_posix(),
                         loader_cls=TextLoader,
                         show_progress=True,)

docs = loader.load()
# print(docs[0].page_content[:500])

chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = chunker.split_documents(docs)

print(f"Number of chunks: {len(chunks)}")

# Create a Chroma vector store

vs = Chroma(collection_name="rag_demo", 
            embedding_function=embeddings, 
            persist_directory=Path("./chroma_db").as_posix())

# Add the chunks to the vector store
vs.add_documents(chunks)

# create a retriever from the vector store
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k":3})


class RAGState(TypedDict):
    query: str
    context: str
    retrieved_docs: list[Document]
    prompt: ChatPromptTemplate
    response: str


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant documents based on the query."""
    query = state["query"]
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    return {"retrieved_docs": retrieved_docs, "context": context}

def augmentation(state: RAGState) -> dict:

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user query
                      based on the given context only. If you do not know the answer
                      say I don't know. Do not add any preamble to the response"""),
        ("human", "context: {context}\n\nquery: {query}")
    ])
    
    return {"prompt": prompt}

def generation(state: RAGState) -> dict:
    
    query = state["query"]
    context = state["context"]
    prompt = state["prompt"]
    
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context":context, "query":query})
    
    return {"response":response}


# create the graph
graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("augmentation", augmentation)
graph_builder.add_node("generation", generation)

graph_builder.add_edge(START,"retrieve")
graph_builder.add_edge("retrieve","augmentation")
graph_builder.add_edge("augmentation","generation")
graph_builder.add_edge("generation",END)

graph = graph_builder.compile()
# graph

result = graph.invoke({"query": "how much fine was imposed on the lawyer who used fake evidences from chatgpt"})
print(result["response"])

result["context"]