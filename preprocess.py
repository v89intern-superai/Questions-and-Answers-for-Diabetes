import json
import numpy as np
import time
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore():
    # Load data
    with open("data/papers_data.json", "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    docs = []
    for paper in papers_data:
        for section in list(paper["sections"].keys()):
            docs.append(paper["sections"][section])

    website_data = np.load("data/cleaned_diabetes_data.npy", allow_pickle=True)
    for webdata in website_data:
        docs.append(webdata["text"])

    # Convert to Document objects
    docss = [Document(page_content=doc) for doc in docs]

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docss)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="./models/all-mpnet-base-v2")

    # Initialize vector store
    vector_store = Chroma(
        collection_name="diabetes_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    # Add documents in batches
    batch_size = 50
    for i in tqdm(range(0, len(all_splits), batch_size)):
        batch = all_splits[i:i+batch_size]
        _ = vector_store.add_documents(documents=batch)
        time.sleep(0.5)

    print("✅ Vector store built and persisted at ./chroma_langchain_db")

if __name__ == "__main__":
    build_vectorstore()
