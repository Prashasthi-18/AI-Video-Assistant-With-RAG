import os
import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
    return _embeddings


def collection_name_for(transcript: str) -> str:
    # One collection per transcript, so chunks from different videos never mix
    # and re-processing the same video reuses the existing embeddings.
    return "video_" + hashlib.md5(transcript.encode("utf-8")).hexdigest()[:16]


def load_vector_store(transcript: str) -> Chroma:
    return Chroma(
        collection_name=collection_name_for(transcript),
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )


def build_vector_store(transcript: str) -> Chroma:
    vector_store = load_vector_store(transcript)

    if vector_store._collection.count() > 0:
        print("Vector store already exists for this transcript — reusing it.")
        return vector_store

    print("Building vector store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store.add_documents(docs)
    print(f"Stored {len(docs)} chunks.")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
