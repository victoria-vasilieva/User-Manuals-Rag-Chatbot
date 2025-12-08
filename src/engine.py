from pathlib import Path
import re
from typing import List, Set

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.readers.file import PDFReader
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SimpleNodeParser

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
    RERANKER_MODEL,
    RERANK_TOP_N,
    RETRIEVER_K,
)

from src.model_loader import (
    get_embedding_model,
    initialise_llm
)

from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


# ---------- METADATA EXTRACTION ----------

def _extract_manual_metadata(text: str, filename: str) -> dict:
    """
    Extract metadata from a Roborock manual PDF.
    Detects which model(s) are mentioned.
    """
    known_models = [
        "Roborock Qrevo Pro",
        "Roborock S8 MaxV Ultra",
        "Roborock S7 Pro Ultra"
    ]

    # Detect models mentioned in this manual
    models_in_text = [model for model in known_models if model in text]

    # Fallback: if none found, use filename as model
    if not models_in_text:
        models_in_text = [filename.replace(".pdf", "")]

    return {
        "filename": filename,
        "models": models_in_text,
        "manual_id": filename.replace(".pdf", "").lower()
    }


# ---------- INDEX CREATION ----------

def _create_new_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from manual PDFs."""

    print("Creating new vector store from manuals in:", DATA_PATH)

    pdf_files = list(Path(DATA_PATH).glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {DATA_PATH}")

    documents: List[Document] = []
    unique_models: Set[str] = set()

    for pdf_file in pdf_files:
        chunks = PDFReader().load_data(str(pdf_file))
        # Merge all chunks text to detect models per manual
        combined_text = " ".join(c.text for c in chunks)
        metadata = _extract_manual_metadata(combined_text, pdf_file.name)
        unique_models.update(metadata["models"])

        for chunk in chunks:
            doc = Document(
                text=chunk.text,
                metadata=metadata
            )
            documents.append(doc)

    # ---------- GLOBAL DATASET SUMMARY ----------

    global_doc = Document(
        text=f"""
GLOBAL ROBOROCK MANUAL DATASET SUMMARY

Total manual files: {len(pdf_files)}
Total unique models: {len(unique_models)}

Manual files:
{', '.join(f.name for f in pdf_files)}

Models in dataset:
{', '.join(sorted(unique_models))}
""",
        metadata={
            "type": "global",
            "total_manuals": len(pdf_files),
            "total_models": len(unique_models),
            "models": sorted(unique_models)
        }
    )

    documents.append(global_doc)

    print(f"Loaded {len(documents)} documents including DATASET SUMMARY.")
    print(f"Detected {len(unique_models)} unique models.")

    # ---------- CHUNKING ----------

    #splitter = SentenceSplitter(
    #    chunk_size=CHUNK_SIZE,
    #    chunk_overlap=CHUNK_OVERLAP
    #)
    parser = SimpleNodeParser.from_defaults(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separator="\n\n"   # paragraph-based splitting
    )
    index = VectorStoreIndex.from_documents(
        documents,
        #transformations=[splitter],
        transformations=[parser],
        embed_model=embed_model
    )

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())

    print("Vector store created and saved at:", VECTOR_STORE_PATH)
    return index


# ---------- LOAD OR CREATE VECTOR STORE ----------

def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """Load existing vector store or create it if missing."""

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store...")
        storage_context = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )

    print("Vector store not found. Creating new one...")
    return _create_new_vector_store(embed_model)


# ---------- CHAT ENGINE ----------

def get_chat_engine(llm: GoogleGenAI, embed_model: HuggingFaceEmbedding) -> BaseChatEngine:
    """Initialise RAG chat engine with reranking."""

    vector_index = get_vector_store(embed_model)

    memory = ChatMemoryBuffer.from_defaults(token_limit=CHAT_MEMORY_TOKEN_LIMIT)

    # ---------- RETRIEVER ----------
    retriever = vector_index.as_retriever(
        similarity_top_k=RETRIEVER_K
    )

    # ---------- RERANKER ----------
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANK_TOP_N
    )

    # ---------- CHAT ENGINE ----------
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        node_postprocessors=[reranker],
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT,
    )

    return chat_engine





# ---------- MAIN CHAT LOOP ----------

def main_chat_loop() -> None:
    """Run Roborock manual chatbot."""
    
    print("--- Initialising models ---")
    llm = initialise_llm()
    embed_model = get_embedding_model()

    chat_engine = get_chat_engine(
        llm=llm,
        embed_model=embed_model
    )

    print("--- Roborock RAG Chatbot Ready ---")
    chat_engine.chat_repl()


if __name__ == "__main__":
    main_chat_loop()
