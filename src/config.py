from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "gemini-2.5-flash-lite"
LLM_MAX_NEW_TOKENS: int = 100
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03
#LLM_QUESTION: str = "Who is Newton?"
LLM_SYSTEM_PROMPT: str = '''You are an expert on Roborock robotic vacuum manuals. 
Use only the retrieved text chunks from the manuals to answer questions. 
Do not hallucinate information or invent instructions not present in the manuals. 
If the user asks for a list of features, models, or error codes, return them as a plain list. 
If the user asks for instructions, give step-by-step guidance exactly as described in the manual. 
If the information is not in the retrieved chunks, respond: 
“I could not find that information in the Roborock manuals.”
You may reference multiple chunks if necessary to give a complete answer.'''


# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 5
# The size of each text chunk in tokens
CHUNK_SIZE: int = 500
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 100

# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/robotic_vacuums"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"

# --- RERANKER SETTINGS ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 2   
RETRIEVER_K = 10   

