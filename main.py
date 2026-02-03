import os
import logging
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore 
from langchain_community.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from typing import List
import datetime
import threading


load_dotenv()

# routers
from src.api.routes import router as api_router
from src.crisis.detector import CrisisDetector 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")


app = FastAPI(title="Mental Health RAG API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- DATABASE SETUP ---

# MongoDB
try:
    MONGO_URI = os.getenv("MONGO_URI")
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()
    db = mongo_client["mental_health_db"]
    chat_collection = db["chat_sessions"]
    logger.info("✅ Connected to MongoDB Atlas")
except Exception as e:
    logger.error(f"❌ MongoDB Connection Failed: {e}")
    raise e

# Helpers for MongoDB
def get_mongo_history(session_id: str, limit: int = 10) -> List:
    record = chat_collection.find_one({"session_id": session_id})
    if not record:
        return []
    messages = []
    for msg in record.get("messages", [])[-limit:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
    return messages

def save_message_to_mongo(session_id: str, role: str, content: str):
    message_doc = {
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.utcnow()
    }
    chat_collection.update_one(
        {"session_id": session_id},
        {"$push": {"messages": message_doc}},
        upsert=True
    )

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "mental_health_rag_local"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment variables.")

# Initialize Embeddings Services
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("✅ HuggingFace Local Embeddings initialized (all-MiniLM-L6-v2)")
except Exception as e:
    logger.error(f"❌ Failed to initialize embeddings: {e}")
    embeddings = None

try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    logger.info("✅ Qdrant client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Qdrant client: {e}")
    qdrant_client = None

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
    logger.info("✅ ChatGoogleGenerativeAI initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}")
    llm = None

def init_knowledge_base():
    if not qdrant_client or not embeddings:
        logger.error("❌ Qdrant client or embeddings not initialized. Cannot initialize knowledge base.")
        return None
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant Collection: {COLLECTION_NAME}")
    
        return QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings)
    except Exception:
        logger.warning("⚠️ Collection not found. Creating new Qdrant collection...")
        seed_docs = [
            Document(page_content="EMERGENCY PROTOCOL: If a user expresses intent of suicide, self-harm, or harm to others, IMMEDIATELY stop therapy and provide: Helpline: 911.", metadata={"source": "Safety Protocol v1"}),
            Document(page_content="Technique: Box Breathing. Inhale 4s, Hold 4s, Exhale 4s, Hold 4s. Useful for panic attacks.", metadata={"source": "Clinical Handbook"})
        ]
        try:
            from qdrant_client.http import models as qdrant_models
            
            # Explicitly create collection appropriately for Local Embeddings 
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE)
            )
            
            # Initialize wrapper and add docs 
            qdrant_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings)
            qdrant_store.add_documents(seed_docs)
            return qdrant_store
            
        except Exception as seed_error:
            logger.error(f"❌ Failed to seed collection: {seed_error}")
            return None

try:
    vector_store = init_knowledge_base()
except Exception as e:
    logger.error(f"❌ Failed to initialize Vector Store: {e}")
    vector_store = None

SYSTEM_PROMPT = """You are a Mental Health Support Assistant. Your goal is to listen carefully to the user's specific concern and provide personalized, actionable guidance.

**Important Guidelines:**
1. Focus ONLY on what the user is asking about RIGHT NOW - don't give generic advice
2. Be warm, empathetic, and conversational
3. If they mention crisis/self-harm: Provide 911 (Helpline)
4. Provide practical, specific advice relevant to their situation
5. Each response should be unique based on their actual question

Respond directly and naturally - no need for special formatting. Just have a helpful conversation."""


detector_instance = CrisisDetector(embeddings_model=embeddings)

app.state.llm = llm
app.state.vector_store = vector_store
app.state.save_message = save_message_to_mongo
app.state.get_history = get_mongo_history
app.state.system_prompt = SYSTEM_PROMPT
app.state.detector = detector_instance 

app.include_router(api_router)
