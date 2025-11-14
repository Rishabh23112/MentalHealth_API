import os
import shutil
import uuid
import re
import datetime
import logging
from typing import List, Optional


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from pymongo import MongoClient, errors as mongo_errors
from qdrant_client import QdrantClient
from qdrant_client.http import models


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.docstore.document import Document

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mental Health RAG API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE ---

# MongoDB (Chat History)
try:
    MONGO_URI = os.getenv("MONGO_URI")
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # connection check
    mongo_client.server_info()
    db = mongo_client["mental_health_db"]
    chat_collection = db["chat_sessions"]
    logger.info("✅ Connected to MongoDB Atlas")
except Exception as e:
    logger.error(f"❌ MongoDB Connection Failed: {e}")
    raise e

# Qdrant (RAG Base)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "mental_health_rag"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment variables. Some features may not work.")

#  error handling
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    logger.info("✅ GoogleGenerativeAIEmbeddings initialized")
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=GEMINI_API_KEY)
    logger.info("✅ ChatGoogleGenerativeAI initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}")
    llm = None


def init_knowledge_base():
    """
    Checks if Qdrant collection exists. If not, creates it and seeds it with default safety protocols.
    Handles quota errors gracefully by creating empty collection.
    """
    if not qdrant_client or not embeddings:
        logger.error("❌ Qdrant client or embeddings not initialized. Cannot initialize knowledge base.")
        return None
    
    try:
        # Check if collection exists
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant Collection: {COLLECTION_NAME}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Qdrant(client=qdrant_client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    except Exception as e:
        logger.warning("⚠️ Collection not found. Creating new Qdrant collection...")
        
        seed_docs = [
            Document(
                page_content="EMERGENCY PROTOCOL: If a user expresses intent of suicide, self-harm, or harm to others, IMMEDIATELY stop therapy and provide: National Suicide Prevention Lifeline: 988, Crisis Text Line: Text HOME to 741741.",
                metadata={"source": "Safety Protocol v1"}
            ),
            Document(
                page_content="Technique: Box Breathing. Inhale 4s, Hold 4s, Exhale 4s, Hold 4s. Useful for panic attacks.",
                metadata={"source": "Clinical Handbook"}
            )
        ]
        
        try:
            return Qdrant.from_documents(
                seed_docs,
                embeddings,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME
            )
        except Exception as seed_error:
            if "429" in str(seed_error) or "quota" in str(seed_error).lower():
                logger.warning("⚠️ API quota exceeded. Creating empty collection. Seeds will be added when quota resets.")
                # Create empty collection by creating Qdrant instance without documents
                from qdrant_client.http import models as qdrant_models
                qdrant_client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=768,  # Default embedding size
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"✅ Created empty Qdrant collection: {COLLECTION_NAME}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    return Qdrant(client=qdrant_client, collection_name=COLLECTION_NAME, embeddings=embeddings)
            else:
                logger.error(f"❌ Failed to seed collection: {seed_error}")
                raise seed_error

# Vector Store
try:
    vector_store = init_knowledge_base()
except Exception as e:
    logger.error(f"❌ Failed to initialize Vector Store: {e}")
    vector_store = None

def get_mongo_history(session_id: str, limit: int = 10) -> List:
    """Fetches the last N messages for a specific user session."""
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
    """Appends a message to the user's history in MongoDB."""
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


SYSTEM_PROMPT = """
You are a supportive Mental Health Assistant. Your role is to provide empathetic, evidence-based guidance for a wide range of mental health and wellness topics.

**CRITICAL INSTRUCTION - HONOR USER REQUESTS:**
- If the user asks for "instant solution" or "quick fix" or similar - provide IMMEDIATE, actionable steps they can do RIGHT NOW
- If the user asks for "detailed explanation" - provide comprehensive information
- If the user asks for "exercises" or "techniques" - focus on specific practices
- LISTEN TO WHAT THEY'RE ASKING FOR AND DELIVER EXACTLY THAT

**CORE PRIORITIES:**
1. **Safety First**: If the user mentions self-harm, suicide, or crisis - IMMEDIATELY provide:
   - National Suicide Prevention Lifeline: 988
   - Crisis Text Line: Text HOME to 741741
   - Emergency Services: 911

2. **Understand the User**: Listen to what they're actually asking about:
   - Mental health concerns (anxiety, depression, stress, relationships)
   - Wellness tips (sleep, exercise, mindfulness)
   - Coping strategies
   - General life advice
   - Information and education
   - Quick solutions vs. detailed explanations vs. exercises

3. **Respond Appropriately**: Match your response to THEIR specific request, not a preset template

**YOUR APPROACH:**
- Be warm, empathetic, and non-judgmental
- Ask clarifying questions if needed
- Provide practical, actionable advice
- Use evidence-based approaches when relevant
- Acknowledge the user's feelings
- Avoid assumptions
- **MOST IMPORTANTLY: Give them what they ask for, not what you think they need**

**OUTPUT FORMAT:**
<thinking>
- What is the user SPECIFICALLY asking for?
- Are they in crisis? If yes, provide resources immediately
- Do they want quick tips, detailed info, exercises, or something else?
- What exactly can I deliver to address their request?
</thinking>

<response>
[Your empathetic, helpful response tailored to their SPECIFIC REQUEST - not generic advice]
</response>

**Remember:** Listen to what they're saying, not what you expect them to say. Everyone's situation is unique. Deliver what they ask for.
"""


def get_mongo_history(session_id: str, limit: int = 10) -> List:
    """Fetches the last N messages for a specific user session."""
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
    """Appends a message to the user's history in MongoDB."""
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


# --- API ENDPOINT ---

@app.post("/chat")
async def chat_endpoint(
    query: str,
    session_id: str,
    file: UploadFile = File(None)
):
    """
    Main Chat Endpoint.
    - query: User's text message.
    - session_id: Unique ID for the user (to track history).
    - file: Optional PDF upload.
    """
    try:
        if not llm:
            raise HTTPException(status_code=503, detail="AI service not available. Please ensure GEMINI_API_KEY is set in environment variables.")
        
        # File Processing
        file_context = ""
        if file:
            logger.info(f"Processing file: {file.filename}")
            # Save temp file with proper encoding
            temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
            try:
                with open(temp_filename, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                # Extract Text
                loader = PyPDFLoader(temp_filename)
                pages = loader.load()
                full_text = "\n".join([p.page_content for p in pages])
                
                file_context = f"\n\n[USER UPLOADED FILE CONTENT]:\n{full_text[:50000]}\n"
                
                save_message_to_mongo(session_id, "system", f"User uploaded file: {file.filename}")
            except Exception as e:
                logger.error(f"File parsing error: {e}")
                file_context = "\n[System: Error reading uploaded file]\n"
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        # RAG(Qdrant)
        docs = []
        try:
            docs = vector_store.similarity_search(query, k=3)
            rag_context = "\n".join([f"PROTOCOL: {d.page_content} (Source: {d.metadata.get('source', 'Unknown')})" for d in docs])
        except Exception as e:
            logger.error(f"⚠️ RAG search failed: {e}")
            rag_context = "[System: Unable to retrieve knowledge base context]"

        # History(MongoDB)
        history_msgs = get_mongo_history(session_id, limit=10)
        history_text = ""
        for msg in history_msgs:
            role_name = "User" if isinstance(msg, HumanMessage) else "AI"
            history_text += f"{role_name}: {msg.content}\n"

        
        final_prompt = f"""
        {SYSTEM_PROMPT}
        
        ================================
        MEMORY (Past Conversation):
        {history_text}
        ================================
        
        TRUSTED KNOWLEDGE BASE (RAG):
        {rag_context}
        
        FILE CONTEXT:
        {file_context}
        
        USER INPUT:
        {query}
        """

        
        try:
            response = llm.invoke([HumanMessage(content=final_prompt)])
        except Exception as e:
            logger.error(f"❌ LLM invocation failed: {e}")
            raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")
        
        full_content = response.content

        # Extract thinking and response with fallback
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_content, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', full_content, re.DOTALL)

        if thinking_match:
            ai_thinking = thinking_match.group(1).strip()
        else:
            ai_thinking = "Processed user request and generated response."
        
        # If XML tags not found, use entire response as reply
        if response_match:
            final_reply = response_match.group(1).strip()
        else:
            logger.warning("⚠️ XML tags not found in LLM response. Using full response.")
            final_reply = full_content.strip()

        # Log the thinking process 
        logger.info({ai_thinking})

        
        # Save User Query
        save_message_to_mongo(session_id, "user", query)
        # Save AI Response
        save_message_to_mongo(session_id, "ai", final_reply)

        return {
            "reply": final_reply,
            "reasoning": ai_thinking, # Optional: Remove this line if you don't want to send logic to frontend
            "citations": [d.metadata.get('source') for d in docs]
        }

    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn main:app --reload