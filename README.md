# Mental Health RAG API
A FastAPI-based mental health chatbot powered by RAG (Retrieval Augmented Generation) using Google's Gemini AI, Qdrant vector database, and MongoDB for chat history.
## Images

<img width="518" height="350" alt="result1" src="https://github.com/user-attachments/assets/8bf7d7da-6416-4a84-8438-cdc7a334655b" /> 
<img width="518" height="350" alt="result2" src="https://github.com/user-attachments/assets/5c1259be-4a1c-4908-9fd1-a3ad6beab08c" />




## Features

- **Conversational AI**: Empathetic mental health support with context-aware responses
- **RAG Integration**: Retrieves relevant information from Qdrant vector store
- **PDF Upload**: Upload and analyze mental health documents
- **Chat History**: Persistent session management via MongoDB
- **Crisis Detection**: Built-in safety protocols for emergency situations

## Tech Stack

- FastAPI, Uvicorn
- LangChain (Google Generative AI, Qdrant)
- MongoDB (chat history)
- Qdrant (vector database)
- Google Gemini AI (LLM + Embeddings)

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Create `.env` file**:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
QDRANT_URL=https://your-cluster.qdrant.cloud
QDRANT_API_KEY=your_qdrant_api_key_here
```

3. **Run the server**:
```bash
uvicorn main:app --reload
```

## API Endpoint

### POST `/chat`

**Parameters**:
- `query` (str): User message
- `session_id` (str): Unique session identifier
- `file` (optional): PDF file upload

**Response**:
```json
{
  "reply": "AI response",
  "reasoning": "Internal thinking process",
  "citations": ["source1", "source2"]
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `MONGO_URI` | MongoDB Atlas connection string |
| `QDRANT_URL` | Qdrant cloud URL |
| `QDRANT_API_KEY` | Qdrant API key |

## Getting API Keys

- **GEMINI_API_KEY**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **MONGO_URI**: [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- **QDRANT**: [Qdrant Cloud](https://cloud.qdrant.io/)


