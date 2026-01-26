# Mental Health RAG API

A FastAPI-based mental health chatbot powered by **Hybrid RAG** (Retrieval Augmented Generation). It combines **Local Embeddings (HuggingFace)** for privacy and infinite search, with **Google's Gemini 2.5 Flash** for empathetic reasoning. It includes a robust Crisis Detection system with multi-channel alerts.

## Images

<img width="650" height="1300" alt="Screenshot 2026-01-26 152546" src="https://github.com/user-attachments/assets/de060f7a-279c-4c4e-91bc-e160a83c4764" /> 
<img width="650" height="1300" alt="Twilio Trigger" src="https://github.com/user-attachments/assets/f275f8ff-4c24-402f-ab78-0a32662d7bc8" /><img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/049fbf62-3a67-45b3-a04c-e464dc476338" />
<img width="650" height="1300" alt="telegram trigger" src="https://github.com/user-attachments/assets/5d6b80ea-cc46-43c8-a618-6f07d2d651ee" /><img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/32155403-3385-4d8e-a372-f58a04ea745d" />





## Features

- **Hybrid RAG Architecture**: 
  - **Search**: Uses `all-MiniLM-L6-v2` locally on CPU (FAST, FREE, NO Rate Limits).
  - **Answer**: Uses Google Gemini 2.5 Flash for high-quality generation.
- **Smart Crisis Detection**:
  - **Sliding Window Analysis**: Detects crisis phrases hidden in long messages.
  - **Regex Fallback**: Instant detection for critical keywords.
  - **Zero-Latency**: Background thread initialization prevents server lag.
- **Multi-Channel Alerts**:
  - **Primary**: SMS via Twilio (High Reliability).
  - **Fallback**: Telegram Bot (Automatic backup if SMS fails).
- **Persistent Chat**: History stored in MongoDB Atlas.

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (Local/CPU)
- **Vector DB**: Qdrant (Cloud)
- **Database**: MongoDB (History)
- **Alerts**: Twilio (SMS), Telegram Bot API

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Environment**:
   Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
   Fill in your keys in `.env` (Gemini, Mongo, Qdrant, Twilio, Telegram).

3. **Run the server**:
   The first run will download the embedding model (20MB) automatically.
```bash
uvicorn main:app --reload
```

## API Endpoint

### POST `/chat`

**Parameters**:
- `query` (str): User message (e.g., "I feel anxious")
- `session_id` (str): Unique session identifier
- `file` (optional): PDF file upload for document analysis
- `user_name` (str): User's name
- `user_location` (str): User's location (for emergency alerts)

**Response**:
```json
{
  "reply": "I understand you're feeling anxious. Have you tried box breathing?...",
  "reasoning": "User expressed anxiety. RAG retrieved calming techniques.",
  "citations": ["Clinical Handbook v1", "Safety Protocol"]
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key (LLM) |
| `MONGO_URI` | MongoDB Atlas connection string |
| `QDRANT_URL` | Qdrant cloud URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `HELPLINE_PHONE_NUMBER` | Emergency contact number for SMS alerts |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_MESSAGING_SERVICE_SID` | Twilio Messaging Service SID |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot Token |
| `TELEGRAM_CHAT_ID` | Telegram Chat ID for alerts |

## Getting API Keys

- **GEMINI_API_KEY**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **MONGO_URI**: [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- **QDRANT**: [Qdrant Cloud](https://cloud.qdrant.io/)
- **TWILIO**: [Twilio Console](https://console.twilio.com/)
- **TELEGRAM**: [Telegram BotFather](https://t.me/botfather)
