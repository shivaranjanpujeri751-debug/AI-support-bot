# 🧠 Customer Support Chatbot (LLM + RAG)

Intelligent e‑commerce **customer support chatbot** built with **Groq’s `llama-3.1-8b-instant`**, **Retrieval‑Augmented Generation (RAG)**, and **Flask**.  
The bot can answer FAQs, track orders, process return requests, analyze sentiment, and suggest escalation to a human agent.

> Built by **Shivaranjan** · Bengaluru, Karnataka, India

---

## 📌 Features

- **Intent recognition** – Classifies queries as order tracking, returns, payment issues, account help, or general FAQs.
- **Order tracking** – Reads mock order database and returns real status for IDs like `ORD001`, `ORD002`, etc.
- **RAG (Retrieval‑Augmented Generation)** – Uses TF‑IDF similarity over an FAQ knowledge base to ground the LLM and reduce hallucinations.
- **Sentiment analysis** – DistilBERT model detects positive/negative sentiment and flags unhappy customers.
- **Smart escalation** – If sentiment is negative or intent confidence is low, the bot suggests connecting to a human agent.
- **Fast LLM responses** – Uses Groq’s `llama-3.1-8b-instant` for sub‑second chat completions.
- **Web interface** – Simple HTML/JS chat UI with dark theme, showing metadata (intent, sentiment, order ID, escalation).
- **REST API** – `/chat` endpoint for programmatic access (can be integrated into any frontend).

---

## 🏗️ Architecture Overview

High‑level design:

Browser (HTML/CSS/JS)
│
▼ POST /chat (JSON)
Flask Backend (app.py)
│
▼
Chatbot Core (chatbot_core.py)
├─ NLP utilities (intent, order ID)
├─ Order DB (order status lookup)
├─ RAG engine (FAQ retrieval via TF‑IDF)
├─ Sentiment module (DistilBERT)
└─ Groq LLM (llama-3.1-8b-instant)

Request flow:

1. User sends a message (e.g., `Where is my order ORD002?`) from the browser.
2. Flask `/chat` endpoint receives JSON `{ "message": "..." }`.
3. `chatbot_core.generate_response`:
   - Detects **intent** and **order ID**.
   - Retrieves matching **FAQ answers** (RAG).
   - Fetches **order details** from mock DB.
   - Runs **sentiment analysis**.
   - Builds a context prompt and calls **Groq LLM**.
4. LLM generates a natural reply grounded in the retrieved context.
5. JSON response (reply + metadata) is returned to the browser and displayed in the chat UI.

---

## 🧰 Tech Stack

- **Language**: Python 3.9.x
- **Backend**: Flask 2.3
- **LLM Inference**: Groq API – `llama-3.1-8b-instant` model
- **NLP & ML**:
  - Hugging Face Transformers
  - DistilBERT sentiment model
  - scikit‑learn (TF‑IDF, cosine similarity)
- **Vector/RAG**: TF‑IDF over FAQ Q&A dataset
- **Data**: In‑memory order database (Python dict)
- **Frontend**: HTML, CSS, vanilla JavaScript (simple chat UI)
- **Config & Utils**: `python-dotenv`, `Flask-CORS`, `pandas`, `numpy`

---

## 📂 Project Structure

customer-support-bot/
├── app.py # Flask application (entry point)
├── requirements.txt # Python dependencies
├── .env # Environment variables (GROQ_API_KEY)
├── templates/
│ └── index.html # Chat UI (frontend)
└── src/
├── init.py
├── config.py # Loads .env, model name, Flask config
├── nlp_utils.py # Intent recognition & order ID extraction
├── order_db.py # Mock order database (ORD001, ORD002, ...)
├── rag_engine.py # FAQ knowledge base + TF‑IDF RAG
├── sentiment_module.py # DistilBERT sentiment analysis pipeline
└── chatbot_core.py # Main orchestration + Groq LLM call

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python**: 3.9.x
- **Conda** (recommended) or `venv`
- **Git**
- **Groq account + API key** (free): sign up at https://console.groq.com

### 2. Clone the repository

`git clone https://github.com/<your-username>/customer-support-bot.git
cd customer-support-bot`

### 3. Create and activate a virtual environment (Conda example)

`conda create -n customer-support-bot python=3.9.13 -y`

### 4. Install dependencies

`pip install -r requirements.txt`

### 5. Configure environment variables

Create a `.env` file in the project root:
`GROQ_API_KEY=your_real_groq_api_key_here`

How to get the key:

1. Go to the **Groq Console** → API Keys.
2. Create a key and copy it.
3. Paste it into `.env` as shown above.

### 6. Run the application

From the project root, with the virtual environment active:
`python app.py`

You should see something like:

Starting Customer Support Bot on http://127.0.0.1:5000

### 7. Open the chat UI

- Open a browser and visit: [**http://127.0.0.1:5000**](http://127.0.0.1:5000)
- Try these sample prompts:

`Hi`
`Where is my order ORD002?`
`I want to return my laptop`
`Your product was broken, I am very disappointed`
`How can I reset my password?`
`What payment methods do you accept?`

You should see:

- Bot replies with empathetic, context‑aware responses.
- Metadata line under each bot message, for example:  
  `Intent: ORDER_TRACKING | Sentiment: POSITIVE | Order ID: ORD002 | Escalation suggested`
