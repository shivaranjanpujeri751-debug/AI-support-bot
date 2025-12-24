from datetime import datetime

from groq import Groq

from .config import GROQ_API_KEY, LLM_MODEL_NAME
from .nlp_utils import extract_intent, extract_order_id
from .order_db import get_order_details
from .rag_engine import retrieve_faqs
from .sentiment_module import analyze_sentiment

_client = Groq(api_key=GROQ_API_KEY)

def generate_response(user_message: str):
    """
    End‑to‑end chatbot logic used by both Jupyter and Flask.
    Returns a dict with all info required by UI and evaluation.
    """

    intent = extract_intent(user_message)
    order_id = extract_order_id(user_message)
    sentiment = analyze_sentiment(user_message)
    faqs = retrieve_faqs(user_message, top_k=2)

    context_lines = []

    if order_id:
        details = get_order_details(order_id)
        if details:
            context_lines.append(
                f"Order {order_id} status: {details.get('status')} "
                f"for product {details.get('product')}."
            )
        else:
            context_lines.append(
                f"User mentioned order {order_id} but it was not found in the database."
            )

    if faqs:
        context_lines.append("Top related FAQ answer: " + faqs[0]["answer"])

    if sentiment["label"] == "NEGATIVE":
        context_lines.append(
            "User seems upset; respond with extra empathy and offer escalation to a human agent."
        )

    context_text = "\n".join(context_lines) if context_lines else "No extra context."

    system_prompt = (
        "You are a helpful e‑commerce customer support chatbot. "
        "Always be concise, polite, and clear. When you are not sure, "
        "suggest connecting with a human support agent. "
        "Use any provided context faithfully."
    )

    user_prompt = f"""Customer message: {user_message}

Context:
{context_text}

Intent you detected: {intent['intent']}

If an order ID is mentioned, update the customer about its status.
If sentiment is negative, acknowledge the issue and be empathetic.
"""

    completion = _client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.7,
    )  # API usage pattern per Groq docs. [web:12][web:13]

    bot_message = completion.choices[0].message.content

    needs_escalation = (
        sentiment["label"] == "NEGATIVE" or intent["confidence"] < 0.3
    )

    return {
        "user_message": user_message,
        "bot_response": bot_message,
        "intent": intent,
        "sentiment": sentiment,
        "order_id": order_id,
        "faqs": faqs,
        "needs_escalation": needs_escalation,
        "timestamp": datetime.now().isoformat(),
    }
