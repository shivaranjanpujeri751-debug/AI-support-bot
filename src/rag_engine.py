import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Static FAQ data (same you used in notebook)
_FAQ_RAW = {
    "order_tracking": [
        {
            "question": "How do I track my order?",
            "answer": "You can track your order using your order ID on our website under 'My Orders' > 'Track'.",
        },
        {
            "question": "What is my order status?",
            "answer": "Orders usually take 3-5 business days to deliver, and we email you at each step.",
        },
    ],
    "returns": [
        {
            "question": "How do I return a product?",
            "answer": "You can initiate a return within 30 days from 'My Orders' by selecting the item and choosing 'Return'.",
        },
        {
            "question": "What is your return policy?",
            "answer": "Most items are returnable within 30 days; original shipping fees are refunded only for our errors.",
        },
    ],
    "payment": [
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept major credit/debit cards, UPI, net banking, and popular wallets.",
        },
        {
            "question": "Is my payment secure?",
            "answer": "All payments are protected with SSL encryption and processed through PCI‑DSS compliant gateways.",
        },
    ],
    "account": [
        {
            "question": "How do I reset my password?",
            "answer": "Use 'Forgot password' on the login page and follow the link sent to your email.",
        }
    ],
}

_rows = []
for cat, items in _FAQ_RAW.items():
    for obj in items:
        _rows.append({"category": cat, "question": obj["question"], "answer": obj["answer"]})

_FAQ_DF = pd.DataFrame(_rows)

_vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
_FAQ_VECTORS = _vectorizer.fit_transform(_FAQ_DF["question"].tolist())

def retrieve_faqs(query: str, top_k: int = 2):
    if not query.strip():
        return []

    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _FAQ_VECTORS)[0]
    idxs = np.argsort(sims)[-top_k:][::-1]

    results = []
    for idx in idxs:
        if sims[idx] < 0.1:
            continue
        row = _FAQ_DF.iloc[idx]
        results.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "similarity": float(sims[idx]),
                "category": row["category"],
            }
        )
    return results
