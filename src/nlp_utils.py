import re

intent_patterns = {
    "ORDER_TRACKING": {
        "keywords": ["track", "order", "status", "where", "delivery", "when"],
        "priority": 3,
    },
    "RETURN_REQUEST": {
        "keywords": ["return", "refund", "exchange", "defective", "broken", "damaged"],
        "priority": 3,
    },
    "PAYMENT_ISSUE": {
        "keywords": ["payment", "card", "billing", "charged", "refund"],
        "priority": 2,
    },
    "ACCOUNT_HELP": {
        "keywords": ["password", "reset", "login", "account", "email"],
        "priority": 2,
    },
    "GENERAL_FAQ": {
        "keywords": ["how", "what", "why", "info", "information"],
        "priority": 1,
    },
}

def extract_intent(message: str):
    text = message.lower()
    matches = []

    for intent, cfg in intent_patterns.items():
        count = sum(1 for k in cfg["keywords"] if k in text)
        if count > 0:
            confidence = count / len(cfg["keywords"])
            matches.append(
                {"intent": intent, "confidence": confidence, "priority": cfg["priority"]}
            )

    if not matches:
        return {"intent": "GENERAL_FAQ", "confidence": 0.3, "priority": 0}

    matches.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
    return matches[0]

def extract_order_id(message: str):
    match = re.search(r"ORD\d+", message.upper())
    return match.group(0) if match else None
