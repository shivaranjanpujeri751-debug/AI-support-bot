from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.config import HOST, PORT, DEBUG
from src.chatbot_core import generate_response

app = Flask(__name__)
CORS(app)  # allow JS from browser

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data or {}).get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = generate_response(user_message)

    return jsonify(
        {
            "reply": result["bot_response"],
            "intent": result["intent"],
            "sentiment": result["sentiment"],
            "order_id": result["order_id"],
            "faqs": result["faqs"],
            "needs_escalation": result["needs_escalation"],
            "timestamp": result["timestamp"],
        }
    )

if __name__ == "__main__":
    print(f"Starting Customer Support Bot on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
