from flask import Flask, request, jsonify
import json
import os
import requests
from dotenv import load_dotenv
import base64

# ------------------------------
# ENVIRONMENT CONFIG
# ------------------------------

load_dotenv()
PLAN_IDS = {
    "pro": "P-5MS58158LM8045458NE4IBOI",
    "diamond": "P-06A12557DS926960YNE4IBOY"
}

PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")

# Use PayPal SANDBOX for testing
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com"

WEBHOOK_ID = os.environ.get("PAYPAL_WEBHOOK_ID")

app = Flask(__name__)

# ------------------------------
# USER STORAGE (JSON FILE)
# ------------------------------

def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return []

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

# ------------------------------
# PAYPAL AUTH
# ------------------------------

def get_access_token():
    auth = base64.b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_SECRET}".encode()).decode()

    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = "grant_type=client_credentials"

    r = requests.post(f"{PAYPAL_API_BASE}/v1/oauth2/token", headers=headers, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

# ------------------------------
# CREATE SUBSCRIPTION
# ------------------------------

@app.route("/create_subscription", methods=["POST"])
def create_subscription():
    data = request.json
    username = data.get("username")
    tier = data.get("tier")

    if not username or not tier:
        return jsonify({"error": "Missing username or tier"}), 400

    if tier not in PLAN_IDS:
        return jsonify({"error": "Invalid tier"}), 400

    plan_id = PLAN_IDS[tier]

    access_token = get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    body = {
        "plan_id": plan_id,
        "custom_id": username,
        "application_context": {
            "brand_name": "Pyxel Morph",
            "locale": "en-US",
            "user_action": "SUBSCRIBE_NOW",
            "return_url": f"https://pyxel-morph-listener-cf123.herokuapp.com/paypal/success?user={username}",
            "cancel_url": f"https://pyxel-morph-listener-cf123.herokuapp.com/paypal/cancel"
        }
    }

    r = requests.post(
        f"{PAYPAL_API_BASE}/v1/billing/subscriptions",
        headers=headers,
        json=body
    )
    r.raise_for_status()
    subscription = r.json()

    # Extract approval URL
    approval_url = next((link["href"] for link in subscription.get("links", []) if link["rel"] == "approve"), None)
    if not approval_url:
        return jsonify({"error": "No approval URL returned"}), 500

    return jsonify({"approval_url": approval_url})

# ------------------------------
# GET USER STATUS
# ------------------------------

@app.route("/get_status", methods=["GET"])
def get_user_status():
    username = request.args.get("user")
    if not username:
        return jsonify({"error": "missing user"}), 400

    users = load_users()
    for u in users:
        if u["username"] == username:
            return jsonify({"type": u["type"]})

    # Default tier if user not found
    return jsonify({"type": "free"})

# ------------------------------
# WEBHOOK LISTENER
# ------------------------------

@app.route("/paypal/webhook", methods=["POST"])
def paypal_webhook():
    event = request.json
    event_type = event.get("event_type")
    resource = event.get("resource", {})
    plan_id = resource.get("plan_id")
    custom_id = resource.get("custom_id")

    print("Received event:", event_type, "for user:", custom_id)

    if event_type in ["BILLING.SUBSCRIPTION.ACTIVATED", "PAYMENT.SALE.COMPLETED"]:
        # Map plan_id to tier
        new_tier = None
        for tier, pid in PLAN_IDS.items():
            if pid == plan_id:
                new_tier = tier
                break

        if not new_tier:
            print("Unknown plan_id:", plan_id)
            return jsonify({"status": "error", "message": "Unknown plan"}), 200

        # Update users.json
        users = load_users()
        found = False
        for u in users:
            if u["username"] == custom_id:
                u["type"] = new_tier
                found = True
                break

        if found:
            save_users(users)
            print(f"User {custom_id} upgraded to {new_tier}")
            return jsonify({"status": "success"}), 200
        else:
            print(f"User {custom_id} not found in database")
            return jsonify({"status": "error", "message": "User not found"}), 404

    elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
        print(f"Subscription cancelled for {custom_id} — implement downgrade logic here.")

    return jsonify({"status": "ok"}), 200

# ------------------------------
# RUN APP
# ------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
