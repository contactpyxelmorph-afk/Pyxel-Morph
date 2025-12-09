import base64
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
CLIENT_SECRET = os.getenv("PAYPAL_SECRET")

PAYPAL_API = "https://api-m.sandbox.paypal.com"


def get_access_token():
    auth = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64 = base64.b64encode(auth.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = "grant_type=client_credentials"

    resp = requests.post(f"{PAYPAL_API}/v1/oauth2/token",
                         headers=headers, data=data)

    resp.raise_for_status()
    return resp.json()["access_token"]


def create_product(name, description):
    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "name": name,
        "description": description,
        "type": "SERVICE",
        "category": "SOFTWARE"
    }

    resp = requests.post(f"{PAYPAL_API}/v1/catalogs/products",
                         headers=headers, json=payload)

    resp.raise_for_status()
    data = resp.json()
    print(f"[OK] Created product: {name} → {data['id']}")
    return data["id"]


def create_plan(product_id, name, price):
    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "product_id": product_id,
        "name": name,
        "billing_cycles": [
            {
                "frequency": {
                    "interval_unit": "MONTH",
                    "interval_count": 1
                },
                "tenure_type": "REGULAR",
                "sequence": 1,
                "total_cycles": 0,
                "pricing_scheme": {
                    "fixed_price": {
                        "value": f"{price}",
                        "currency_code": "EUR"
                    }
                }
            }
        ],
        "payment_preferences": {
            "auto_bill_outstanding": True,
            "setup_fee_failure_action": "CONTINUE",
            "payment_failure_threshold": 1
        }
    }

    resp = requests.post(f"{PAYPAL_API}/v1/billing/plans",
                         headers=headers, json=payload)

    resp.raise_for_status()
    data = resp.json()
    print(f"[OK] Created plan: {name} → {data['id']}")
    return data["id"]


def main():
    print("=== Creating PayPal Sandbox Products & Plans ===")

    # 1. Create products
    prod_pro = create_product("Pro Subscription", "Monthly Pro Tier")
    prod_diamond = create_product("Diamond Subscription", "Monthly Diamond Tier")

    # 2. Create plans
    plan_pro = create_plan(prod_pro, "Pro Monthly", 0.01)
    plan_diamond = create_plan(prod_diamond, "Diamond Monthly", 15.99)

    print("\n=== DONE ===")
    print("Save these IDs inside your backend:\n")
    print(f"PLAN_PRO = '{plan_pro}'")
    print(f"PLAN_DIAMOND = '{plan_diamond}'")


if __name__ == "__main__":
    main()
