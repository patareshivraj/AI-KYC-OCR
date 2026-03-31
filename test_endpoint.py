import httpx
import json

def test_full_check():
    url = "http://127.0.0.1:8000/kyc/full-check"
    payload = {
        "pan": {
            "pan_no": "ABCDE1234F",
            "name": "RAHUL KUMAR SHARMA",
            "dob": "1990-05-15",
            "father_name": "ANIL SHARMA"
        },
        "aadhaar": {
            "aadhaar_no": "1234 5678 9012",
            "name": "RAHUL KUMAR SHARMA",
            "dob": "1990-05-15"
        },
        "bank": {
            "customer_name": "RAHUL KUMAR SHARMA",
            "account_number": "1234567890123",
            "ifsc": "SBIN0001234",
            "address": "123 Main St, Mumbai"
        }
    }
    
    print(f"Testing {url}...")
    try:
        response = httpx.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print("Response Payload:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_full_check()
