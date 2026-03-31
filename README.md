# AI KYC OCR

A deterministic local pre-validation API and OCR engine for processing KYC documents (PAN, Aadhaar, Bank Statements). Built on FastAPI. Uses Groq for primary vision OCR with automatic local fallback to Ollama. 

No AI/ML is used for the decision engine—only deterministic rules.

## Features
- **OCR Fallback Support:** Tries Groq first, drops to local Ollama on failure/timeout.
- **Deterministic Validation:** 10-step rule engine for matching names, formats, and dates using RapidFuzz.
- **Fraud Detection:** Detects string mismatches, Verhoeff checksum failures, and format anomalies.
- **Official Verifications:** Fully integrated with Sandbox.co.in APIs.

## Setup

1. Install dependencies:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or source venv/bin/activate # Unix
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
```
Ensure you set `GROQ_API_KEY` and `OLLAMA_BASE_URL`.

3. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Swagger UI available at `http://localhost:8000/docs`.

## Logic Flow
1. **Extraction:** Base64 encodes files, sends to LLM.
2. **Normalization:** Trims whitespace, cleans dates, parses JSON.
3. **Scoring:** `kyc_validator` tests names against strict thresholds.
4. **Output:** Returns JSON with `APPROVE`, `REVIEW`, or `REJECT` classification and complete audit log.
