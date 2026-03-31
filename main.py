import os
import re
import io
import json
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

try:
    import fitz
except ImportError:
    fitz = None

from kyc_validator import KYCValidator
from sandbox_client import SandboxClient

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kyc-local-validator")

APP_NAME = "Local KYC Pre-Validation API"
APP_VERSION = "2.0.0"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:latest")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "auto").lower()
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "15"))

# API security & Rate limiting
API_KEY = os.getenv("API_KEY", "")
ENABLE_RATE_LIMIT = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
RATE_LIMIT_STR = f"{MAX_REQUESTS_PER_MINUTE}/minute"

# CORS — set ALLOWED_ORIGINS in .env as comma-separated list
# Always include the local API server so Swagger UI works
_env_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
ALLOWED_ORIGINS = list({
    *_env_origins,
    "http://127.0.0.1:8000",
    "http://localhost:8000",
})

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
SUPPORTED_PDF_TYPES = {"application/pdf"}

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    enabled=ENABLE_RATE_LIMIT,
    default_limits=[RATE_LIMIT_STR]
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="OCR + local KYC pre-validation with optional Sandbox official verification.",
    swagger_ui_parameters={"requestTimeout": 120000},  # 120s timeout for slow Ollama responses
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Accept", "Authorization"],
)

validator = KYCValidator()
sandbox_client = SandboxClient()     # one consistent name used everywhere

# ---------------------------------------------------------------------------
# Optional API key guard
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: Optional[str] = Security(api_key_header)) -> None:
    """Enforce API key if API_KEY env var is set."""
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def mask_aadhaar(aadhaar: Optional[str]) -> Optional[str]:
    """Return only last 4 digits, e.g. XXXXXXXX1234."""
    if not aadhaar:
        return None
    clean = re.sub(r"[^0-9]", "", aadhaar)
    if len(clean) >= 4:
        return "X" * (len(clean) - 4) + clean[-4:]
    return "X" * len(clean)


def mask_sensitive(doc_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask Aadhaar number in any data dict before including in responses."""
    if not data:
        return data
    masked = dict(data)
    if doc_type == "aadhaar" and masked.get("aadhaar"):
        masked["aadhaar"] = mask_aadhaar(str(masked["aadhaar"]))
    return masked


def ensure_file_size(upload: UploadFile, content: bytes) -> None:
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.2f} MB. Max allowed is {MAX_FILE_SIZE_MB} MB.",
        )


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value or None


def normalize_pan(pan: Optional[str]) -> Optional[str]:
    if not pan:
        return None
    pan = re.sub(r"[^A-Za-z0-9]", "", pan).upper()
    return pan or None


def normalize_aadhaar(aadhaar: Optional[str]) -> Optional[str]:
    if not aadhaar:
        return None
    aadhaar = aadhaar.replace(" ", "").replace("-", "").upper()
    aadhaar = re.sub(r"[^0-9X]", "", aadhaar)
    return aadhaar or None


def normalize_ifsc(ifsc: Optional[str]) -> Optional[str]:
    if not ifsc:
        return None
    ifsc = re.sub(r"[^A-Za-z0-9]", "", ifsc).upper()
    return ifsc or None


def normalize_account_number(account_number: Optional[str]) -> Optional[str]:
    if not account_number:
        return None
    return re.sub(r"[^0-9]", "", account_number) or None


def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = clean_text(name)
    if name:
        name = re.sub(r"[^A-Za-z.\s]", "", name)
        name = re.sub(r"\s+", " ", name).strip().upper()
    return name or None


def normalize_dob(dob: Optional[str]) -> Optional[str]:
    if not dob:
        return None
    dob = clean_text(dob)
    if not dob:
        return None

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(dob, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    if re.fullmatch(r"(19|20)\d{2}", dob):
        return dob

    return dob


def normalize_document_fields(doc_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data or {})

    for key in ("name", "full_name", "father_name", "customer_name", "account_holder_name"):
        if key in normalized:
            normalized[key] = normalize_name(normalized.get(key))

    if "dob" in normalized:
        normalized["dob"] = normalize_dob(normalized.get("dob"))
    if "pan" in normalized:
        normalized["pan"] = normalize_pan(normalized.get("pan"))
    if "aadhaar" in normalized:
        normalized["aadhaar"] = normalize_aadhaar(normalized.get("aadhaar"))
    if "ifsc" in normalized:
        normalized["ifsc"] = normalize_ifsc(normalized.get("ifsc"))
    if "account_number" in normalized:
        normalized["account_number"] = normalize_account_number(normalized.get("account_number"))

    normalized["document_type"] = doc_type
    return normalized


# ---------------------------------------------------------------------------
# Image / PDF utilities
# ---------------------------------------------------------------------------

def pil_from_bytes(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content)).convert("RGB")


def pdf_to_images(content: bytes, max_pages: int = 3) -> List[Image.Image]:
    if fitz is None:
        raise HTTPException(
            status_code=500,
            detail="PDF support is unavailable. Install PyMuPDF.",
        )
    pdf = fitz.open(stream=content, filetype="pdf")
    images: List[Image.Image] = []
    try:
        for i in range(min(len(pdf), max_pages)):
            page = pdf.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
    finally:
        pdf.close()          # always release the file descriptor
    return images


def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def file_to_images(upload: UploadFile, content: bytes, doc_type: str) -> List[Image.Image]:
    content_type = (upload.content_type or "").lower()

    if content_type in SUPPORTED_IMAGE_TYPES:
        return [pil_from_bytes(content)]

    if content_type in SUPPORTED_PDF_TYPES:
        max_pages = 3 if doc_type == "bank" else 1
        return pdf_to_images(content, max_pages=max_pages)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {upload.content_type}. Allowed: JPEG, PNG, WEBP, PDF.",
    )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def safe_json_loads(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    candidate = extract_json_block(raw_text) or raw_text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=502,
                detail=f"OCR provider returned invalid JSON: {str(e)}",
            )


# ---------------------------------------------------------------------------
# OCR prompts
# ---------------------------------------------------------------------------

def build_prompt(doc_type: str) -> str:
    common = (
        "You are an OCR extraction engine. "
        "Return ONLY valid JSON. "
        "Do not include markdown, explanations, or extra text. "
        "Extract visible document fields conservatively. "
        "If a field is missing, set it to null."
    )

    prompts: Dict[str, str] = {
        "pan": """
Extract PAN card fields into this JSON schema:
{
  "document_type": "pan",
  "name": null,
  "father_name": null,
  "dob": null,
  "pan": null
}
""",
        "aadhaar": """
Extract Aadhaar card fields into this JSON schema:
{
  "document_type": "aadhaar",
  "name": null,
  "dob": null,
  "gender": null,
  "aadhaar": null,
  "address": null,
  "is_masked": null
}
If Aadhaar number appears masked, preserve X characters and set is_masked=true.
""",
        "bank": """
Extract bank statement/passbook fields into this JSON schema:
{
  "document_type": "bank",
  "customer_name": null,
  "account_number": null,
  "ifsc": null,
  "bank_name": null,
  "branch": null,
  "statement_period": null
}
""",
    }

    return common + "\n" + prompts[doc_type].strip()


# ---------------------------------------------------------------------------
# Vision provider calls — fully async with httpx (non-blocking)
# ---------------------------------------------------------------------------

async def call_groq_vision(base64_images: List[str], prompt: str) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured.")

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_b64 in base64_images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Groq request failed: {resp.status_code} {resp.text[:500]}",
        )

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return safe_json_loads(text)


async def call_ollama_vision(base64_images: List[str], prompt: str) -> Dict[str, Any]:
    ollama_endpoint = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": base64_images,
        "stream": False,
    }
    logger.info(f"Calling Ollama at {ollama_endpoint} with model '{OLLAMA_MODEL}'")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(ollama_endpoint, json=payload)
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Ollama connection refused at {ollama_endpoint}. Is Ollama running? Error: {e}")
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=502, detail=f"Ollama request timed out after 120s. Error: {e}")

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {resp.status_code} {resp.text[:500]}",
        )

    data = resp.json()
    return safe_json_loads(data.get("response", ""))


async def run_ocr(
    doc_type: str, images: List[Image.Image], provider: str
) -> Tuple[Dict[str, Any], str]:
    """Execute OCR using the requested or default provider, with Groq -> Ollama fallback logic."""
    prompt = build_prompt(doc_type)
    base64_images = [pil_to_base64(img) for img in images]
    
    # Standardize provider to lowercase
    provider = (provider or DEFAULT_PROVIDER).lower()

    # Case 1: Force Ollama
    if provider == "ollama":
        return await call_ollama_vision(base64_images, prompt), "ollama"

    # Case 2: Force Groq (with NO fallback)
    if provider == "groq" and GROQ_API_KEY:
        return await call_groq_vision(base64_images, prompt), "groq"

    # Case 3: Auto / Default / Groq-with-fallback (User policy: groq first, then ollama)
    # We try Groq if the key is present. If it fails or is missing, we drop to Ollama.
    if GROQ_API_KEY:
        try:
            logger.info(f"Attempting Groq vision OCR for {doc_type}...")
            res = await call_groq_vision(base64_images, prompt)
            return res, "groq"
        except Exception as e:
            logger.warning(f"Groq OCR failed for {doc_type}, falling back to Ollama: {e}")
    else:
        logger.info(f"Groq API key not found. Proceeding with Ollama fallback for {doc_type}.")

    # Final Fallback to Ollama
    try:
        logger.info(f"Using Ollama fallback for {doc_type} — model: '{OLLAMA_MODEL}' at {OLLAMA_URL}")
        res = await call_ollama_vision(base64_images, prompt)
        return res, "ollama"
    except HTTPException as e:
        detail = e.detail
        logger.error(f"Ollama OCR failed for {doc_type}: {detail}")
        raise HTTPException(
            status_code=502,
            detail=f"Both OCR providers failed. Last error: {detail}"
        )
    except Exception as e:
        logger.error(f"Ollama OCR unexpected error for {doc_type}: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Both OCR providers failed. Unexpected error: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def build_local_validation_response(
    doc_type: str,
    raw_ocr: Dict[str, Any],
    normalized: Dict[str, Any],
) -> Dict[str, Any]:
    validation = validator.validate_document(doc_type, normalized)
    return {
        "document_type": doc_type,
        "ocr_status": "completed",
        "ocr_data": mask_sensitive(doc_type, raw_ocr),
        "normalized_data": mask_sensitive(doc_type, dict(normalized)),
        "local_validation": validation,
        "official_verification_status": "not_performed",
        "timestamp": now_iso(),
    }


async def _process_single_upload(
    doc_type: str,
    upload: Optional[UploadFile],
    provider: str,
) -> Optional[Dict[str, Any]]:
    if upload is None:
        return None

    content = await upload.read()
    ensure_file_size(upload, content)

    images = file_to_images(upload, content, doc_type)
    raw, used_provider = await run_ocr(doc_type, images, provider)
    normalized = normalize_document_fields(doc_type, raw)
    validation = validator.validate_document(doc_type, normalized)

    return {
        "provider_used": used_provider,
        "ocr_data": mask_sensitive(doc_type, raw),
        "normalized_data": mask_sensitive(doc_type, dict(normalized)),
        "local_validation": validation,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "mode": "local_pre_validation_with_optional_official_verification",
        "docs": "/docs",
        "timestamp": now_iso(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "groq_configured": bool(GROQ_API_KEY),
        "ollama_url": OLLAMA_URL,
        "default_provider": DEFAULT_PROVIDER,
        "sandbox_configured": sandbox_client.is_configured(),
        "timestamp": now_iso(),
    }


@app.post("/ocr/pan")
@limiter.limit(RATE_LIMIT_STR)
async def ocr_pan(
    request: Request,
    file: UploadFile = File(...),
    provider: str = Form(DEFAULT_PROVIDER),
    _: None = Security(verify_api_key),
):
    content = await file.read()
    ensure_file_size(file, content)
    images = file_to_images(file, content, "pan")
    raw, _ = await run_ocr("pan", images, provider)
    normalized = normalize_document_fields("pan", raw)
    return build_local_validation_response("pan", raw, normalized)


@app.post("/ocr/aadhaar")
@limiter.limit(RATE_LIMIT_STR)
async def ocr_aadhaar(
    request: Request,
    file: UploadFile = File(...),
    provider: str = Form(DEFAULT_PROVIDER),
    _: None = Security(verify_api_key),
):
    content = await file.read()
    ensure_file_size(file, content)
    images = file_to_images(file, content, "aadhaar")
    raw, _ = await run_ocr("aadhaar", images, provider)
    normalized = normalize_document_fields("aadhaar", raw)
    return build_local_validation_response("aadhaar", raw, normalized)


@app.post("/ocr/bank")
@limiter.limit(RATE_LIMIT_STR)
async def ocr_bank(
    request: Request,
    file: UploadFile = File(...),
    provider: str = Form(DEFAULT_PROVIDER),
    _: None = Security(verify_api_key),
):
    is_pdf = (
        (file.filename or "").lower().endswith(".pdf")
        or file.content_type == "application/pdf"
    )
    if not is_pdf:
        raise HTTPException(
            status_code=400, detail="Bank Statement MUST be a PDF document."
        )
    content = await file.read()
    ensure_file_size(file, content)
    images = file_to_images(file, content, "bank")
    raw, _ = await run_ocr("bank", images, provider)
    normalized = normalize_document_fields("bank", raw)
    return build_local_validation_response("bank", raw, normalized)


@app.post("/kyc/compare")
@limiter.limit(RATE_LIMIT_STR)
async def compare_kyc_documents(
    request: Request,
    payload: Dict[str, Any],
    _: None = Security(verify_api_key),
):
    pan = normalize_document_fields("pan", payload["pan"]) if payload.get("pan") else None
    aadhaar = normalize_document_fields("aadhaar", payload["aadhaar"]) if payload.get("aadhaar") else None
    bank = normalize_document_fields("bank", payload["bank"]) if payload.get("bank") else None

    result = validator.compare_documents(pan=pan, aadhaar=aadhaar, bank=bank)

    return {
        "comparison_status": "completed",
        "normalized_inputs": {
            "pan": pan,
            "aadhaar": mask_sensitive("aadhaar", aadhaar) if aadhaar else None,
            "bank": bank,
        },
        "comparison_result": result,
        "official_verification_status": "not_performed",
        "timestamp": now_iso(),
    }


@app.post("/kyc/full-check")
@limiter.limit(RATE_LIMIT_STR)
async def full_kyc_check(
    request: Request,
    payload: Dict[str, Any],
    _: None = Security(verify_api_key),
):
    pan = normalize_document_fields("pan", payload["pan"]) if payload.get("pan") else None
    aadhaar = normalize_document_fields("aadhaar", payload["aadhaar"]) if payload.get("aadhaar") else None
    bank = normalize_document_fields("bank", payload["bank"]) if payload.get("bank") else None

    kyc_result = validator.run_pipeline(
        {"pan": pan, "aadhaar": aadhaar, "statement": bank}
    )
    return kyc_result


@app.post("/kyc/upload-and-check-all")
@limiter.limit(RATE_LIMIT_STR)
async def upload_and_check_all(
    request: Request,
    pan_file: Optional[UploadFile] = File(None),
    aadhaar_file: Optional[UploadFile] = File(None),
    bank_file: Optional[UploadFile] = File(None),
    provider: str = Form(DEFAULT_PROVIDER),
    _: None = Security(verify_api_key),
):
    if not any([pan_file, aadhaar_file, bank_file]):
        raise HTTPException(
            status_code=400,
            detail="At least one of pan_file, aadhaar_file, or bank_file must be provided.",
        )

    pan_result = await _process_single_upload("pan", pan_file, provider)
    aadhaar_result = await _process_single_upload("aadhaar", aadhaar_file, provider)
    bank_result = await _process_single_upload("bank", bank_file, provider)

    per_doc: Dict[str, Any] = {}
    sandbox_pan_res = None

    if pan_result:
        per_doc["pan"] = pan_result["local_validation"]
        pan_data = pan_result["normalized_data"]
        if sandbox_client.is_configured() and pan_data.get("pan"):
            try:
                sandbox_pan_res = await sandbox_client.verify_pan_async(
                    pan=pan_data["pan"],
                    name_as_per_pan=pan_data.get("name"),
                    date_of_birth=pan_data.get("dob"),
                )
            except Exception as e:
                logger.error(f"Sandbox upload-check failed: {e}")

    if aadhaar_result:
        per_doc["aadhaar"] = aadhaar_result["local_validation"]
    if bank_result:
        per_doc["bank"] = bank_result["local_validation"]

    comparison = validator.compare_documents(
        pan=pan_result["normalized_data"] if pan_result else None,
        aadhaar=aadhaar_result["normalized_data"] if aadhaar_result else None,
        bank=bank_result["normalized_data"] if bank_result else None,
        sandbox_pan_res=sandbox_pan_res,
    )
    decision = validator.final_decision(per_doc=per_doc, comparison=comparison)

    return {
        "status": "completed",
        "mode": "upload_and_check_all",
        "documents": {
            "pan": pan_result,
            "aadhaar": aadhaar_result,
            "bank": bank_result,
        },
        "sandbox_verification": sandbox_pan_res,
        "cross_document_comparison": comparison,
        "decision": decision,
        "timestamp": now_iso(),
    }


# ---------------------------------------------------------------------------
# Sandbox official verification endpoints
# ---------------------------------------------------------------------------

@app.post("/kyc/official-verify/pan")
@limiter.limit(RATE_LIMIT_STR)
async def official_verify_pan(
    request: Request,
    payload: Dict[str, Any],
    _: None = Security(verify_api_key),
):
    pan = normalize_pan(payload.get("pan"))
    name = normalize_name(payload.get("name") or payload.get("name_as_per_pan"))
    dob = normalize_dob(payload.get("dob") or payload.get("date_of_birth"))

    if not pan:
        raise HTTPException(status_code=400, detail="PAN is required.")
    if not sandbox_client.is_configured():
        raise HTTPException(status_code=503, detail="Sandbox credentials are not configured.")

    try:
        sandbox_result = await sandbox_client.verify_pan_async(
            pan=pan, name_as_per_pan=name, date_of_birth=dob
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "status": "completed",
        "verification_type": "official_pan_verification",
        "input": {"pan": pan, "name": name, "dob": dob},
        "sandbox_result": sandbox_result,
        "timestamp": now_iso(),
    }


@app.post("/kyc/official-verify/pan-aadhaar-link")
@limiter.limit(RATE_LIMIT_STR)
async def official_verify_pan_aadhaar_link(
    request: Request,
    payload: Dict[str, Any],
    _: None = Security(verify_api_key),
):
    pan = normalize_pan(payload.get("pan"))

    if not pan:
        raise HTTPException(status_code=400, detail="PAN is required.")
    if not sandbox_client.is_configured():
        raise HTTPException(status_code=503, detail="Sandbox credentials are not configured.")

    try:
        sandbox_result = await sandbox_client.pan_aadhaar_link_status_async(pan=pan)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "status": "completed",
        "verification_type": "official_pan_aadhaar_link_status",
        "input": {"pan": pan},
        "sandbox_result": sandbox_result,
        "timestamp": now_iso(),
    }


@app.post("/kyc/official-verify/aadhaar/send-otp")
@limiter.limit(RATE_LIMIT_STR)
async def official_verify_aadhaar_send_otp(
    request: Request,
    payload: Dict[str, Any],
    _: None = Security(verify_api_key),
):
    aadhaar = normalize_aadhaar(payload.get("aadhaar"))
    consent = payload.get("consent", "Y")
    reason = payload.get("reason", "KYC verification")

    if not aadhaar:
        raise HTTPException(status_code=400, detail="Aadhaar is required.")
    if "X" in aadhaar:
        raise HTTPException(
            status_code=400,
            detail="Masked Aadhaar cannot be used for official OTP verification.",
        )
    if not sandbox_client.is_configured():
        raise HTTPException(status_code=503, detail="Sandbox credentials are not configured.")

    try:
        sandbox_result = await sandbox_client.send_aadhaar_otp_async(
            aadhaar_number=aadhaar, consent=consent, reason=reason
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "status": "completed",
        "verification_type": "official_aadhaar_send_otp",
        "input": {
            "aadhaar": mask_aadhaar(aadhaar),   # never echo full Aadhaar back
            "consent": consent,
            "reason": reason,
        },
        "sandbox_result": sandbox_result,
        "timestamp": now_iso(),
    }


@app.post("/kyc/official-verify/aadhaar/verify-otp")
@limiter.limit(RATE_LIMIT_STR)
async def official_verify_aadhaar_verify_otp(
    request: Request,
    payload: Dict[str, Any],
    _: None = Security(verify_api_key),
):
    reference_id = clean_text(payload.get("reference_id"))
    otp = clean_text(payload.get("otp"))

    if not reference_id or not otp:
        raise HTTPException(status_code=400, detail="reference_id and otp are required.")
    if not sandbox_client.is_configured():
        raise HTTPException(status_code=503, detail="Sandbox credentials are not configured.")

    try:
        sandbox_result = await sandbox_client.verify_aadhaar_otp_async(
            reference_id=reference_id, otp=otp
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "status": "completed",
        "verification_type": "official_aadhaar_verify_otp",
        "input": {"reference_id": reference_id},
        "sandbox_result": sandbox_result,
        "timestamp": now_iso(),
    }
