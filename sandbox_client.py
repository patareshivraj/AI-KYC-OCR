import os
from typing import Any, Dict, Optional

import httpx
import requests


class SandboxClient:
    """Client for Sandbox.co.in KYC APIs.

    Provides both synchronous methods (kept for compatibility) and
    async methods (suffixed _async) for use inside FastAPI route handlers.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("SANDBOX_BASE_URL", "https://api.sandbox.co.in")
        self.api_key = os.getenv("SANDBOX_API_KEY", "")
        self.auth_token = os.getenv("SANDBOX_AUTH_TOKEN", "")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.auth_token)

    def _headers(self) -> Dict[str, str]:
        if not self.is_configured():
            raise ValueError("Sandbox credentials are not configured.")
        return {
            "accept": "application/json",
            "authorization": self.auth_token,
            "x-api-key": self.api_key,
            "content-type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    # ---------------------------------------------------------------------------
    # Synchronous methods (use only outside async context)
    # ---------------------------------------------------------------------------

    def verify_pan(
        self,
        pan: str,
        name_as_per_pan: Optional[str] = None,
        date_of_birth: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not pan:
            raise ValueError("PAN is required.")
        payload: Dict[str, Any] = {"pan": pan}
        if name_as_per_pan:
            payload["name_as_per_pan"] = name_as_per_pan
        if date_of_birth:
            payload["date_of_birth"] = date_of_birth

        resp = requests.post(
            self._url("/kyc/pan/verify"),
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox PAN verify failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()

    def pan_aadhaar_link_status(self, pan: str) -> Dict[str, Any]:
        if not pan:
            raise ValueError("PAN is required.")
        resp = requests.post(
            self._url("/kyc/pan-aadhaar-link-status"),
            headers=self._headers(),
            json={"pan": pan},
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox PAN-Aadhaar link status failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()

    def send_aadhaar_otp(
        self, aadhaar_number: str, consent: str, reason: str
    ) -> Dict[str, Any]:
        if not aadhaar_number:
            raise ValueError("Aadhaar number is required.")
        resp = requests.post(
            self._url("/kyc/aadhaar/okyc/otp"),
            headers=self._headers(),
            json={"aadhaar_number": aadhaar_number, "consent": consent, "reason": reason},
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox Aadhaar OTP send failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()

    def verify_aadhaar_otp(self, reference_id: str, otp: str) -> Dict[str, Any]:
        if not reference_id or not otp:
            raise ValueError("reference_id and otp are required.")
        resp = requests.post(
            self._url("/kyc/aadhaar/okyc/otp/verify"),
            headers=self._headers(),
            json={"reference_id": reference_id, "otp": otp},
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox Aadhaar OTP verify failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()

    # ---------------------------------------------------------------------------
    # Async methods — use these inside FastAPI route handlers
    # ---------------------------------------------------------------------------

    async def verify_pan_async(
        self,
        pan: str,
        name_as_per_pan: Optional[str] = None,
        date_of_birth: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not pan:
            raise ValueError("PAN is required.")
        payload: Dict[str, Any] = {"pan": pan}
        if name_as_per_pan:
            payload["name_as_per_pan"] = name_as_per_pan
        if date_of_birth:
            payload["date_of_birth"] = date_of_birth

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self._url("/kyc/pan/verify"),
                headers=self._headers(),
                json=payload,
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox PAN verify failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()

    async def pan_aadhaar_link_status_async(self, pan: str) -> Dict[str, Any]:
        if not pan:
            raise ValueError("PAN is required.")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self._url("/kyc/pan-aadhaar-link-status"),
                headers=self._headers(),
                json={"pan": pan},
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox PAN-Aadhaar link status failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()

    async def send_aadhaar_otp_async(
        self, aadhaar_number: str, consent: str, reason: str
    ) -> Dict[str, Any]:
        if not aadhaar_number:
            raise ValueError("Aadhaar number is required.")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self._url("/kyc/aadhaar/okyc/otp"),
                headers=self._headers(),
                json={"aadhaar_number": aadhaar_number, "consent": consent, "reason": reason},
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox Aadhaar OTP send failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()

    async def verify_aadhaar_otp_async(
        self, reference_id: str, otp: str
    ) -> Dict[str, Any]:
        if not reference_id or not otp:
            raise ValueError("reference_id and otp are required.")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self._url("/kyc/aadhaar/okyc/otp/verify"),
                headers=self._headers(),
                json={"reference_id": reference_id, "otp": otp},
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Sandbox Aadhaar OTP verify failed: {resp.status_code} {resp.text[:500]}"
            )
        return resp.json()
