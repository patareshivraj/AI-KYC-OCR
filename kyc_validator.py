"""
Deterministic KYC Validation, Fraud Detection & Risk Scoring Engine
====================================================================

Design principles:
  • Deterministic > Intelligent
  • Explainable > Complex
  • Safe rejection > False approval
  • Banking-grade reliability

NO AI/ML. NO probabilistic decisions. ONLY deterministic rules.
Every decision is logged with full reasoning.
"""

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Audit Logger — dedicated logger for the validation pipeline
# ---------------------------------------------------------------------------
logger = logging.getLogger("kyc-validator")
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)


class KYCValidator:
    """Deterministic KYC validation engine.

    Field naming convention (post-normalization from main.py):
      PAN:     name, father_name, dob, pan
      Aadhaar: name, dob, gender, aadhaar, address, is_masked
      Bank:    customer_name, account_number, ifsc, bank_name, branch
    """

    # ── Compiled regex patterns ──────────────────────────────────────────────
    PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
    DOB_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")            # Normalized YYYY-MM-DD
    DOB_INPUT_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")       # Raw DD/MM/YYYY

    # ── Identity risk buckets (HIGH score = HIGH risk) ───────────────────────
    IDENTITY_BUCKETS = [
        (95, 0,  "All three names match (≥95) — LOW RISK"),
        (85, 20, "Strong name match (≥85)"),
        (70, 40, "Partial name match (≥70)"),
        (50, 70, "Weak name match (≥50)"),
        (0,  90, "No meaningful name match (<50) — HIGH RISK"),
    ]

    # ── Decision thresholds ──────────────────────────────────────────────────
    APPROVE_THRESHOLD = 30   # 0–30   → APPROVE
    REVIEW_THRESHOLD = 70    # 30–70  → REVIEW
    # 70–100 → REJECT

    # =====================================================================
    #  PUBLIC API — called by main.py
    # =====================================================================

    def validate_document(self, doc_type: str, data: Dict[str, Any], forensics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate a single normalized document. Returns a structured result."""
        if doc_type == "pan":
            res = self._validate_pan(data)
        elif doc_type == "aadhaar":
            res = self._validate_aadhaar(data)
        elif doc_type == "bank":
            res = self._validate_bank(data)
        else:
            res = {
                "document_type": doc_type,
                "is_valid": False,
                "issues": ["Unknown document type"],
                "score": 0,
                "summary": "unknown",
            }
            
        # ── Apply Forensics Penalties ──
        if forensics and forensics.get("is_tampered"):
            res["is_valid"] = False
            res["issues"].append(forensics.get("reason", "Forensics detected image tampering"))
            res["score"] = max(0, res["score"] - 40)
            res["summary"] = self._score_summary(res["score"])
            
        return res

    def compare_documents(
        self,
        pan: Optional[Dict] = None,
        aadhaar: Optional[Dict] = None,
        bank: Optional[Dict] = None,
        sandbox_pan_res: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Cross-document consistency check (used by individual upload routes)."""
        scores: Dict[str, Optional[int]] = {
            "pan_vs_aadhaar_name": None,
            "pan_vs_bank_name": None,
            "aadhaar_vs_bank_name": None,
            "pan_vs_aadhaar_dob": None,
        }
        signals: List[str] = []

        pan_name = self._get_name(pan, "pan") if pan else None
        aadhaar_name = self._get_name(aadhaar, "aadhaar") if aadhaar else None
        bank_name = self._get_name(bank, "bank") if bank else None

        if pan_name and aadhaar_name:
            scores["pan_vs_aadhaar_name"] = fuzz.ratio(pan_name, aadhaar_name)
            if scores["pan_vs_aadhaar_name"] < 70:
                signals.append("Name mismatch between PAN and Aadhaar")

        if pan_name and bank_name:
            scores["pan_vs_bank_name"] = fuzz.ratio(pan_name, bank_name)
            if scores["pan_vs_bank_name"] < 60:
                signals.append("Name mismatch between PAN and bank account")

        if aadhaar_name and bank_name:
            scores["aadhaar_vs_bank_name"] = fuzz.ratio(aadhaar_name, bank_name)
            if scores["aadhaar_vs_bank_name"] < 60:
                signals.append("Name mismatch between Aadhaar and bank account")

        if pan and aadhaar:
            pan_dob = (pan.get("dob") or "").strip()
            aadhaar_dob = (aadhaar.get("dob") or "").strip()
            if pan_dob and aadhaar_dob:
                scores["pan_vs_aadhaar_dob"] = 100 if pan_dob == aadhaar_dob else 0
                if scores["pan_vs_aadhaar_dob"] == 0:
                    signals.append("DOB mismatch between PAN and Aadhaar")

        # Official verification signal
        official_pan_verified = False
        if sandbox_pan_res:
            status = str(sandbox_pan_res.get("data", {}).get("status", "")).upper()
            official_pan_verified = status == "VALID"

        available = [v for v in scores.values() if v is not None]
        avg_score = int(sum(available) / len(available)) if available else 0

        decision = "PASS"
        if signals:
            critical_signals = [s for s in signals if "mismatch" in s.lower()]
            decision = "FAIL" if len(critical_signals) >= 2 else "REVIEW"

        return {
            "scores": scores,
            "signals": signals,
            "average_name_score": avg_score,
            "official_pan_verified": official_pan_verified,
            "decision": decision,
        }

    def final_decision(
        self,
        per_doc: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine per-document and comparison results into one final verdict."""
        doc_scores = [v.get("score", 0) for v in per_doc.values() if isinstance(v, dict)]
        avg_doc_score = int(sum(doc_scores) / len(doc_scores)) if doc_scores else 0

        comp_decision = comparison.get("decision", "REVIEW")
        signals = comparison.get("signals", [])

        doc_issues: List[str] = []
        fraud_penalty = 0
        
        for doc_type, val in per_doc.items():
            if isinstance(val, dict):
                issues = val.get("issues", [])
                if not val.get("is_valid", True) or issues:
                    doc_issues.extend([f"[{doc_type.upper()}] {i}" for i in issues])
                    for i in issues:
                        # Physical Forensics (Heavy Fraud Penalty)
                        if any(kw in i for kw in ["FFT", "Moir", "Analysis", "EXIF", "Photo", "tampered", "Tampered"]):
                            fraud_penalty += 50
                        # Logical Forgery (Checksum failed / Fake ID formats)
                        elif any(kw in i.lower() for kw in ["checksum", "invalid"]):
                            fraud_penalty += 30
                        # Missing crucial data
                        elif "missing" in i.lower() or "not found" in i.lower():
                            fraud_penalty += 10

        all_signals = signals + doc_issues
        
        # Explicit Score Derivations
        identity_score = comparison.get("average_name_score", 0) if comparison else 0
        fraud_score = min(100, fraud_penalty)
        
        # Comprehensive Risk Score (Combines doc errors, identity mismatch, and fraud detection)
        # Formula: (Points lost from docs) + (Points lost from identity match) + (Fraud Score)
        risk_score = min(100, int((100 - avg_doc_score) + (100 - identity_score) + fraud_score))

        if comp_decision == "FAIL" or avg_doc_score < 40 or fraud_score >= 50 or risk_score >= 70:
            verdict = "REJECT"
        elif comp_decision == "REVIEW" or avg_doc_score < 75 or doc_issues or risk_score >= 30:
            verdict = "REVIEW"
        else:
            verdict = "APPROVE"

        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "fraud_score": fraud_score,
            "identity_score": identity_score,
            "average_document_score": avg_doc_score,
            "comparison_decision": comp_decision,
            "signals": all_signals,
            "message": self._verdict_message(verdict, all_signals),
        }

    # =====================================================================
    #  FULL PIPELINE — 10-step deterministic engine
    # =====================================================================

    def run_pipeline(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Full deterministic pipeline: validate → match → resolve → score → decide.

        Input format:
            {
                "pan":       { "name", "pan", "dob", "father_name" },
                "aadhaar":   { "name", "aadhaar", "dob" },
                "statement": { "name"/"customer_name", "account_number", "ifsc", "address" }
            }

        Returns the STEP 9 output exactly as specified.
        """
        audit_log: List[Dict[str, Any]] = []

        # ── STEP 1 — LOG INPUT ──────────────────────────────────────────────
        logger.info("═" * 72)
        logger.info("OCR response received")
        logger.info("Starting validation pipeline")
        logger.info("═" * 72)
        audit_log.append({"step": 1, "action": "PIPELINE_START", "timestamp": self._now()})

        # ── STEP 2 — STRICT FIELD VALIDATION ────────────────────────────────
        logger.info("─── STEP 2: Strict Field Validation ───")
        field_errors = self._validate_fields_strict(ocr_data)

        if field_errors:
            for field, reason in field_errors.items():
                logger.error("Validation failed → %s: %s", field, reason)

            audit_log.append({
                "step": 2,
                "action": "FIELD_VALIDATION",
                "result": "FAILED",
                "errors": field_errors,
                "timestamp": self._now(),
            })

            error_response = {
                "status": "ERROR",
                "message": "Missing or invalid required fields",
                "details": self._group_field_errors(field_errors),
                "audit_log": audit_log,
            }
            logger.error("PIPELINE HALTED — field validation failed")
            return error_response

        logger.info("Field validation: ALL PASSED ✓")
        audit_log.append({
            "step": 2,
            "action": "FIELD_VALIDATION",
            "result": "PASSED",
            "timestamp": self._now(),
        })

        # ── STEP 3 — NAME MATCHING (DETERMINISTIC) ──────────────────────────
        logger.info("─── STEP 3: Name Matching ───")

        pan_data = ocr_data.get("pan") or {}
        aadhaar_data = ocr_data.get("aadhaar") or {}
        statement_data = ocr_data.get("statement") or {}

        pan_name = self._normalize_name(pan_data.get("name"))
        aadhaar_name = self._normalize_name(aadhaar_data.get("name"))
        statement_name = self._normalize_name(
            statement_data.get("customer_name") or statement_data.get("name")
        )

        match_scores = {
            "pan_aadhaar_score": round(fuzz.ratio(pan_name, aadhaar_name), 2),
            "pan_statement_score": round(fuzz.ratio(pan_name, statement_name), 2),
            "aadhaar_statement_score": round(fuzz.ratio(aadhaar_name, statement_name), 2),
        }

        logger.info("Normalized names → PAN: '%s' | Aadhaar: '%s' | Statement: '%s'",
                     pan_name, aadhaar_name, statement_name)
        logger.info("Name match scores → %s", match_scores)

        audit_log.append({
            "step": 3,
            "action": "NAME_MATCHING",
            "normalized_names": {
                "pan": pan_name,
                "aadhaar": aadhaar_name,
                "statement": statement_name,
            },
            "match_scores": match_scores,
            "timestamp": self._now(),
        })

        # ── STEP 4 — NAME RESOLUTION (STRICT ORDERED RULES) ─────────────────
        logger.info("─── STEP 4: Name Resolution ───")
        final_name, resolution_rule = self._resolve_name_strict(
            pan_name, aadhaar_name, statement_name, match_scores
        )
        logger.info("Name resolution completed → '%s' (rule: %s)", final_name, resolution_rule)

        audit_log.append({
            "step": 4,
            "action": "NAME_RESOLUTION",
            "final_name": final_name,
            "rule_applied": resolution_rule,
            "timestamp": self._now(),
        })

        # ── STEP 5 — IDENTITY SCORE (HIGH = RISK) ───────────────────────────
        logger.info("─── STEP 5: Identity Risk Score ───")
        identity_score, identity_reason = self._compute_identity_risk(match_scores)
        logger.info("Identity risk score → %d (%s)", identity_score, identity_reason)

        audit_log.append({
            "step": 5,
            "action": "IDENTITY_SCORING",
            "identity_score": identity_score,
            "reason": identity_reason,
            "timestamp": self._now(),
        })

        # ── STEP 6 — FRAUD DETECTION (HIGH = RISK) ──────────────────────────
        logger.info("─── STEP 6: Fraud Detection ───")
        fraud_score, fraud_signals, fraud_category = self._detect_fraud_signals(
            ocr_data, match_scores
        )
        logger.info("Fraud score calculated → %d (%s)", fraud_score, fraud_category)
        for sig in fraud_signals:
            logger.info("  ⚠ Fraud signal: %s", sig)

        audit_log.append({
            "step": 6,
            "action": "FRAUD_DETECTION",
            "fraud_score": fraud_score,
            "fraud_category": fraud_category,
            "fraud_signals": fraud_signals,
            "timestamp": self._now(),
        })

        # ── STEP 7 — FINAL RISK SCORE ───────────────────────────────────────
        logger.info("─── STEP 7: Final Risk Score ───")
        final_score = round((identity_score * 0.6) + (fraud_score * 0.4), 2)
        logger.info(
            "Final risk score → %.2f  (identity=%d×0.6 + fraud=%d×0.4)",
            final_score, identity_score, fraud_score,
        )

        audit_log.append({
            "step": 7,
            "action": "RISK_SCORING",
            "formula": f"({identity_score} × 0.6) + ({fraud_score} × 0.4)",
            "final_score": final_score,
            "timestamp": self._now(),
        })

        # ── STEP 8 — DECISION ENGINE ────────────────────────────────────────
        logger.info("─── STEP 8: Decision Engine ───")
        if final_score <= self.APPROVE_THRESHOLD:
            decision = "APPROVE"
            risk_level = "LOW RISK"
        elif final_score <= self.REVIEW_THRESHOLD:
            decision = "REVIEW"
            risk_level = "MEDIUM RISK"
        else:
            decision = "REJECT"
            risk_level = "HIGH RISK"

        logger.info("Decision: %s (%s) — final score: %.2f", decision, risk_level, final_score)

        audit_log.append({
            "step": 8,
            "action": "DECISION",
            "decision": decision,
            "risk_level": risk_level,
            "thresholds": {
                "approve": f"0 – {self.APPROVE_THRESHOLD}",
                "review": f"{self.APPROVE_THRESHOLD} – {self.REVIEW_THRESHOLD}",
                "reject": f"{self.REVIEW_THRESHOLD} – 100",
            },
            "timestamp": self._now(),
        })

        # ── STEP 9 — FINAL OUTPUT ───────────────────────────────────────────
        message = self._build_decision_message(
            decision, risk_level, identity_score, fraud_score,
            final_score, fraud_signals, resolution_rule,
        )

        result = {
            "status": decision,
            "identityScore": identity_score,
            "fraudScore": fraud_score,
            "finalScore": final_score,
            "finalName": final_name,
            "message": message,
            "matchScores": match_scores,
            "nameResolutionRule": resolution_rule,
            "fraudSignals": fraud_signals,
            "audit_log": audit_log,
        }

        # ── STEP 10 — LOGGING (MANDATORY) ───────────────────────────────────
        logger.info("═" * 72)
        logger.info("PIPELINE COMPLETE")
        logger.info("  OCR received         : ✓")
        logger.info("  Field validation     : PASSED")
        logger.info("  Match scores         : %s", match_scores)
        logger.info("  Name resolution      : '%s' via %s", final_name, resolution_rule)
        logger.info("  Identity risk score  : %d (%s)", identity_score, identity_reason)
        logger.info("  Fraud score          : %d (%s)", fraud_score, fraud_category)
        logger.info("  Final risk score     : %.2f", final_score)
        logger.info("  Decision             : %s (%s)", decision, risk_level)
        logger.info("  Message              : %s", message)
        logger.info("═" * 72)

        return result

    # =====================================================================
    #  PER-DOCUMENT VALIDATORS (used by individual OCR routes)
    # =====================================================================

    def _validate_pan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        score = 100

        name = data.get("name")
        if not name or len(str(name).strip()) <= 2:
            issues.append("Name is missing or too short (must be >2 characters)")
            score -= 30

        pan_no = str(data.get("pan") or "").upper()
        if not pan_no or not self.PAN_RE.match(pan_no):
            issues.append(f"PAN number invalid or missing (got: '{pan_no}')")
            score -= 40

        dob = str(data.get("dob") or "")
        if not dob or not self.DOB_RE.match(dob):
            issues.append(f"DOB invalid or missing (got: '{dob}')")
            score -= 20

        score = max(score, 0)
        return {
            "document_type": "pan",
            "is_valid": len(issues) == 0,
            "score": score,
            "normalized_fields": {
                "name": data.get("name"),
                "dob": data.get("dob"),
                "pan": data.get("pan"),
            },
            "issues": issues,
            "summary": self._score_summary(score),
        }

    def _validate_aadhaar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        score = 100

        name = data.get("name")
        if not name or len(str(name).strip()) <= 2:
            issues.append("Name is missing or too short (must be >2 characters)")
            score -= 30

        aadhaar_no = str(data.get("aadhaar") or "").replace(" ", "")
        is_masked = data.get("is_masked") or ("X" in aadhaar_no.upper())

        if is_masked:
            if not re.match(r"^[X0-9]{12}$", aadhaar_no.upper()):
                issues.append("Aadhaar number format invalid even for masked")
                score -= 20
        else:
            digits = re.sub(r"[^0-9]", "", aadhaar_no)
            if len(digits) != 12:
                issues.append(f"Aadhaar must be 12 digits (got {len(digits)})")
                score -= 40
            elif not self._verhoeff_check(digits):
                issues.append("Aadhaar checksum (Verhoeff) failed")
                score -= 30

        dob = str(data.get("dob") or "")
        if not dob or not self.DOB_RE.match(dob):
            issues.append(f"DOB invalid or missing (got: '{dob}')")
            score -= 15

        # Address validation temporarily disabled by request
        # if not data.get("address"):
        #     issues.append("Address is missing")
        #     score -= 10

        score = max(score, 0)
        return {
            "document_type": "aadhaar",
            "is_valid": len(issues) == 0,
            "score": score,
            "is_masked": is_masked,
            "normalized_fields": {
                "name": data.get("name"),
                "dob": data.get("dob"),
                "aadhaar": data.get("aadhaar"),
            },
            "issues": issues,
            "summary": self._score_summary(score),
        }

    def _validate_bank(self, data: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        score = 100

        name = data.get("customer_name") or data.get("name")
        if not name:
            issues.append("Customer name is missing")
            score -= 30

        acc = str(data.get("account_number") or "")
        if not acc or len(acc) < 9:
            issues.append(f"Account number invalid or missing (got: '{acc}')")
            score -= 30

        ifsc = str(data.get("ifsc") or "").upper()
        if not ifsc or not self.IFSC_RE.match(ifsc):
            issues.append(f"IFSC code invalid or missing (got: '{ifsc}')")
            score -= 30

        score = max(score, 0)
        return {
            "document_type": "bank",
            "is_valid": len(issues) == 0,
            "score": score,
            "normalized_fields": {
                "customer_name": name,
                "account_number": data.get("account_number"),
                "ifsc": data.get("ifsc"),
            },
            "issues": issues,
            "summary": self._score_summary(score),
        }

    # =====================================================================
    #  PIPELINE INTERNALS
    # =====================================================================

    def _validate_fields_strict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """STEP 2 — Strict field presence & format validation.

        Returns empty dict if all fields pass, otherwise field→reason map.
        """
        errors: Dict[str, str] = {}

        # ── PAN validation ───────────────────────────────────────────────
        pan = data.get("pan") or {}

        pan_name = pan.get("name")
        if not pan_name or len(str(pan_name).strip()) <= 2:
            errors["pan_name"] = "Missing or too short (must be >2 characters)"

        pan_no = str(pan.get("pan") or pan.get("pan_no") or "").upper()
        if not pan_no or not self.PAN_RE.match(pan_no):
            errors["pan_no"] = f"Must match [A-Z]{{5}}[0-9]{{4}}[A-Z] (got: '{pan_no}')"

        pan_dob = str(pan.get("dob") or "")
        if not pan_dob:
            errors["pan_dob"] = "DOB is missing"
        else:
            # Accept both DD/MM/YYYY (raw) and YYYY-MM-DD (normalized)
            if not self.DOB_INPUT_RE.match(pan_dob) and not self.DOB_RE.match(pan_dob):
                errors["pan_dob"] = f"Invalid date format (got: '{pan_dob}')"

        # ── Aadhaar validation ───────────────────────────────────────────
        aadhaar = data.get("aadhaar") or {}

        aadhaar_name = aadhaar.get("name")
        if not aadhaar_name or len(str(aadhaar_name).strip()) <= 2:
            errors["aadhaar_name"] = "Missing or too short (must be >2 characters)"

        aadhaar_no = str(aadhaar.get("aadhaar") or aadhaar.get("aadhaar_no") or "").replace(" ", "")
        # Remove spaces, must be exactly 12 digits (or masked with X)
        if not re.match(r"^[X0-9]{12}$", aadhaar_no.upper()):
            errors["aadhaar_no"] = f"Must be exactly 12 digits after removing spaces (got: '{aadhaar_no}')"

        # ── Bank Statement validation ────────────────────────────────────
        stmt = data.get("statement") or {}

        stmt_name = stmt.get("customer_name") or stmt.get("name")
        if not stmt_name:
            errors["statement_name"] = "Customer/account holder name is missing"

        stmt_acc = stmt.get("account_number") or stmt.get("acc_no")
        if not stmt_acc:
            errors["statement_acc_no"] = "Account number is missing"

        stmt_ifsc = str(stmt.get("ifsc") or stmt.get("ifsc_no") or "").upper()
        if not stmt_ifsc or not self.IFSC_RE.match(stmt_ifsc):
            errors["statement_ifsc"] = f"Must match [A-Z]{{4}}0[A-Z0-9]{{6}} (got: '{stmt_ifsc}')"

        return errors

    @staticmethod
    def _group_field_errors(errors: Dict[str, str]) -> Dict[str, Any]:
        """Group flat field errors into per-document structure for the error response."""
        grouped: Dict[str, Dict[str, str]] = {"pan": {}, "aadhaar": {}, "statement": {}}
        for key, msg in errors.items():
            if key.startswith("pan_"):
                grouped["pan"][key] = msg
            elif key.startswith("aadhaar_"):
                grouped["aadhaar"][key] = msg
            elif key.startswith("statement_"):
                grouped["statement"][key] = msg
            else:
                grouped.setdefault("other", {})[key] = msg

        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}

    @staticmethod
    def _normalize_name(name: Optional[str]) -> str:
        """Normalize a name: uppercase, trim, collapse whitespace."""
        if not name:
            return ""
        name = str(name).strip().upper()
        name = re.sub(r"\s+", " ", name)
        return name

    def _resolve_name_strict(
        self,
        pan_name: str,
        aadhaar_name: str,
        statement_name: str,
        scores: Dict[str, float],
    ) -> Tuple[str, str]:
        """STEP 4 — Strict ordered name resolution rules.

        Rules applied IN ORDER:
          1. PAN == Statement (score ≥ 95) → PAN name
          2. PAN partial match with any (score ≥ 80) → PAN name
          3. PAN & Aadhaar match (score ≥ 90) → Aadhaar name
          4. Fallback → Aadhaar name
        """
        pan_stmt = scores.get("pan_statement_score", 0)
        pan_aadh = scores.get("pan_aadhaar_score", 0)
        aadh_stmt = scores.get("aadhaar_statement_score", 0)

        # Rule 1: PAN == Statement (≥ 95)
        if pan_stmt >= 95:
            return pan_name, f"Rule 1: PAN matches Statement (score={pan_stmt} ≥ 95)"

        # Rule 2: PAN partial match with ANY (≥ 80)
        if pan_stmt >= 80 or pan_aadh >= 80:
            best_match = "Statement" if pan_stmt >= pan_aadh else "Aadhaar"
            best_score = max(pan_stmt, pan_aadh)
            return pan_name, f"Rule 2: PAN partial match with {best_match} (score={best_score} ≥ 80)"

        # Rule 3: PAN & Aadhaar match (≥ 90)
        if pan_aadh >= 90:
            return aadhaar_name, f"Rule 3: PAN & Aadhaar match (score={pan_aadh} ≥ 90)"

        # Rule 4: Fallback to Aadhaar
        return aadhaar_name, "Rule 4: Default fallback to Aadhaar name (no strong match found)"

    def _compute_identity_risk(self, scores: Dict[str, float]) -> Tuple[int, str]:
        """STEP 5 — Identity risk score using the MINIMUM of all pairwise scores.

        HIGH score = HIGH risk.

        Buckets:
          All ≥ 95 →  0  (LOW RISK)
          All ≥ 85 → 20
          All ≥ 70 → 40
          All ≥ 50 → 70
          Any < 50 → 90  (HIGH RISK)
        """
        all_scores = [
            scores.get("pan_aadhaar_score", 0),
            scores.get("pan_statement_score", 0),
            scores.get("aadhaar_statement_score", 0),
        ]
        min_score = min(all_scores)

        for threshold, risk, reason in self.IDENTITY_BUCKETS:
            if min_score >= threshold:
                detail = (
                    f"{reason} — min pairwise score={min_score}, "
                    f"all scores={all_scores}"
                )
                return risk, detail

        # Should never reach here, but defensive
        return 90, f"No bucket matched (min={min_score})"

    def _detect_fraud_signals(
        self,
        data: Dict[str, Any],
        scores: Dict[str, float],
    ) -> Tuple[int, List[str], str]:
        """STEP 6 — Fraud detection with additive signal model.

        Returns: (fraud_score, signal_list, category)

        Categories:
          Clean (≤10)              → 10
          Minor inconsistency      → 40
          Suspicious mismatch      → 70
          Strong fraud signal      → 90
        """
        signals: List[str] = []
        raw_score = 0

        pan_data = data.get("pan") or {}
        aadhaar_data = data.get("aadhaar") or {}
        stmt_data = data.get("statement") or {}

        pan_aadh = scores.get("pan_aadhaar_score", 0)
        pan_stmt = scores.get("pan_statement_score", 0)
        aadh_stmt = scores.get("aadhaar_statement_score", 0)

        # ── Format anomaly checks ───────────────────────────────────────
        pan_no = str(pan_data.get("pan") or pan_data.get("pan_no") or "").upper()
        if pan_no and not self.PAN_RE.match(pan_no):
            signals.append(f"PAN format anomaly: '{pan_no}' doesn't match expected pattern")
            raw_score += 15

        ifsc = str(stmt_data.get("ifsc") or stmt_data.get("ifsc_no") or "").upper()
        if ifsc and not self.IFSC_RE.match(ifsc):
            signals.append(f"IFSC format anomaly: '{ifsc}' doesn't match expected pattern")
            raw_score += 10

        # ── Name inconsistency checks ───────────────────────────────────
        if pan_aadh < 70:
            signals.append(
                f"Identity document name mismatch: PAN vs Aadhaar (score={pan_aadh})"
            )
            raw_score += 30

        if pan_stmt < 50:
            signals.append(
                f"Bank account name does not match PAN (score={pan_stmt})"
            )
            raw_score += 25

        if aadh_stmt < 50:
            signals.append(
                f"Bank account name does not match Aadhaar (score={aadh_stmt})"
            )
            raw_score += 20

        # ── Cross-document mismatch: DOB ─────────────────────────────────
        pan_dob = str(pan_data.get("dob") or "").strip()
        aadhaar_dob = str(aadhaar_data.get("dob") or "").strip()
        if pan_dob and aadhaar_dob and pan_dob != aadhaar_dob:
            signals.append(
                f"DOB mismatch: PAN='{pan_dob}' vs Aadhaar='{aadhaar_dob}'"
            )
            raw_score += 25

        # ── Edge case: multiple weak matches ──────────────────────────────
        weak_count = sum(1 for s in [pan_aadh, pan_stmt, aadh_stmt] if 50 <= s < 70)
        if weak_count >= 2:
            signals.append(
                f"Multiple weak matches detected ({weak_count}/3 pairs in 50-70 range)"
            )
            raw_score += 15

        # ── Edge case: all three names very different ─────────────────────
        if pan_aadh < 40 and pan_stmt < 40 and aadh_stmt < 40:
            signals.append("All three document names are significantly different")
            raw_score += 20

        # Cap at 100
        fraud_score = min(raw_score, 100)

        # Categorize
        if fraud_score <= 10:
            category = "Clean"
            fraud_score = max(fraud_score, 10)  # Floor at 10 as per spec
        elif fraud_score <= 40:
            category = "Minor inconsistency"
        elif fraud_score <= 70:
            category = "Suspicious mismatch"
        else:
            category = "Strong fraud signal"

        return fraud_score, signals, category

    # =====================================================================
    #  UTILITY METHODS
    # =====================================================================

    @staticmethod
    def _get_name(doc: Optional[Dict], doc_type: str) -> Optional[str]:
        if not doc:
            return None
        if doc_type == "bank":
            return (doc.get("customer_name") or doc.get("name") or "").strip().upper() or None
        return (doc.get("name") or "").strip().upper() or None

    @staticmethod
    def _score_summary(score: int) -> str:
        if score >= 90:
            return "strong"
        if score >= 70:
            return "acceptable"
        if score >= 40:
            return "weak"
        return "invalid"

    @staticmethod
    def _verdict_message(verdict: str, signals: List[str]) -> str:
        if verdict == "APPROVE":
            return "All documents validated successfully. Low risk profile."
        if verdict == "REVIEW":
            joined = ", ".join(signals) if signals else "minor inconsistencies detected"
            return f"Manual review required. Signals: {joined}."
        joined = ", ".join(signals) if signals else "high risk indicators"
        return f"KYC rejected. Signals: {joined}."

    @staticmethod
    def _build_decision_message(
        decision: str,
        risk_level: str,
        identity_score: int,
        fraud_score: int,
        final_score: float,
        fraud_signals: List[str],
        resolution_rule: str,
    ) -> str:
        """Generate a clear, human-readable explanation of the decision."""
        parts: List[str] = []

        if decision == "APPROVE":
            parts.append(
                f"KYC APPROVED — {risk_level}. "
                f"All document names match consistently. "
                f"Identity risk: {identity_score}/100, Fraud risk: {fraud_score}/100, "
                f"Final score: {final_score}/100."
            )
        elif decision == "REVIEW":
            parts.append(
                f"KYC requires MANUAL REVIEW — {risk_level}. "
                f"Identity risk: {identity_score}/100, Fraud risk: {fraud_score}/100, "
                f"Final score: {final_score}/100."
            )
            if fraud_signals:
                parts.append(f"Signals: {'; '.join(fraud_signals)}.")
        else:  # REJECT
            parts.append(
                f"KYC REJECTED — {risk_level}. "
                f"Identity risk: {identity_score}/100, Fraud risk: {fraud_score}/100, "
                f"Final score: {final_score}/100."
            )
            if fraud_signals:
                parts.append(f"Reasons: {'; '.join(fraud_signals)}.")

        parts.append(f"Name resolved via: {resolution_rule}.")
        return " ".join(parts)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat() + "Z"

    # =====================================================================
    #  VERHOEFF CHECKSUM (for Aadhaar number validation)
    # =====================================================================

    _VERHOEFF_D = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]
    _VERHOEFF_P = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
    ]
    _VERHOEFF_INV = [0, 4, 3, 2, 1, 9, 8, 7, 6, 5]

    def _verhoeff_check(self, number: str) -> bool:
        """Return True if the number passes the Verhoeff checksum."""
        try:
            c = 0
            for i, ch in enumerate(reversed(number)):
                c = self._VERHOEFF_D[c][self._VERHOEFF_P[i % 8][int(ch)]]
            return c == 0
        except (IndexError, ValueError):
            return False
