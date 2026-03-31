import os
import io
import logging
from typing import Tuple, Dict, Any, List
from PIL import Image, ImageChops
import numpy as np
import cv2

logger = logging.getLogger("kyc-image-forensics")

FORENSICS_ELA_MAX_DIFF = float(os.getenv("FORENSICS_ELA_MAX_DIFF", "60.0"))
FORENSICS_ELA_VARIANCE = float(os.getenv("FORENSICS_ELA_VARIANCE", "100.0"))
FORENSICS_MOIRE_RATIO = float(os.getenv("FORENSICS_MOIRE_RATIO", "75.0"))

def analyze_exif(image: Image.Image) -> Tuple[bool, str]:
    """
    Level 1 Forensics: Reads EXIF data to detect known photo editing software.
    Returns (is_tampered, reason)
    """
    try:
        exif = image.getexif()
        if not exif:
            return False, ""

        software = str(exif.get(305, "")).lower()
        if not software:
            return False, ""

        blacklisted_software = [
            "photoshop", "gimp", "lightroom", "pixelmator", "canva", "snapseed", "affinity"
        ]
        
        for bad in blacklisted_software:
            if bad in software:
                logger.warning(f"Forensics Alert: Image tampered with {software}")
                return True, f"EXIF metadata indicates editing software: '{software}'"
                
        return False, ""
    except Exception as e:
        logger.error(f"EXIF analysis failed: {e}")
        return False, ""


def analyze_ela(image: Image.Image) -> Tuple[bool, str, float]:
    """
    Level 2 Forensics: Error Level Analysis (ELA).
    Detects copy-pasted fields or recent edits by finding areas of compression variance.
    Returns (is_tampered, reason, max_diff)
    """
    try:
        base_img = image.convert("RGB")
        temp_io = io.BytesIO()
        base_img.save(temp_io, "JPEG", quality=90)
        temp_io.seek(0)
        
        resaved = Image.open(temp_io).convert("RGB")
        ela_image = ImageChops.difference(base_img, resaved)
        
        ela_array = np.array(ela_image)
        max_diff = float(np.max(ela_array))
        variance = float(np.var(ela_array))
        
        if max_diff > FORENSICS_ELA_MAX_DIFF and variance > FORENSICS_ELA_VARIANCE:
            logger.warning(f"Forensics Alert: High ELA variance (Var: {variance:.2f}, MaxDiff: {max_diff})")
            return True, f"Error Level Analysis indicates copy-paste or text tampering (Max Diff: {max_diff})", max_diff
            
        return False, "", max_diff
    except Exception as e:
        logger.error(f"ELA analysis failed: {e}")
        return False, "", 0.0


def analyze_moire(image: Image.Image) -> Tuple[bool, str, float]:
    """
    Level 3 Forensics: Screen / Moiré pattern detection via FFT.
    Detects if the document is a photo taken of a digital screen.
    Returns (is_tampered, reason, high_freq_ratio)
    """
    try:
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        
        f = np.fft.fft2(cv_img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        
        r = min(h, w) // 4
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = x*x + y*y <= r*r
        
        # Calculate energy using pure amplitude, NOT the log-compressed magnitude
        # which flattens the energy curve and causes false positives on normal photos.
        f_mag = np.abs(fshift)
        total_energy = float(np.sum(f_mag))
        low_energy = float(np.sum(f_mag[mask]))
        high_energy = total_energy - low_energy
        
        hf_ratio = (high_energy / total_energy) * 100 if total_energy > 0 else 0
        
        # Threshold: Bank statements are pure text, containing extreme high-frequency edges (ratios around 50-65%). 
        # Screen moiré grids are overwhelmingly high-frequency and push ratios to 75%+.
        # We must keep the threshold high enough to allow sharp text layers to pass.
        if hf_ratio > FORENSICS_MOIRE_RATIO:
            logger.warning(f"Forensics Alert: Screen photo detected (High Frequency Ratio: {hf_ratio:.1f}%)")
            return True, f"FFT analysis indicates a photo of a screen (Moiré pattern detected)", hf_ratio
            
        return False, "", hf_ratio
    except Exception as e:
        logger.error(f"Moiré analysis failed: {e}")
        return False, "", 0.0


def run_forensics(images: List[Image.Image]) -> Dict[str, Any]:
    """
    Master function to run all forensic checks on a list of images (e.g. all pages in a PDF).
    Returns immediately if tampering is found on any page. If clean, returns the metrics
    for the first page.
    """
    if not images:
        return {"is_tampered": False, "reason": "No valid image provided"}
        
    best_clean = None
    
    for idx, image in enumerate(images):
        exif_tampered, exif_reason = analyze_exif(image)
        ela_tampered, ela_reason, ela_max_diff = analyze_ela(image)
        moire_tampered, moire_reason, hf_ratio = analyze_moire(image)
        
        reasons = []
        if exif_tampered: reasons.append(exif_reason)
        if ela_tampered: reasons.append(ela_reason)
        if moire_tampered: reasons.append(moire_reason)
            
        res = {
            "is_tampered": exif_tampered or ela_tampered or moire_tampered,
            "reason": " | ".join(reasons) if reasons else "Clean",
            "ela_max_diff": ela_max_diff,
            "fft_high_freq_ratio": round(hf_ratio, 2)
        }
        
        if res["is_tampered"]:
            if len(images) > 1:
                res["reason"] = f"[Page {idx+1}] {res['reason']}"
            return res
            
        if not best_clean:
            best_clean = res
            
    return best_clean or {"is_tampered": False, "reason": "Clean"}
