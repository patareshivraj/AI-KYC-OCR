import io
import logging
from typing import Tuple, Dict, Any
from PIL import Image, ImageChops
import numpy as np
import cv2

logger = logging.getLogger("kyc-image-forensics")

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
        
        if max_diff > 60 and variance > 100:
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
        
        total_energy = float(np.sum(magnitude_spectrum))
        low_energy = float(np.sum(magnitude_spectrum[mask]))
        high_energy = total_energy - low_energy
        
        hf_ratio = (high_energy / total_energy) * 100 if total_energy > 0 else 0
        
        if hf_ratio > 30.0:
            logger.warning(f"Forensics Alert: Screen photo detected (High Frequency Ratio: {hf_ratio:.1f}%)")
            return True, f"FFT analysis indicates a photo of a screen (Moiré pattern detected)", hf_ratio
            
        return False, "", hf_ratio
    except Exception as e:
        logger.error(f"Moiré analysis failed: {e}")
        return False, "", 0.0


def run_forensics(image: Image.Image) -> Dict[str, Any]:
    """
    Master function to run all forensic checks on an image.
    """
    if not image:
        return {"is_tampered": False, "reason": "No valid image provided"}
        
    exif_tampered, exif_reason = analyze_exif(image)
    ela_tampered, ela_reason, ela_max_diff = analyze_ela(image)
    moire_tampered, moire_reason, hf_ratio = analyze_moire(image)
    
    reasons = []
    if exif_tampered: reasons.append(exif_reason)
    if ela_tampered: reasons.append(ela_reason)
    if moire_tampered: reasons.append(moire_reason)
        
    return {
        "is_tampered": exif_tampered or ela_tampered or moire_tampered,
        "reason": " | ".join(reasons) if reasons else "Clean",
        "ela_max_diff": ela_max_diff,
        "fft_high_freq_ratio": round(hf_ratio, 2)
    }
