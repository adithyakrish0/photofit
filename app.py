"""
Government Photo/Signature Processor
Automatically crops, resizes, and compresses images to meet government requirements.
"""

import io
import base64
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="GovPhoto Processor")

# Serve static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Presets for common government requirements
PRESETS = {
    "passport_india": {
        "name": "Indian Passport Photo",
        "width": 150,
        "height": 200,
        "min_kb": 20,
        "max_kb": 30,
        "mode": "face"
    },
    "signature": {
        "name": "Signature",
        "width": 140,
        "height": 60,
        "min_kb": 10,
        "max_kb": 20,
        "mode": "signature"
    },
    "aadhaar": {
        "name": "Aadhaar Photo",
        "width": 240,
        "height": 320,
        "min_kb": 20,
        "max_kb": 50,
        "mode": "face"
    },
    "ssc_photo": {
        "name": "SSC/Government Exam Photo",
        "width": 200,
        "height": 230,
        "min_kb": 20,
        "max_kb": 50,
        "mode": "face"
    },
    "ssc_signature": {
        "name": "SSC/Government Exam Signature",
        "width": 140,
        "height": 60,
        "min_kb": 10,
        "max_kb": 20,
        "mode": "signature"
    }
}


def detect_signature(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect signature region in an image using contour analysis.
    Returns bounding box (x, y, w, h) or None if not found.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area and aspect ratio
    valid_contours = []
    img_area = image.shape[0] * image.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Signature should be at least 0.5% but not more than 80% of image
        if 0.005 * img_area < area < 0.8 * img_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            # Signatures are typically wider than tall (aspect ratio > 1)
            if 0.5 < aspect_ratio < 10:
                valid_contours.append(cnt)
    
    if not valid_contours:
        # If no valid contours, try to find any dark region
        return find_dark_region(gray)
    
    # Combine all valid contours to get overall bounding box
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add some padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)
    
    return (x, y, w, h)


def find_dark_region(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Fallback: find the darkest region in the image."""
    # Simple threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)
    
    return (x, y, w, h)


def detect_face(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face region for passport photo cropping.
    Returns bounding box (x, y, w, h) or None if not found.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        # Try with different parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )
    
    if len(faces) == 0:
        return None
    
    # Get the largest face (assuming it's the main subject)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Expand region for passport photo
    # Standard passport photo: face is about 50-60% of total height
    # Need space above head (~10%) and below chin for shoulders (~30-40%)
    img_h, img_w = image.shape[:2]
    
    # Calculate passport-style crop
    face_center_x = x + w // 2
    
    # For 3:4 aspect ratio (passport standard)
    # Face should be approximately 50-55% of the total height
    # This leaves room for: ~15% above head, ~30% below chin (shoulders)
    passport_h = int(h / 0.5)  # Face is 50% of photo height
    passport_w = int(passport_h * 0.75)  # 3:4 aspect ratio
    
    # Calculate crop coordinates
    # Position face so there's ~15% space above head
    space_above = int(passport_h * 0.15)
    crop_y = max(0, y - space_above)
    crop_x = max(0, face_center_x - passport_w // 2)
    
    # Ensure we don't exceed image boundaries
    if crop_x + passport_w > img_w:
        crop_x = max(0, img_w - passport_w)
        passport_w = min(passport_w, img_w)
    if crop_y + passport_h > img_h:
        passport_h = img_h - crop_y
    
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    
    return (crop_x, crop_y, min(passport_w, img_w - crop_x), min(passport_h, img_h - crop_y))


def compress_to_size(
    image: Image.Image,
    target_width: int,
    target_height: int,
    min_kb: float,
    max_kb: float,
    output_format: str = 'JPEG'
) -> Tuple[bytes, str]:
    """
    Compress image to fit within the specified KB range.
    Returns tuple of (bytes, mime_type).
    
    Strategy:
    1. Try different quality levels to hit the target range
    2. If file is too small at max quality, add subtle noise to increase size
    3. If file is too large at min quality, use stronger compression
    """
    # Resize image first
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Determine format settings
    format_upper = output_format.upper()
    if format_upper in ('JPG', 'JPEG'):
        format_upper = 'JPEG'
        mime_type = 'image/jpeg'
        extension = 'jpg'
    elif format_upper == 'PNG':
        mime_type = 'image/png'
        extension = 'png'
    elif format_upper == 'WEBP':
        mime_type = 'image/webp'
        extension = 'webp'
    else:
        format_upper = 'JPEG'
        mime_type = 'image/jpeg'
        extension = 'jpg'
    
    # Convert to RGB if necessary (for JPEG)
    if format_upper == 'JPEG':
        if resized.mode in ('RGBA', 'P'):
            background = Image.new('RGB', resized.size, (255, 255, 255))
            if resized.mode == 'RGBA':
                background.paste(resized, mask=resized.split()[3])
            else:
                background.paste(resized)
            resized = background
        elif resized.mode != 'RGB':
            resized = resized.convert('RGB')
    elif format_upper == 'PNG':
        if resized.mode not in ('RGB', 'RGBA', 'P', 'L'):
            resized = resized.convert('RGBA')
    
    min_bytes = min_kb * 1024
    max_bytes = max_kb * 1024
    
    def save_with_quality(img, quality):
        """Save image and return bytes."""
        buffer = io.BytesIO()
        if format_upper == 'JPEG':
            img.save(buffer, format='JPEG', quality=quality, optimize=False)
        elif format_upper == 'PNG':
            # PNG doesn't use quality, use compression level
            img.save(buffer, format='PNG', optimize=True)
        elif format_upper == 'WEBP':
            img.save(buffer, format='WEBP', quality=quality)
        return buffer.getvalue()
    
    def add_subtle_noise(img, intensity=3):
        """Add subtle noise to increase file size without visible degradation."""
        img_array = np.array(img)
        noise = np.random.randint(-intensity, intensity + 1, img_array.shape, dtype=np.int16)
        noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    # For PNG, we can't easily control size, just return it
    if format_upper == 'PNG':
        buffer = io.BytesIO()
        resized.save(buffer, format='PNG', optimize=True)
        result = buffer.getvalue()
        return result, mime_type
    
    # Binary search for optimal quality (JPEG/WEBP)
    best_result = None
    best_size = 0
    
    # First, try quality 100 to see if we can even reach minimum size
    max_quality_result = save_with_quality(resized, 100)
    max_quality_size = len(max_quality_result)
    
    if max_quality_size >= min_bytes and max_quality_size <= max_bytes:
        # Perfect! Max quality is within range
        return max_quality_result, mime_type
    
    if max_quality_size < min_bytes:
        # Even at max quality, file is too small
        # Use multiple methods to increase file size
        
        # Method 1: Try with 4:4:4 subsampling (no chroma subsampling = larger file)
        buffer = io.BytesIO()
        resized.save(buffer, format='JPEG', quality=100, subsampling=0)
        result = buffer.getvalue()
        
        if len(result) >= min_bytes and len(result) <= max_bytes:
            return result, mime_type
        
        # If 4:4:4 is larger, use it as base
        if len(result) > max_quality_size:
            max_quality_result = result
            max_quality_size = len(result)
        
        # Method 2: Pad the file with JPEG APP0 marker comments
        # This is invisible and doesn't affect image quality
        bytes_needed = int(min_bytes - max_quality_size)
        
        if bytes_needed > 0:
            # Account for JPEG comment header overhead (4 bytes per chunk: FF FE + 2 length bytes)
            # Calculate actual padding data needed
            overhead_per_chunk = 4
            
            # JPEG files end with FF D9, insert comment before that
            result = max_quality_result
            if len(result) >= 2 and result[-2:] == b'\xff\xd9':
                # Insert as JPEG comment (FF FE) before end marker
                # Comment format: FF FE [2 bytes length big-endian] [data]
                
                # Calculate exact padding to reach target (accounting for overhead)
                # We need: current_size + overhead + padding_data = target_size
                # So: padding_data = target_size - current_size - overhead
                target_size = int(min_bytes)  # Use min as target for exact
                current_size = len(result)
                
                # For a single chunk, overhead is 4 bytes
                padding_data_needed = max(0, target_size - current_size - overhead_per_chunk)
                
                if padding_data_needed > 0:
                    # Create padding data
                    padding = b'PhotoFit' * (padding_data_needed // 8 + 1)
                    padding = padding[:padding_data_needed]
                    
                    # Create comment chunk
                    length = len(padding) + 2  # +2 for length bytes
                    comment_chunk = b'\xff\xfe' + length.to_bytes(2, 'big') + padding
                    
                    # Insert before final FF D9
                    result = result[:-2] + comment_chunk + b'\xff\xd9'
            else:
                # Fallback: just append padding (no overhead adjustment needed)
                padding = b'PhotoFit' * (bytes_needed // 8 + 1)
                padding = padding[:bytes_needed]
                result = max_quality_result + padding
        else:
            result = max_quality_result
        
        # Final check
        if len(result) >= min_bytes and len(result) <= max_bytes:
            return result, mime_type
        
        # If still not enough or too big, return best effort
        return result, mime_type
    
    # Max quality is too large, binary search for optimal quality
    low, high = 1, 100
    
    while low <= high:
        mid = (low + high) // 2
        result = save_with_quality(resized, mid)
        size = len(result)
        
        if min_bytes <= size <= max_bytes:
            best_result = result
            best_size = size
            # Try to get higher quality within range
            low = mid + 1
        elif size < min_bytes:
            low = mid + 1
        else:
            high = mid - 1
            if best_result is None:
                best_result = result
                best_size = size
    
    if best_result and min_bytes <= best_size <= max_bytes:
        return best_result, mime_type
    
    # Fallback: return whatever we have
    if best_result:
        return best_result, mime_type
    
    return save_with_quality(resized, 85), mime_type


def process_image(
    image_data: bytes,
    mode: str,
    width: int,
    height: int,
    min_kb: float,
    max_kb: float,
    auto_crop: bool = True,
    output_format: str = 'JPEG',
    grayscale: bool = False
) -> Tuple[bytes, str]:
    """
    Main processing function.
    Returns tuple of (image_bytes, mime_type).
    """
    # Load image
    nparr = np.frombuffer(image_data, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if cv_image is None:
        raise ValueError("Could not decode image")
    
    # Auto-crop based on mode
    if auto_crop:
        if mode == "signature":
            bbox = detect_signature(cv_image)
            if bbox:
                x, y, w, h = bbox
                cv_image = cv_image[y:y+h, x:x+w]
        elif mode == "face":
            bbox = detect_face(cv_image)
            if bbox:
                x, y, w, h = bbox
                cv_image = cv_image[y:y+h, x:x+w]
    
    # Apply grayscale filter if enabled
    if grayscale:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    
    # Convert to PIL Image
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)
    
    # Compress to target size
    result, mime_type = compress_to_size(pil_image, width, height, min_kb, max_kb, output_format)
    
    return result, mime_type


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_path = static_path / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Static files not found. Please check installation.</h1>")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "presets": list(PRESETS.keys())}


@app.get("/presets")
async def get_presets():
    """Get available presets."""
    return PRESETS


@app.post("/detect")
async def detect_mode(file: UploadFile = File(...)):
    """
    Detect whether image contains a face or is likely a signature.
    Returns suggested mode: 'face' or 'signature'
    """
    try:
        image_data = await file.read()
        
        # Load image
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            return {"mode": "face", "confidence": "low"}
        
        # Try face detection
        face_result = detect_face(cv_image)
        
        if face_result is not None:
            return {"mode": "face", "confidence": "high", "face_detected": True}
        else:
            # No face detected, likely a signature
            return {"mode": "signature", "confidence": "medium", "face_detected": False}
            
    except Exception as e:
        # Default to face on error
        return {"mode": "face", "confidence": "low", "error": str(e)}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    preset: Optional[str] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    min_kb: Optional[float] = Form(None),
    max_kb: Optional[float] = Form(None),
    mode: Optional[str] = Form("face"),
    auto_crop: bool = Form(True),
    output_format: str = Form("jpeg"),
    grayscale: bool = Form(False)
):
    """
    Process an uploaded image.
    Either provide a preset name OR custom dimensions.
    """
    try:
        # Read image data
        image_data = await file.read()
        
        # Determine parameters
        if preset and preset in PRESETS:
            p = PRESETS[preset]
            width = p["width"]
            height = p["height"]
            min_kb = p["min_kb"]
            max_kb = p["max_kb"]
            mode = p["mode"]
        elif width and height and min_kb is not None and max_kb is not None:
            pass  # Use provided values
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a valid preset or all custom parameters (width, height, min_kb, max_kb)"
            )
        
        # Process image
        result, mime_type = process_image(
            image_data,
            mode=mode,
            width=width,
            height=height,
            min_kb=min_kb,
            max_kb=max_kb,
            auto_crop=auto_crop,
            output_format=output_format,
            grayscale=grayscale
        )
        
        # Return as base64 for preview and actual size info
        b64_result = base64.b64encode(result).decode('utf-8')
        actual_size_kb = len(result) / 1024
        rounded_size_kb = round(actual_size_kb, 2)
        
        return JSONResponse({
            "success": True,
            "image": f"data:{mime_type};base64,{b64_result}",
            "size_kb": rounded_size_kb,
            "width": width,
            "height": height,
            "format": output_format.lower(),
            "in_range": min_kb <= rounded_size_kb <= max_kb
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

