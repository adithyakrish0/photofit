/**
 * GovPhoto Processor - Frontend Logic
 */

// State
let currentFile = null;
let currentPreset = 'custom';
let currentFormat = 'jpeg';
let processedImageData = null;
let processedFormat = 'jpeg';
let cameraStream = null;
let useFrontCamera = true;

// Preset configurations (should match backend)
const PRESETS = {
    passport_india: { width: 150, height: 200, min_kb: 20, max_kb: 30, mode: 'face' },
    signature: { width: 140, height: 60, min_kb: 10, max_kb: 20, mode: 'signature' },
    aadhaar: { width: 240, height: 320, min_kb: 20, max_kb: 50, mode: 'face' },
    ssc_photo: { width: 200, height: 230, min_kb: 20, max_kb: 50, mode: 'face' },
    custom: { width: 150, height: 200, min_kb: 20, max_kb: 30, mode: 'face' }
};

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const presetButtons = document.getElementById('presetButtons');
const customSettings = document.getElementById('customSettings');
const processBtn = document.getElementById('processBtn');
const previewSection = document.getElementById('previewSection');
const originalPreview = document.getElementById('originalPreview');
const processedPreview = document.getElementById('processedPreview');
const originalInfo = document.getElementById('originalInfo');
const processedInfo = document.getElementById('processedInfo');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultStatus = document.getElementById('resultStatus');
const downloadBtn = document.getElementById('downloadBtn');

// Camera elements
const cameraBtn = document.getElementById('cameraBtn');
const cameraModal = document.getElementById('cameraModal');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraLoading = document.getElementById('cameraLoading');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const captureBtn = document.getElementById('captureBtn');

// Input elements
const widthInput = document.getElementById('widthInput');
const heightInput = document.getElementById('heightInput');
const minKbInput = document.getElementById('minKbInput');
const maxKbInput = document.getElementById('maxKbInput');
const modeSelect = document.getElementById('modeSelect');
const autoCropCheck = document.getElementById('autoCropCheck');
const formatButtons = document.getElementById('formatButtons');

// Initialize
document.addEventListener('DOMContentLoaded', init);

function init() {
    setupEventListeners();
    updateUIFromPreset(currentPreset);
    // Show custom settings since it's the default
    if (currentPreset === 'custom') {
        customSettings.classList.add('visible');
    }
}

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Preset buttons
    presetButtons.addEventListener('click', handlePresetClick);

    // Format buttons
    formatButtons.addEventListener('click', handleFormatClick);

    // Process button
    processBtn.addEventListener('click', processImage);

    // Download button
    downloadBtn.addEventListener('click', downloadImage);

    // Camera buttons
    cameraBtn.addEventListener('click', openCamera);
    closeCameraBtn.addEventListener('click', closeCamera);
    switchCameraBtn.addEventListener('click', switchCamera);
    captureBtn.addEventListener('click', capturePhoto);

    // Close camera on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && cameraModal.classList.contains('visible')) {
            closeCamera();
        }
    });

    // Custom input changes - auto-switch to custom preset
    [widthInput, heightInput, minKbInput, maxKbInput].forEach(input => {
        input.addEventListener('change', () => {
            if (currentPreset !== 'custom') {
                selectPreset('custom');
            }
        });
    });

    // Unit selector - show/hide DPI field
    const unitSelect = document.getElementById('unitSelect');
    const dpiGroup = document.getElementById('dpiGroup');

    unitSelect.addEventListener('change', () => {
        if (unitSelect.value === 'px') {
            dpiGroup.style.display = 'none';
        } else {
            dpiGroup.style.display = 'flex';
        }
        if (currentPreset !== 'custom') {
            selectPreset('custom');
        }
    });

    // Exact size input - syncs to min and max
    const exactKbInput = document.getElementById('exactKbInput');

    exactKbInput.addEventListener('input', () => {
        const exactValue = exactKbInput.value;
        if (exactValue) {
            minKbInput.value = exactValue;
            maxKbInput.value = exactValue;
        }
        if (currentPreset !== 'custom') {
            selectPreset('custom');
        }
    });

    // Clear exact when min/max is manually changed
    minKbInput.addEventListener('input', () => {
        if (minKbInput.value !== maxKbInput.value) {
            exactKbInput.value = '';
        }
    });
    maxKbInput.addEventListener('input', () => {
        if (minKbInput.value !== maxKbInput.value) {
            exactKbInput.value = '';
        }
    });
}

// Convert dimensions to pixels based on unit
function getDimensionsInPixels() {
    const unit = document.getElementById('unitSelect').value;
    let width = parseFloat(widthInput.value);
    let height = parseFloat(heightInput.value);

    if (unit === 'px') {
        return { width: Math.round(width), height: Math.round(height) };
    }

    const dpi = parseInt(document.getElementById('dpiInput').value) || 300;

    if (unit === 'cm') {
        // cm to pixels: cm * dpi / 2.54
        width = Math.round(width * dpi / 2.54);
        height = Math.round(height * dpi / 2.54);
    } else if (unit === 'inch') {
        // inches to pixels: inches * dpi
        width = Math.round(width * dpi);
        height = Math.round(height * dpi);
    }

    return { width, height };
}

// ==================== Camera Functions ====================

async function openCamera() {
    cameraModal.classList.add('visible');
    cameraLoading.style.display = 'flex';

    try {
        await startCamera();
    } catch (error) {
        console.error('Camera error:', error);
        alert('Could not access camera. Please check permissions or try uploading a file instead.');
        closeCamera();
    }
}

async function startCamera() {
    // Stop any existing stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: {
            facingMode: useFrontCamera ? 'user' : 'environment',
            width: { ideal: 1280 },
            height: { ideal: 960 }
        }
    };

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = cameraStream;

        // Wait for video to be ready
        cameraVideo.onloadedmetadata = () => {
            cameraLoading.style.display = 'none';
        };
    } catch (error) {
        // If front camera fails, try any camera
        if (useFrontCamera) {
            useFrontCamera = false;
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 960 } }
            });
            cameraVideo.srcObject = cameraStream;
            cameraVideo.onloadedmetadata = () => {
                cameraLoading.style.display = 'none';
            };
        } else {
            throw error;
        }
    }
}

function closeCamera() {
    cameraModal.classList.remove('visible');

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    cameraVideo.srcObject = null;
}

async function switchCamera() {
    useFrontCamera = !useFrontCamera;
    cameraLoading.style.display = 'flex';

    try {
        await startCamera();
    } catch (error) {
        console.error('Failed to switch camera:', error);
        // Revert and try again
        useFrontCamera = !useFrontCamera;
        await startCamera();
    }
}

function capturePhoto() {
    if (!cameraVideo.srcObject) return;

    // Set canvas size to video size
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;

    // Draw current frame
    const ctx = cameraCanvas.getContext('2d');

    // Flip horizontally if using front camera (mirror effect)
    if (useFrontCamera) {
        ctx.translate(cameraCanvas.width, 0);
        ctx.scale(-1, 1);
    }

    ctx.drawImage(cameraVideo, 0, 0);

    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Convert to blob
    cameraCanvas.toBlob((blob) => {
        // Create a File object from the blob
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const file = new File([blob], `camera_photo_${timestamp}.jpg`, { type: 'image/jpeg' });

        // Close camera and handle the file
        closeCamera();
        handleFile(file);
    }, 'image/jpeg', 0.92);
}

// ==================== File Handling ====================

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, WEBP)');
        return;
    }

    currentFile = file;

    // Update upload area
    uploadArea.classList.add('has-file');
    uploadArea.querySelector('h2').textContent = file.name;
    uploadArea.querySelector('p').textContent = `${(file.size / 1024).toFixed(1)} KB`;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        originalPreview.src = e.target.result;

        // Get image dimensions
        const img = new Image();
        img.onload = () => {
            originalInfo.textContent = `${img.width}×${img.height}px • ${(file.size / 1024).toFixed(1)} KB`;

            // Auto-detect mode based on image characteristics
            autoDetectMode(img);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Show preview section and enable process button
    previewSection.classList.add('visible');
    processBtn.disabled = false;

    // Reset processed preview
    processedPreview.src = '';
    processedInfo.textContent = '-';
    resultStatus.textContent = '';
    resultStatus.className = 'result-status';
    downloadBtn.disabled = true;
    processedImageData = null;
}

// Auto-detect whether image is a signature or face photo using backend
async function autoDetectMode(img) {
    // Skip detection if no file
    if (!currentFile) return;

    try {
        // Send image to backend for face detection
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            // Update mode based on detection
            modeSelect.value = result.mode;

            // Optional: show detection result
            console.log(`Auto-detected: ${result.mode} (face_detected: ${result.face_detected})`);
        }
    } catch (error) {
        // Silently fail, keep default mode
        console.log('Auto-detection failed, using default mode');
    }
}

// ==================== Preset Handling ====================

function handlePresetClick(e) {
    const btn = e.target.closest('.preset-btn');
    if (!btn) return;

    const preset = btn.dataset.preset;
    selectPreset(preset);
}

function selectPreset(preset) {
    currentPreset = preset;

    // Update active button
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.preset === preset);
    });

    // Show/hide custom settings
    if (preset === 'custom') {
        customSettings.classList.add('visible');
    } else {
        customSettings.classList.remove('visible');
        updateUIFromPreset(preset);
    }
}

function updateUIFromPreset(preset) {
    if (preset === 'custom') return;

    const config = PRESETS[preset];
    if (!config) return;

    widthInput.value = config.width;
    heightInput.value = config.height;
    minKbInput.value = config.min_kb;
    maxKbInput.value = config.max_kb;
    modeSelect.value = config.mode;
}

// ==================== Format Handling ====================

function handleFormatClick(e) {
    const btn = e.target.closest('.format-btn');
    if (!btn) return;

    currentFormat = btn.dataset.format;

    // Update active button
    document.querySelectorAll('.format-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.format === currentFormat);
    });
}

async function processImage() {
    if (!currentFile) return;

    // Show loading
    loadingOverlay.classList.add('visible');
    processBtn.disabled = true;
    resultStatus.textContent = '';
    resultStatus.className = 'result-status';

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);

        if (currentPreset !== 'custom') {
            formData.append('preset', currentPreset);
        } else {
            // Convert to pixels if using cm/inch
            const dims = getDimensionsInPixels();
            formData.append('width', dims.width);
            formData.append('height', dims.height);
            formData.append('min_kb', minKbInput.value);
            formData.append('max_kb', maxKbInput.value);
            formData.append('mode', modeSelect.value);
        }

        formData.append('auto_crop', autoCropCheck.checked);
        formData.append('output_format', currentFormat);
        formData.append('grayscale', document.getElementById('bwCheck').checked);

        // Send request
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        const result = await response.json();

        // Update preview
        processedPreview.src = result.image;
        processedInfo.textContent = `${result.width}×${result.height}px • ${result.size_kb} KB`;

        // Store for download
        processedImageData = result.image;
        processedFormat = result.format || currentFormat;

        // Update status
        if (result.in_range) {
            resultStatus.textContent = `✓ Perfect! Size is within ${minKbInput.value}-${maxKbInput.value} KB range`;
            resultStatus.className = 'result-status success';
        } else {
            resultStatus.textContent = `⚠ Size (${result.size_kb} KB) is outside target range. Try adjusting settings.`;
            resultStatus.className = 'result-status warning';
        }

        // Enable download
        downloadBtn.disabled = false;

    } catch (error) {
        console.error('Processing error:', error);
        resultStatus.textContent = `✗ Error: ${error.message}`;
        resultStatus.className = 'result-status error';
    } finally {
        // Hide loading
        loadingOverlay.classList.remove('visible');
        processBtn.disabled = false;
    }
}

function downloadImage() {
    if (!processedImageData) return;

    // Determine mime type and extension based on format
    let mimeType = 'image/jpeg';
    let extension = 'jpg';

    if (processedFormat === 'png') {
        mimeType = 'image/png';
        extension = 'png';
    } else if (processedFormat === 'webp') {
        mimeType = 'image/webp';
        extension = 'webp';
    }

    // Generate clean filename
    let baseName = 'photo';
    if (currentFile && currentFile.name) {
        baseName = currentFile.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');
    }

    // Get dimensions in pixels
    const dims = getDimensionsInPixels();
    const filename = `${baseName}_${dims.width}x${dims.height}.${extension}`;

    // Extract base64 data and convert to blob
    const base64Data = processedImageData.split(',')[1];
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);

    // Use File instead of Blob for better Chrome compatibility
    const file = new File([byteArray], filename, { type: mimeType });

    // Create download link with explicit filename
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;

    // Append to body, click, and cleanup
    document.body.appendChild(a);

    // Use setTimeout to ensure Chrome processes the download correctly
    setTimeout(() => {
        a.click();
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }, 0);
}
