---
title: PhotoFit
emoji: 📸
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# 📸 PhotoFit

**Auto-crop and compress photos for exam sites, job portals & ID uploads.**

[![Try it on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Try%20Demo-yellow)](https://huggingface.co/spaces/adithyakrishnan/photofit)

---

## ✨ Features

- 🎯 **Quick Presets** - SSC, IBPS, Passport, Aadhaar, Custom
- 📷 **Camera Capture** - Take photo directly from webcam
- 🔍 **Auto-detect** - Face/signature detection with smart cropping
- 📦 **Smart Compression** - Hit exact KB range with invisible padding
- 🖼️ **Multiple Formats** - JPG, PNG, WEBP output
- ⚫ **B&W Filter** - Black & white option for signatures
- 📱 **Responsive** - Works on PC and mobile

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repo
git clone https://github.com/adithyakrish0/photofit.git
cd photofit

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn app:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

### Run with Docker

```bash
docker build -t photofit .
docker run -p 8000:7860 photofit
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI, OpenCV, Pillow |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker, Hugging Face Spaces |

---

## 📖 How It Works

1. **Upload** or capture a photo
2. **Select** a preset or enter custom dimensions
3. **Process** - auto-crops face/signature, resizes, compresses
4. **Download** the perfectly sized result!

---

## 📝 License

MIT License - feel free to use and modify!

---

Made with ❤️ for hassle-free photo uploads
