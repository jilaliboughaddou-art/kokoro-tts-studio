# Kokoro TTS Studio & WebGPU

High-performance Text-to-Speech (TTS) studio powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), supporting 54 voices across multiple languages.

This repository contains two implementations:
1. **Desktop App (Rust + Axum + Tauri / Browser)**: Local native ONNX Runtime inference with DirectML GPU and multi-core CPU acceleration.
2. **Serverless Web App (`/web`)**: 100% in-browser speech synthesis using **WebGPU / WebAssembly** (`kokoro-js`). Zero server backend required; model runs on the visitor's device.

---

## Features

- **54 High-Quality Voices**: American English, British English, Japanese, Mandarin Chinese, Spanish, French, Italian, Hindi, and Brazilian Portuguese.
- **Precision Controls**: Speed adjustments (0.5x to 2.0x), pitch shifting, and instant WAV download.
- **Multiple Quantization Levels**:
  - INT8 Quantized (~88 MB, lowest RAM & fastest inference)
  - FP32 Full Precision (~311 MB)
  - FP16 Half Precision (~156 MB)
  - 4-bit Quantized (~147 MB)
- **WebGPU In-Browser Engine**: Static web client in `web/` that runs directly in any modern browser without servers.

---

## Running Locally

### 1. Web App (WebGPU)
```bash
cd web
python -m http.server 8000
```
Open `http://localhost:8000` in Chrome, Edge, or any WebGPU-capable browser.

### 2. Desktop Server (Rust)
```bash
cargo run --release
```
The server binds to `http://localhost:7860` (or scans up to 7880 if busy) and automatically launches your browser.

---

## Deployment (Web App)

The `web/` directory contains a standalone static web application:
- **GitHub Pages**: Set source to `/web` or push the contents of `web/` to a `gh-pages` branch.
- **Vercel**: Deploy with `npx vercel` inside `web/`.
- **Cloudflare Pages**: Drag and drop the `web/` folder.
