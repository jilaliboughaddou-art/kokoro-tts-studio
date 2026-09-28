# Kokoro TTS WebGPU (Serverless In-Browser Speech Synthesis)

100% Client-Side Text-to-Speech using **Kokoro-82M** via **WebGPU / WebAssembly** (`kokoro-js`).
Zero backend servers. Runs directly on the visitor's PC GPU or CPU.

---

## How It Works

1. **Static Hosting:** `index.html` is a single static file hosted on GitHub Pages, Vercel, or Cloudflare Pages.
2. **Client Download:** When the visitor clicks "Initialize", the browser downloads the quantized ONNX model weights (~88MB INT8) directly from Hugging Face into their browser's **IndexedDB / Cache API**.
3. **Local GPU/CPU Inference:** The model executes directly in the user's browser using `WebGPU` (hardware-accelerated GPU) or `WebAssembly` (CPU fallback).
4. **Permanent Cache:** Once downloaded, the model stays cached in the browser for future visits.

---

## Deployment Options

### Option 1: Vercel (Fastest - 1 Command)
```bash
# Inside G:\tts\web
npx vercel deploy --prod
```
Or push this folder to GitHub and import the repository on [vercel.com](https://vercel.com).

---

### Option 2: GitHub Pages (Free forever)
1. Push `index.html` to a GitHub repository (e.g. `your-username/kokoro-tts-web`).
2. Go to **Settings** -> **Pages**.
3. Under **Branch**, select `main` and `/ (root)`.
4. Click **Save**. Your site will be live at `https://your-username.github.io/kokoro-tts-web/`.

---

### Option 3: Cloudflare Pages
1. Go to the Cloudflare Dashboard -> **Workers & Pages** -> **Create application** -> **Pages**.
2. Connect your Git repository or directly drag and drop the `web` folder.
3. Build output directory: leave empty or `.`. Deploy.

---

## Local Testing

To test locally in your browser:
```bash
# Python
python -m http.server 8000 --directory web

# Or Node.js
npx serve web
```
Then open [http://localhost:8000](http://localhost:8000) in Chrome, Edge, or any WebGPU-compatible browser.
