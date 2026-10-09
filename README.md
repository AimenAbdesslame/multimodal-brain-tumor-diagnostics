# Multimodal Brain Tumor Diagnostics (BraTS20)

Multi-task learning framework evaluating simultaneous tumor segmentation and classification on BraTS20 MRI scans, comparing standard multi-tasking (Kendall uncertainty weighting), gradient surgery (PCGrad), and dedicated transfer learning baselines.

---

## Project Structure

```
multimodal-brain-tumor-diagnostics/
├── api/                      # FastAPI backend service
│   ├── main.py               # REST API endpoints & model inference
│   └── requirements.txt      # API dependencies
├── data/                     # Raw / persistent dataset directory (.npz slices)
├── docs/                     # Research logs & documentation
├── outputs/                  # Model checkpoints (.pth / .zip)
├── report/                   # Technical report & assets
│   ├── figures/              # Generated publication figures (.png)
│   ├── tables/               # Experiment summary metrics (.csv, .tex)
│   ├── REPORT.md             # Markdown comprehensive report
│   └── project_report.tex    # XeLaTeX report (compiles to PDF)
├── src/                      # Core machine learning source code
│   ├── config.py             # Hyperparameters & path configurations
│   ├── data/                 # Dataset loader (tumor_dataset.py) & preprocessing
│   ├── losses/               # MultiTaskLoss & SegmentationLoss (Dice + Focal)
│   ├── models/               # UNet, Classification Head, & MTL architectures
│   └── train/                # Training pipelines (segmentation, cls, MTL, PCGrad)
├── web/                      # React + Vite web dashboard
│   ├── src/                  # Components, pages, and API hooks
│   └── package.json          # Node.js frontend dependencies
├── requirements.txt          # Python dependencies
└── run.txt                   # Execution commands
```

---

## Prerequisites

- **Python:** 3.10+
- **Node.js:** 18+ and npm
- **CUDA (Optional):** NVIDIA GPU recommended for training and fast inference

---

## Installation

### 1. Python Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Frontend Dependencies
```bash
cd web
npm install
cd ..
```

---

## Running the Web Interface

### Terminal 1 — FastAPI Backend:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 — Vite Frontend:
```bash
cd web
npm run dev
# Or direct command from run.txt:
# node node_modules/vite/bin/vite.js --port 5173
```

Open browser at **`http://localhost:5173`**.


## Live Demo

see : 
[https://multimodal-brain-tumor-diagnostics.vercel.app](https://multimodal-brain-tumor-diagnostics.vercel.app)