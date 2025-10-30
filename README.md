# 🧪 EHL Film Prediction App

A full-featured **Streamlit web application** for **data preprocessing, statistical analysis, and machine-learning modeling** of **EHL (Elastohydrodynamic Lubrication) film-thickness experiments**.

It provides:
- automated parsing and cleaning of raw TXT data  
- statistical summaries, correlations, and outlier detection  
- interactive ML regression modeling  
- order-conditioned “deep-learning-style” group modeling  
- ready-to-deploy Docker container

## 📁 Project Structure
```
EHL-Film-Prediction/
│
├── app.py                     # Main Streamlit entrypoint (Home / upload)
├── pages/
│   ├── 01_statistics.py       # Statistics dashboard
│   ├── 02_machine_learning.py # ML modeling & evaluation
│   ├── 03_deep_learning.py    # Order-grouped (DL-style) analysis
│
├── data_processor.py          # TXT parsing, merging, and cleanup
├── preprocess.py              # Data normalization & imputation
├── statistics.py              # Shared statistics logic
├── models.py                  # ML model definitions & utilities
├── ui_shared.py               # Navigation bar and UI helpers
│
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container definition
├── .dockerignore              # Ignored files during build
└── README.md
```

## 🚀 Run Locally (without Docker)
```bash
git clone https://gitlab.com/<your-namespace>/<your-project>.git
cd EHL-Film-Prediction
pip install -r requirements.txt
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501).

## 🐳 Run with Docker
### 1️⃣ Build the image
```bash
docker build -t ehl-film-prediction:latest .
```
For GitLab registry:
```bash
docker build -t registry.gitlab.com/<your-namespace>/<your-project>/ehl-film:latest .
```

### 2️⃣ Run the container
```bash
docker run --rm -p 8501:8501 ehl-film-prediction:latest
```
Visit → [http://localhost:8501](http://localhost:8501)

Background mode:
```bash
docker run -d --name ehl-film -p 8501:8501 ehl-film-prediction:latest
```

### 3️⃣ Stop the container
```bash
docker stop ehl-film
docker rm ehl-film
```

### 4️⃣ Optional Environment Variables
| Variable | Default | Description |
|-----------|----------|-------------|
| `STREAMLIT_SERVER_HEADLESS` | `true` | Run Streamlit headless |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disable telemetry |
| `STREAMLIT_SERVER_PORT` | `8501` | Internal port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Bind to all interfaces |

Example:
```bash
docker run -p 8080:8080 -e STREAMLIT_SERVER_PORT=8080 ehl-film-prediction:latest
```

## 🧩 Dockerfile Summary
```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8501)); s.close()"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📦 .dockerignore
```text
__pycache__/
*.pyc
*.pyo
*.pyd
*.swp
*.swo
.env
.venv
venv
.git
.gitlab-ci.yml
.idea
.vscode
.DS_Store
*.xlsx
*.csv
*.zip
```

## 🧠 Key Features
- Automated TXT/ZIP ingestion & merging  
- Robust preprocessing (duplicate merge, imputation, scaling)  
- Statistics & correlation heatmaps  
- ML regression (Linear, Ridge, Lasso, RF, SVR)  
- Order-grouped modeling for deeper insight  
- Interactive histograms, residuals, learning curves  
- Fully containerized for GitLab CI/CD

## 🧰 Tech Stack
**Python 3.11**, Streamlit, scikit-learn, Plotly, pandas, NumPy  
Tested on Linux & Windows containers.

## 👤 Author
**Mehdi Aminazadeh**  
M.Sc. Computer Science – RPTU Kaiserslautern <br />
Email: mehdi-amina@outlook.de

## 🏁 License
MIT License © 2025 Mehdi Aminazadeh
