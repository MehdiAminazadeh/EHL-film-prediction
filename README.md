# EHL Film Prediction

This project is a Streamlit web application for exploring, cleaning, and modeling EHL (elastohydrodynamic lubrication) film-thickness data.

It provides:
- data upload and preprocessing
- statistical analysis (summary, correlations, distributions, outlier detection)
- classical machine learning models for film prediction
- order-based (grouped) modeling
- Docker support for reproducible deployment

## 1. Project Structure

```text
.
├── app.py
├── pages/
│   ├── 01_statistics.py
│   ├── 02_machine_learning.py
│   └── 03_deep_learning.py
├── data_processor.py
├── preprocess.py
├── statistics.py
├── models.py
├── ui_shared.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 2. Prerequisites

- Python 3.11 (recommended) or 3.10+
- Git (if you clone the repository)
- Docker (only if you want to run it in a container)

## 3. Run on Local Machine (with virtual environment)

### 3.1 Get the project

```bash
git clone https://gitlab.com/your-namespace/ehl-film-prediction.git
cd ehl-film-prediction
```

(If you already have the folder, just go into it.)

### 3.2 Create a virtual environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

You should now see `(.venv)` in your terminal prompt.

### 3.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Start the application

```bash
streamlit run app.py
```

Then open: http://localhost:8501

If you want a different port:

```bash
streamlit run app.py --server.port=8502
```

## 4. Run with Docker

This is useful when you want a clean, reproducible run without installing Python and dependencies on the host.

### 4.1 Build the image

Run this inside the project directory (where the Dockerfile is):

```bash
docker build -t ehl-film-prediction:latest .
```

### 4.2 Run the container

```bash
docker run --rm -p 8501:8501 ehl-film-prediction:latest
```

Now visit: http://localhost:8501

To run in background:

```bash
docker run -d --name ehl-film -p 8501:8501 ehl-film-prediction:latest
```

To stop it later:

```bash
docker stop ehl-film
docker rm ehl-film
```

### 4.3 Environment overrides (optional)

The Dockerfile already sets reasonable defaults for Streamlit. If you need to override the port inside the container:

```bash
docker run --rm -p 8502:8502 -e STREAMLIT_SERVER_PORT=8502 ehl-film-prediction:latest
```

## 5. Notes

- The pages under `pages/` depend on a cleaned dataframe stored in `st.session_state["last_clean_df"]`, which is filled on the main page (`app.py`). So the normal workflow is: open the app → upload the data → go to Statistics / ML / DL pages.
- Do not commit virtual environments, `__pycache__`, or large data files.

## 6. Author

- Mehdi Aminazadeh
- M.Sc. Computer Science, RPTU Kaiserslautern
- m.aminazadeh@edu.rptu.de
