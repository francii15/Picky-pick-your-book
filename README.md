# Picky — Pick Your Book

Picky is a computer-vision bookshelf assistant.

Upload a full bookshelf photo, type a title, part of a title, or an author's name,
and Picky highlights the requested book.

## Required files

Your GitHub repository should contain:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `bookshelf_efficientnetb0_v2.keras`
- `class_names.json` (optional)

## Streamlit secret

In Streamlit Community Cloud, add:

```toml
ROBOFLOW_API_KEY = "your_actual_roboflow_api_key"
```

Do not commit the API key to GitHub.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Backend

The detector, OCR, classifier, matching thresholds, and fusion rules remain hidden
from the app interface.
