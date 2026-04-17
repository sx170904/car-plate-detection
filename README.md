# SSPD Demo — License Plate Detector

UCCD3094 Group 6 — Streamlit demo for the Single-Shot Plate Detector.

## Files
- `app.py`            Streamlit interface
- `model_arch.py`     SSPD model classes
- `sspd_best.pth`     Trained weights (produced by running the main notebook)
- `model_config.json` Model config (grid size, input size) (produced by the main notebook)
- `requirements.txt`  Python dependencies

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed in the terminal (usually http://localhost:8501),
upload a car image, and adjust the confidence threshold in the sidebar.
