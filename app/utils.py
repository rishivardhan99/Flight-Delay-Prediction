# app/utils.py
import os
import pandas as pd
from datetime import datetime

DATA_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input")
os.makedirs(DATA_INPUT_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """
    Save a Streamlit uploaded file to data/input/input.csv (or .json)
    Returns the path and loaded DataFrame.
    """
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"input_{timestamp}.{ext}"
    save_path = os.path.join(DATA_INPUT_DIR, save_name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # load into DataFrame
    if ext in ["csv"]:
        df = pd.read_csv(save_path)
    elif ext in ["json"]:
        df = pd.read_json(save_path, lines=False)
    else:
        raise ValueError("Unsupported file type. Upload CSV or JSON.")
    # Also save a canonical CSV as data/input/input.csv (overwrite)
    canonical_path = os.path.join(DATA_INPUT_DIR, "input.csv")
    df.to_csv(canonical_path, index=False)
    return canonical_path, df
