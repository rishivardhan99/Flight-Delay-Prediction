# Flight Delay Prediction — Project README

## Project Overview
This repository contains an end-to-end Flight Delay Prediction project: data preprocessing, baseline models (Logistic Regression), a tuned non-linear model (Random Forest), threshold tuning, explainability, and a production-ready Streamlit UI. The app accepts CSV/JSON uploads, predicts delays using both models, shows explanations (LR contributions + RF SHAP/importances), and saves uploaded inputs/predictions to disk.

This README covers setup, directory layout, running locally, Docker deployment, and useful troubleshooting tips.

---

## Directory structure (recommended)
```
project-root/
├─ app/
│  ├─ app.py                 # Streamlit UI
│  ├─ inference.py           # Model loading & predict_both()
│  ├─ explain.py             # LR contribution & RF SHAP/importances helpers
│  └─ utils.py               # Upload helpers (save_uploaded_file)
├─ data/
│  ├─ input/                 # Saved uploaded inputs & predictions (persisted)
│  └─ processed/             # (optional) processed datasets
├─ models/
│  ├─ random_forest_final.joblib
│  ├─ random_forest_final_threshold.joblib
│  ├─ random_forest_feature_list.joblib
│  ├─ log_reg_final_class_weighted.joblib
│  ├─ log_reg_final_threshold.joblib
│  ├─ log_reg_feature_list.joblib
│  └─ scaler.joblib          # optional but strongly recommended
├─ Flight_Delay_Prediction_Final_Conclusion.md
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ README_Flight_Delay_Project.md  # this file
```

---

## Quick setup (developer machine)
1. Clone the repo to your machine.
2. (Optional) Create a Python venv and activate:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venv\Scripts\activate     # Windows (PowerShell)
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure `models/` contains final model artifacts (see models/ list above). If you trained models locally, save them using `joblib.dump()` (examples later). Also ensure `models/scaler.joblib` exists if you scaled features during training (strongly recommended).

---

## Files you must have (artifacts)
- `random_forest_final.joblib` — saved RF model (joblib)
- `random_forest_final_threshold.joblib` — threshold (0.30 recommended)
- `random_forest_feature_list.joblib` — list of features in order
- `log_reg_final_class_weighted.joblib` — saved logistic regression
- `log_reg_final_threshold.joblib` — logistic threshold (0.60)
- `log_reg_feature_list.joblib` — LR feature list (optional)
- `scaler.joblib` — StandardScaler fitted on X_train (optional but required for raw input scaling)

If you do not have `scaler.joblib`, save one from your training notebook (fit on `X_train` only):
```python
from sklearn.preprocessing import StandardScaler
import joblib
scaler = StandardScaler()
scaler.fit(X_train)  # X_train must be the training features (DataFrame)
joblib.dump(scaler, "models/scaler.joblib")
```

---

## Running the Streamlit app locally (development)
From project root:
```bash
export STREAMLIT_SERVER_MAXUPLOADSIZE=50  # (optional) max upload size in MB
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## Sample single-row CSVs for quick testing
Save any of these files into your local machine and upload via the UI.

**Minimal single row (file: `sample_input_one_row.csv`)**
```csv
dep_hour,precip_in,avg_wind_speed_kts,DISTANCE,has_turnaround
18,3.0,5.0,500,0
```

**Extended single row (file: `sample_input_extended_one_row.csv`)**
```csv
MONTH,DAY_OF_MONTH,day_of_week,dep_hour,CRS_ELAPSED_TIME,DISTANCE,precip_in,avg_wind_speed_kts,snow_in,max_temp_f,min_temp_f,temp_range,avg_feel,DEP_1hrpre_num,Arr_1hrpre_num,has_turnaround,scheduled_Turnarnd,late_airjet_when_turnaround_within_180,affected_turnaround_lessthan45,affected_turnaround_lessthan60,affected_turnaround_lessthan90,affected_turnaround_lessthan120,is_rush_hour,ORIGIN_freq,DEST_freq,rain_flag,high_wind_flag,snow_flag
1,12,Friday,18,160,500,3.0,5.0,0.0,28.0,18.0,10.0,23.0,4,2,0,0,0,1,1,1,1,1,0.012,0.009,1,0,0
```

The app will align and fill missing features automatically (features not present will be filled with zeros).

---

## Docker (single-image) — build & run
Make sure `.dockerignore` excludes large files and the `data/` folder to keep your image small. Example `.dockerignore`:
```
.git
__pycache__/
*.pyc
.ipynb_checkpoints/
notebooks/
data/
*.csv
*.json
.env
.venv
venv/
.DS_Store
outputs/
```

**Dockerfile (example)**:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build image** (from repo root):
```bash
docker build -t flight-delay-app:latest .
```

**Run container** (recommended with model & data mounts):
```bash
mkdir -p ~/flight_delay_app/models ~/flight_delay_app/data/input
# copy models to ~/flight_delay_app/models
docker run -d   --name flight-delay-app   -p 8501:8501   -v ~/flight_delay_app/models:/app/models:ro   -v ~/flight_delay_app/data:/app/data   -e STREAMLIT_SERVER_MAXUPLOADSIZE=50   --restart unless-stopped   flight-delay-app:latest
```

Open `http://<SERVER_IP>:8501` (or `http://localhost:8501` if local).

---

## Adding domain + HTTPS (nginx + certbot)
If you want to expose the app via `https://yourdomain.com` using a single host:

1. Install nginx and certbot on the host.
2. Create an nginx site config that proxies to `http://127.0.0.1:8501`.
3. Use `certbot --nginx -d yourdomain.com` to obtain and install certificates.

Small nginx snippet:
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Model saving / loading samples (training notebook)
Example: save models and feature lists using `joblib`:
```python
import joblib
joblib.dump(rf_model, "models/random_forest_final.joblib")
joblib.dump(final_rf_threshold, "models/random_forest_final_threshold.joblib")
joblib.dump(list(X_train.columns), "models/random_forest_feature_list.joblib")

joblib.dump(lr_model, "models/log_reg_final_class_weighted.joblib")
joblib.dump(final_lr_threshold, "models/log_reg_final_threshold.joblib")
joblib.dump(list(X_train.columns), "models/log_reg_feature_list.joblib")
```

---

## Production best practices & notes
- **Do not retrain** models in the UI; inference only.
- Always persist `models/` separately and back it up.
- Keep `scaler.joblib` (fitted on training set) to ensure correct scaling at inference.
- Validate and limit uploaded file size (`STREAMLIT_SERVER_MAXUPLOADSIZE` env var).
- For public use, place the app behind nginx and enable HTTPS.
- Consider adding authentication or restricting access if the app is public-facing.
- Maintain a log of uploads and predictions (simple CSV or structured log) for auditing.
- Monitor memory usage — large Random Forests require more RAM (8GB+ recommended depending on model size).

---

## Troubleshooting
- **Build context too large**: ensure `.dockerignore` excludes `data/`, notebooks, and other heavy files.
- **Models not found inside container**: mount the `models/` folder with `-v ~/flight_delay_app/models:/app/models:ro`.
- **SHAP import errors / heavy dependency**: remove `shap` from `requirements.txt` if size is an issue; the app falls back to `feature_importances_` (less granular but still useful).
- **App crashes on startup**: check `docker logs flight-delay-app` or `streamlit run` console for errors.
- **Prediction mismatch vs notebook**: ensure `scaler.joblib` exists and models were saved with same feature ordering.

---

## Example commands summary (copy-paste)
```bash
# Build image
docker build -t flight-delay-app:latest .

# Run container (with mounts)
docker run -d --name flight-delay-app -p 8501:8501 -v ~/flight_delay_app/models:/app/models:ro -v ~/flight_delay_app/data:/app/data -e STREAMLIT_SERVER_MAXUPLOADSIZE=50 --restart unless-stopped flight-delay-app:latest

# View logs
docker logs -f flight-delay-app

# Stop container
docker stop flight-delay-app && docker rm flight-delay-app
```

---

## Optional next steps (ideas for improvement)
- Evaluate gradient boosting models.
- Integrate real-time data feeds.
- Expand explainability tooling.
- Production-grade deployment with monitoring.

---



