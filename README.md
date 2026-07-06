# Mini ElectroNa (SaltGuard) — ML Mortality Risk Dashboard

This project provides a small web dashboard backed by a Flask API that predicts **mortality risk** from a few clinical laboratory inputs and stores prediction history in a SQLite database.

## Project Structure

- **`dashboard/`**: Front-end HTML/CSS/JS (login page + dashboard UI).
- **`ML-engine/`**: Back-end ML + Flask API + data/ML artifacts.
  - **`app.py`**: Flask server + API routes.
  - **`requirements.txt`**: Python dependencies for the back-end.
  - **`xgb_mortality_model.pkl`**: Trained XGBoost model.
  - **`xgb_feature_columns.pkl`**: Feature column ordering expected by the model.
  - **`init_db.py`**: Builds SQLite tables (and/or loads CSVs into SQLite).
  - **`final.csv`, `patients.csv`, `labevents.csv`, ...`**: Dataset sources used for building `final_features`.

## What the API Does

- **`POST /predict`**
  - Accepts JSON containing lab inputs such as:
    - `anchor_age`
    - `gender`
    - `Creatinine`
    - `Hemoglobin`
    - `Sodium`
    - `Urea Nitrogen`
    - `WBC`
  - Computes engineered features (ratios/flags), aligns them to the model’s expected feature columns, runs the XGBoost model, and returns:
    - `mortality_probability`
    - `prediction_class`

- **`GET /api/history`**
  - Returns the latest stored predictions from `prediction_history`.

- **`DELETE /api/history`**
  - Clears `prediction_history`.

- **`GET /api/patients`**
  - Returns sample patient rows from the SQLite DB (`final_features` joined with `patients`).

- **`POST /api/ai-insight`**
  - Generates an “AI clinical insight” report from the returned probability and provided lab inputs.

## Setup & Run

### 1) Create and activate a Python virtual environment

From the repository root:

```bat
python -m venv .venv
.
venv\Scripts\activate
```

### 2) Install dependencies

```bat
cd ML-engine
pip install -r requirements.txt
```

### 3) Start the Flask API

```bat
cd ..
python "ML-engine\app.py"
```

The server runs at:

- **`http://127.0.0.1:5000/`**

### 4) Open the Dashboard

Open in your browser:

- **`http://127.0.0.1:5000/`**

## SQLite Database

- The API uses **`saltguard.db`**.
- `prediction_history` is created automatically on server startup if it does not exist.

If you need to (re)build the DB from CSVs, use `ML-engine/init_db.py`.

## Files Produced / Used

- `ML-engine/xgb_mortality_model.pkl`
- `ML-engine/xgb_feature_columns.pkl`
- `saltguard.db` (created/updated at runtime)

## Notes / Troubleshooting

- The model artifacts (the `.pkl` files) must match the engineered features produced by the API.
- The Flask server must load the correct model/feature pickle files from `ML-engine/`.

## License

No license specified. Add one if you plan to publish or share this project.

