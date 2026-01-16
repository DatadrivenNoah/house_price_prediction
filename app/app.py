from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib
import pandas as pd

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="House Price Predictor")

# -----------------------------
# Path resolution
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

# -----------------------------
# Mount static & templates
# -----------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load(ARTIFACTS_DIR / "house_price_model.pkl")
feature_columns = joblib.load(ARTIFACTS_DIR / "feature_columns.pkl")

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
async def predict(data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(df)[0]
    lower = prediction * 0.90
    upper = prediction * 1.10
    return {
        "prediction": round(prediction, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2)
    }

# -----------------------------
# Run server when executing this file
# -----------------------------
if name == "main":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
