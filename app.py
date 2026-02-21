import logging
from contextlib import asynccontextmanager
from pathlib import Path
from enum import StrEnum

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from predict import ISINMA_OPTIONS, MAHALLE_OPTIONS, load_model, predict_price

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Pydantic schema

IsinmaEnum  = StrEnum("IsinmaEnum",  {v: v for v in ISINMA_OPTIONS})
MahalleEnum = StrEnum("MahalleEnum", {v: v for v in MAHALLE_OPTIONS})


class ApartmentInput(BaseModel):
    net_m2:          float = Field(..., gt=0, description="Net square meters")
    oda_sayisi:      float = Field(..., ge=0)
    salon_sayisi:    int   = Field(..., ge=0)
    banyo_sayisi:    float = Field(..., ge=0)
    bina_yasi:       int   = Field(..., ge=0, description="Ordinal building age group (0–6)")
    bulundugu_kat:   int   = Field(..., ge=0, description="Ordinal floor position")
    kat_sayisi:      int   = Field(..., ge=0, description="Ordinal total floor count")
    balkon:          bool
    asansor:         bool
    otopark:         bool
    esyali:          bool
    site_icerisinde: bool
    isinma:          IsinmaEnum
    mahalle:         MahalleEnum

    @model_validator(mode="after")
    def floor_cannot_exceed_total(self):
        if self.bulundugu_kat > self.kat_sayisi:
            raise ValueError(
                f"bulundugu_kat ({self.bulundugu_kat}) cannot exceed "
                f"kat_sayisi ({self.kat_sayisi})"
            )
        return self

    @model_validator(mode="after")
    def total_rooms_must_be_positive(self):
        if self.oda_sayisi + self.salon_sayisi <= 0:
            raise ValueError("oda_sayisi + salon_sayisi must be greater than zero")
        return self


# App lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup, reuse across requests
    app.state.model = load_model()
    yield


app = FastAPI(title="R.E.P.A.S — Ataşehir Fiyat Tahmini", lifespan=lifespan)


# Routes

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/predict")
def predict(input: ApartmentInput):
    logger.info("predict | mahalle=%s isinma=%s net_m2=%s",
                input.mahalle, input.isinma, input.net_m2)
    try:
        price = predict_price(input.model_dump(), app.state.model)
    except Exception as e:
        logger.error("predict | error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"predict | result={price:,.0f} TL")
    return {"predicted_price": round(price)}


@app.get("/options")
def get_options():
    """Return dropdown options for the UI — includes ordinal label→value maps."""
    return {
        "isinma":  ISINMA_OPTIONS,
        "mahalle": MAHALLE_OPTIONS,
        "bina_yasi": [
            {"label": "0–5 yıl",   "value": 0},
            {"label": "6–10 yıl",  "value": 1},
            {"label": "11–15 yıl", "value": 2},
            {"label": "16–20 yıl", "value": 3},
            {"label": "21–25 yıl", "value": 4},
            {"label": "26–30 yıl", "value": 5},
            {"label": "31+ yıl",   "value": 6},
        ],
        "kat_sayisi": [
            {"label": "1–5 kat",   "value": 0},
            {"label": "6–10 kat",  "value": 1},
            {"label": "11–15 kat", "value": 2},
            {"label": "16–20 kat", "value": 3},
            {"label": "21–25 kat", "value": 4},
            {"label": "26–29 kat", "value": 5},
            {"label": "30+ kat",   "value": 6},
        ],
        "bulundugu_kat": [
            {"label": "Bodrum Kat", "value": 0},
            {"label": "Zemin Kat",  "value": 1},
            {"label": "1–5. kat",   "value": 2},
            {"label": "6–10. kat",  "value": 3},
            {"label": "11–15. kat", "value": 4},
            {"label": "16–20. kat", "value": 5},
            {"label": "21–25. kat", "value": 6},
            {"label": "26–29. kat", "value": 7},
            {"label": "30+. kat",   "value": 8},
            {"label": "Çatı Katı",  "value": 9},
        ],
    }
