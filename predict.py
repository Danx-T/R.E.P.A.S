import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor


MODEL_PATH = Path(__file__).parent / "models" / "catboost_production.cbm"

ISINMA_OPTIONS = [
    "Doğalgaz", "Elektrik / Klima", "Merkezi Sistem",
    "Soba", "Yerden Isıtma", "Yok",
]

MAHALLE_OPTIONS = [
    "Atatürk Mh.", "Aşıkveysel Mh.", "Barbaros", "Esatpaşa Mh.",
    "Ferhatpaşa Mh.", "Fetih", "Kayışdağı Mh.", "Küçükbakkalköy Mh.",
    "Mevlana", "Mimar Sinan", "Mustafa Kemal", "Yeni Sahra",
    "Yeni Çamlıca Mh.", "Yenişehir Mh.", "Örnek Mh.", "İnönü Mh.", "İçerenköy Mh.",
]


def load_model(model_path=MODEL_PATH) -> CatBoostRegressor:
    """Load the production CatBoost model."""
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def build_row(data: dict, feature_columns: list) -> dict:
    """
    Derive all engineered features from raw inputs and expand
    isinma/mahalle strings to one-hot columns.
    Returns a dict aligned to model feature order.
    """
    f = dict(data)

    # toplam_oda = oda + salon
    f["toplam_oda"] = f["oda_sayisi"] + f["salon_sayisi"]
    toplam = f["toplam_oda"]
    m2     = f["net_m2"]

    # M2-based features
    f["m2_per_room"]   = m2 / toplam              if toplam > 0 else 0
    f["room_density"]  = toplam / m2              if m2 > 0    else 0
    f["bath_per_room"] = f["banyo_sayisi"] / toplam if toplam > 0 else 0

    # Floor features — NaN when kat_sayisi == 0 (matches training)
    bk = f["bulundugu_kat"]
    ks = f["kat_sayisi"]
    f["floor_ratio"]    = bk / ks if ks > 0 else np.nan
    f["is_top_floor"]   = int(bk == ks)
    f["is_ground_floor"] = int(bk == 0)

    # Building age — ordinal: 0 = 0-5 yrs, >= 4 = 21+ yrs
    f["is_new"] = int(f["bina_yasi"] == 0)
    f["is_old"] = int(f["bina_yasi"] >= 4)

    # One-hot expand isinma and mahalle
    for opt in ISINMA_OPTIONS:
        f[f"isinma_{opt}"] = int(f["isinma"] == opt)
    for opt in MAHALLE_OPTIONS:
        f[f"mahalle_{opt}"] = int(f["mahalle"] == opt)

    return {col: f.get(col, 0) for col in feature_columns}


def predict_price(data: dict, model: CatBoostRegressor) -> float:
    """Build feature row and return predicted price in TL."""
    feature_columns = model.feature_names_
    row = build_row(data, feature_columns)
    X   = pd.DataFrame([row])
    return float(np.expm1(model.predict(X)[0]))
