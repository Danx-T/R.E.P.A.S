# R.E.P.A.S — Real Estate Price Analysis System

> **Ataşehir Apartment Rental Price Predictor**  
> End-to-end ML project: scraping → cleaning → feature engineering → CatBoost model → FastAPI serving → web UI

---

## Overview

R.E.P.A.S is a machine learning system that predicts monthly rental prices for apartments in **Ataşehir, Istanbul**. It covers the full pipeline from raw data collection to a live prediction API with a browser-based UI.

The dataset was collected from a major Turkish real estate listings platform, targeting all active rental listings in the Ataşehir district. Data was collected on **December 15, 2025**, capturing a point-in-time snapshot of the local rental market.

---

## Dataset

> **Note:** The raw dataset is not included in this repository due to the source platform's terms of service. The column schema and statistics are documented below for full reproducibility context. The pre-trained model is included — see [Getting Started](#getting-started) to run the API without retraining.

| Property | Value |
|---|---|
| **Source** | Major Turkish real estate listings platform |
| **Location** | Ataşehir, Istanbul, Turkey |
| **Listing type** | Residential rentals only (`Kiralık Daire`) |
| **Collection date** | December 15, 2025 |
| **Raw records** | 1,258 listings |

### Column Reference

| Column (Turkish) | Type | Description |
|---|---|---|
| `evin_durumu` | string | Listing type — `Kiralık` (For Rent) |
| `il` | string | Province — always `İstanbul` |
| `ilce` | string | District — always `Ataşehir` |
| `mahalle` | string | Neighbourhood (17 unique values, see below) |
| `baslik` | string | Listing title (free text, in Turkish) |
| `fiyat` | int | Monthly rent in Turkish Lira (₺) — **target variable** |
| `ilan_no` | int | Unique listing ID on sahibinden.com |
| `emlak_tipi` | string | Property type — always `Kiralık Daire` |
| `brut_m2` | float | Gross area in m² (includes common areas) |
| `net_m2` | float | Net usable area in m² — used as model feature |
| `oda_sayisi` | string | Room count in Turkish format (e.g. `2+1` = 2 bedrooms + 1 living room) |
| `bina_yasi` | string | Building age group (e.g. `6-10 arası` = 6–10 years old) |
| `bulundugu_kat` | string | Floor the apartment is on (e.g. `3`, `Zemin Kat`, `Çatı Katı`) |
| `kat_sayisi` | string/int | Total number of floors in the building |
| `banyo_sayisi` | float | Number of bathrooms |
| `balkon` | string | Has balcony — `Evet` (Yes) / `Hayır` (No) |
| `asansor` | string | Has elevator — `Evet` / `Hayır` |
| `otopark` | string | Has parking — `Evet` / `Hayır` |
| `esyali` | string | Furnished — `Evet` / `Hayır` |
| `site_icerisinde` | string | Inside a gated complex — `Evet` / `Hayır` |
| `kullanim_durumu` | string | Occupancy status — `Boş` (Empty) / `Kiracılı` (Tenanted) |
| `tapu_durumu` | string | Title deed type (e.g. `Kat Mülkiyetli`, `Arsa Tapulu`) |
| `isinma` | string | Heating system (see breakdown below) |
| `ilan_sahibi` | string | Listing owner name (agent or private) |
| `telefon_is` | string | Work phone number |
| `telefon_cep` | string | Mobile phone number |
| `url` | string | Full URL of the original listing |
| `tarih` | datetime | Scrape timestamp |

#### `mahalle` — Neighbourhoods (17)
`Atatürk Mh.`, `Aşıkveysel Mh.`, `Barbaros`, `Esatpaşa Mh.`, `Ferhatpaşa Mh.`, `Fetih`, `Kayışdağı Mh.`, `Küçükbakkalköy Mh.`, `Mevlana`, `Mimar Sinan`, `Mustafa Kemal`, `Yeni Sahra`, `Yeni Çamlıca Mh.`, `Yenişehir Mh.`, `Örnek Mh.`, `İnönü Mh.`, `İçerenköy Mh.`

#### `oda_sayisi` — Room Format
Turkish listings use `X+Y` notation where `X` = number of bedrooms and `Y` = number of living rooms.  
Examples: `1+1` (1 bed + 1 living), `3+1`, `Stüdyo (1+0)` (studio).

#### `isinma` — Heating Systems (grouped in pipeline)
| Raw value | Mapped category |
|---|---|
| `Kombi (Doğalgaz)`, `Doğalgaz Sobası` | `Doğalgaz` (Natural gas) |
| `Merkezi`, `Merkezi (Pay Ölçer)`, `Kat Kaloriferi`, `Fancoil Ünitesi` | `Merkezi Sistem` (Central heating) |
| `Kombi (Elektrik)`, `Klima` | `Elektrik / Klima` (Electric/AC) |
| `Yerden Isıtma` | `Yerden Isıtma` (Underfloor heating) |
| `Soba` | `Soba` (Stove) |
| `Yok` | `Yok` (None) |

#### `bina_yasi` — Building Age (ordinal encoded)
| Label | Ordinal value |
|---|---|
| 0–5 years | 0 |
| 6–10 years | 1 |
| 11–15 years | 2 |
| 16–20 years | 3 |
| 21–25 years | 4 |
| 26–30 years | 5 |
| 31+ years | 6 |

#### `bulundugu_kat` — Floor Position (ordinal encoded)
| Label | Ordinal value |
|---|---|
| Basement (`Bodrum Kat`) | 0 |
| Ground floor (`Zemin Kat`, `Bahçe Katı`, etc.) | 1 |
| 1–5th floor | 2 |
| 6–10th floor | 3 |
| 11–15th floor | 4 |
| 16–20th floor | 5 |
| 21–25th floor | 6 |
| 26–29th floor | 7 |
| 30th floor and above | 8 |
| Penthouse (`Çatı Katı`) | 9 |

---

## Project Structure

```
R.E.P.A.S/
├── data/
│   ├── raw/              # Raw scraped CSV
│   ├── processed/        # Cleaned and feature-engineered datasets
│   └── splits/           # Train / test splits
├── models/
│   ├── catboost_production.cbm   # Model served by the API
│   ├── catboost_final.cbm        # Model trained on train set only
│   └── oof_predictions.csv       # Out-of-fold predictions for diagnostics
├── src/                  # Training pipeline (numbered scripts)
│   ├── 01null_value_analysis.py
│   ├── 02data_cleaning.py
│   ├── 03eda.py
│   ├── 04feature_transformation.py
│   ├── 05model_eda.py
│   ├── 06feature_engineering.py
│   ├── 07train_test_split.py
│   ├── 08random_forest_baseline.py
│   ├── 09boosting_baselines.py
│   ├── 10catboost_tuning.py      # Optuna hyperparameter search
│   ├── 11oof_final_model.py      # 5-fold OOF evaluation
│   ├── 12test_evaluation.py      # Holdout test set evaluation
│   └── 13final_refit.py          # Final model trained on all data
├── logs/                 # Per-step log files
├── app.py                # FastAPI application
├── predict.py            # Inference logic (feature builder + model loader)
├── index.html            # Single-file web UI
├── main.py               # Pipeline runner (executes src/ scripts in order)
└── README.md
```

---

## ML Pipeline

### 1. Data Cleaning (`02data_cleaning.py`)
- Remove duplicates and listings with missing critical fields
- Filter out extreme price outliers (e.g. likely data entry errors)

### 2. Feature Transformation (`04feature_transformation.py`)
- Parse Turkish room notation (`2+1` → `oda_sayisi=2`, `salon_sayisi=1`, `toplam_oda=3`)
- Ordinal encode `bina_yasi`, `kat_sayisi`, `bulundugu_kat`
- Group and one-hot encode `isinma` (11 raw values → 6 categories)
- One-hot encode `mahalle` (17 neighbourhoods)
- Binary encode boolean amenities (`Evet`→`True`, `Hayır`→`False`)

### 3. Feature Engineering (`06feature_engineering.py`)
- **Target**: `fiyat_log = log1p(fiyat)` — log-transforms the price to reduce skewness
- `m2_per_room` = net_m2 / total_rooms
- `room_density` = total_rooms / net_m2
- `bath_per_room` = bathrooms / total_rooms
- `floor_ratio` = floor / total_floors (NaN if building has 0 floors)
- `is_top_floor`, `is_ground_floor` — binary flags
- `is_new` (building age ordinal == 0), `is_old` (>= 4)

### 4. Train/Test Split (`07train_test_split.py`)
- 80/20 stratified split, stratified on **mahalle** to preserve neighbourhood distribution

### 5. Model Training
- **Baseline**: Random Forest (`08`), XGBoost + LightGBM (`09`)
- **Tuning**: Optuna with 75 trials, 5-fold CV (`10catboost_tuning.py`)
- **Evaluation**: 5-fold Out-of-Fold predictions, then holdout test set (`11`, `12`)
- **Final model**: Retrained on full dataset (train + test) with tuned parameters (`13`)

### Best Parameters (CatBoost)
```python
CatBoostRegressor(
    iterations=559,
    learning_rate=0.036957,
    depth=5,
    subsample=0.930574,
    colsample_bylevel=0.709951,
    l2_leaf_reg=0.898362,
    min_data_in_leaf=9,
    random_seed=42,
)
```

### Model Performance
| Metric | OOF (train) | Holdout test |
|---|---|---|
| RMSE (log space) | 0.1841 | ~0.185 |
| R² | ~0.89 | ~0.88 |

---

## Getting Started

### Step 1 — Install dependencies
```bash
pip install fastapi uvicorn catboost pandas numpy pydantic optuna scikit-learn
```

### Step 2 — Start the prediction API

The pre-trained model (`models/catboost_production.cbm`) is included in the repository. No retraining required.

```bash
uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser to use the web UI.

---

### Adapting the pipeline to a different dataset

The training pipeline in `src/` is not tied to a specific city or platform. If you have a dataset that follows the same column schema (see [Column Reference](#column-reference)), you can run the full pipeline on it to train a model for a different location or property type.

```bash
python main.py
```

All 13 scripts in `src/` are executed in order. Progress and metrics are logged to `logs/pipeline_<timestamp>.log`. The pipeline expects the raw CSV to be placed at `data/raw/`.

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI (`index.html`) |
| `POST` | `/predict` | Returns predicted monthly rent |
| `GET` | `/options` | Returns dropdown options for the UI |

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "net_m2": 80,
    "oda_sayisi": 2,
    "salon_sayisi": 1,
    "banyo_sayisi": 1,
    "bina_yasi": 2,
    "bulundugu_kat": 3,
    "kat_sayisi": 2,
    "balkon": true,
    "asansor": true,
    "otopark": false,
    "esyali": false,
    "site_icerisinde": false,
    "isinma": "Doğalgaz",
    "mahalle": "Kayışdağı Mh."
  }'
```

```json
{"predicted_price": 34200}
```

### Input Validation Rules

| Field | Type | Constraint |
|---|---|---|
| `net_m2` | float | > 0 |
| `oda_sayisi` | float | ≥ 0 (supports 0.5 increments for half-rooms) |
| `salon_sayisi` | int | ≥ 0 |
| `banyo_sayisi` | float | ≥ 0 |
| `bina_yasi` | int | 0–6 (ordinal; see table above) |
| `bulundugu_kat` | int | 0–9 (ordinal; see table above) |
| `kat_sayisi` | int | 0–6 (ordinal) |
| `balkon` / `asansor` / `otopark` / `esyali` / `site_icerisinde` | bool | — |
| `isinma` | string | One of 6 valid heating categories |
| `mahalle` | string | One of 17 valid neighbourhoods |

> Cross-field rule: `bulundugu_kat` cannot exceed `kat_sayisi`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | `pandas`, `numpy` |
| ML model | `CatBoost` |
| Hyperparameter tuning | `Optuna` |
| API | `FastAPI` + `Uvicorn` |
| Validation | `Pydantic v2` |
| Frontend | Vanilla HTML/CSS/JS (single file) |

---

## Notes for International Readers

- **Turkish Lira (₺ / TRY)**: All prices are in Turkish Lira. As of December 2025, approximate exchange rate: 1 USD ≈ 42 TRY.
- **Ataşehir**: A modern, upper-middle-class district on the Asian side of Istanbul. Known for its financial centre, high-rise residences, and proximity to Yeditepe University.
