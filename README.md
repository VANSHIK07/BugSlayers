# 🌿 GreenGrid — Satellite Environmental Intelligence Platform for Smart Cities

> **Problem Statement 4 — Sustainable Environment**
> *Hackathon Submission*

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![ML](https://img.shields.io/badge/ML-IsolationForest%20%7C%20ARIMA%20%7C%20DBSCAN-green?style=flat-square)
![Satellite](https://img.shields.io/badge/Data-NASA%20POWER%20%7C%20MODIS%20%7C%20OpenWeatherMap-orange?style=flat-square)
![AI](https://img.shields.io/badge/AI-Claude%20by%20Anthropic-purple?style=flat-square)
![Language](https://img.shields.io/badge/Language-English%20%7C%20%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80-red?style=flat-square)

---

## 🎯 The Problem We Were Given

Earth observation satellites from **ISRO, NASA, and ESA** collect massive amounts of environmental data every single day — temperature, air pollution, vegetation health, soil moisture, and more.

Yet:
- 🚫 Civic bodies and urban planners **have no tools** to translate this into decisions
- 🚫 Satellite datasets are **too technical** for municipal corporations to use
- 🚫 Cities face heat islands, air pollution, land degradation — **all visible from space, none addressed systematically**
- 🚫 The data exists. The tools do not.

**Our job: Build the tools.**

---

## ✅ What We Built — GreenGrid

GreenGrid is a **unified Satellite Environmental Intelligence Platform** that:

- 📡 Ingests real data from **NASA POWER, MODIS (via Google Earth Engine), OpenWeatherMap, and WAQI**
- 🤖 Runs **3 ML models** — anomaly detection, time-series forecasting, and pollution zone clustering
- 🗺️ Generates **interactive geospatial pollution hotspot maps**
- 🧠 Outputs a **practical AI-generated Environment Action Plan** using Claude AI
- 📄 Produces a **downloadable PDF report** ready for municipal commissioners
- 🗣️ Works in both **English and Hindi** — so even villagers can understand their environment

---

## 🖥️ Live Demo — How It Works

### 1. Home Page — City Overview Map
The landing page shows an **interactive map of Ahmedabad and Gandhinagar** with all monitored areas colour-coded by risk level. Users can click any area to open its full dashboard.

### 2. ML Dashboard — Area Deep Dive
Each area gets a full analysis page showing:

| Section | What it shows |
|---|---|
| 🌡️ Temperature Card | Average temp, max temp, badge (Normal / Elevated / Critical) |
| 💨 Air Quality Card | NO₂ level in μg/m³, pollution severity |
| 💧 Humidity Index | Moisture percentage in the air |
| 🔥 Heat Index | Feels-like temperature for humans |
| 🎯 Risk Score | 0–100 score with colour-coded radial dial |
| 🏠 Habitability Index | How safe and comfortable this area is to live in |
| 📈 Temperature Chart | Full year of data + 30-day forecast |
| 📊 Pollution Zone Chart | NO₂ levels across all zone types |
| 🗺️ Hotspot Map | Satellite map with colour-coded pollution dots |
| 📋 Anomaly Table | Every dangerous day listed with severity |
| 🤖 AI Action Plan | Claude AI gives a 30-day plan for the area |
| 📄 PDF Report | One-click download for government submission |

---

## 🤖 Machine Learning Models — Exactly What the Problem Statement Asked For

### Model 1 — Isolation Forest (Anomaly Detection)
**What the problem asked:** *"ML analytics — anomaly detection"*

Isolation Forest scans the entire year of daily temperature data and identifies days that were **statistically unusual** — dangerous heat waves or unexpected cold snaps. Each anomaly is labelled Low / Medium / High severity based on how far it deviates from the yearly mean.

```
Input:  365 days of satellite temperature data
Output: List of dangerous days with severity score
```

### Model 2 — ARIMA (3,1,0) — Time-Series Forecasting
**What the problem asked:** *"trend analysis, predictive modelling"*

ARIMA is trained on the past year of temperature data to **predict the next 30 days**. It also calculates whether the trend is Rising, Falling, or Stable — giving planners advance warning of coming heat stress.

```
Input:  365 days of temperature values
Output: 30-day forecast + Rising / Falling / Stable trend label
```

### Model 3 — DBSCAN Clustering — Pollution Zone Mapping
**What the problem asked:** *"hotspot identification, clustering"*

DBSCAN groups NO₂ sensor readings across the area into geographic clusters and labels them by danger level: Red Alert, High Risk, Moderate Risk, Safe Area. This creates the **pollution hotspot map** that officials can act on immediately.

```
Input:  Grid of lat/lng points with NO₂ readings
Output: Colour-coded pollution zones on interactive map
```

---

## 📡 Satellite & Data Sources

| Source | Data Provided | Cost |
|---|---|---|
| **NASA POWER API** | Daily temperature (T2M) for any lat/lng on Earth | Free, no key needed |
| **MODIS via Google Earth Engine** | Land surface temperature, NDVI vegetation health | Free with Google account |
| **OpenWeatherMap Air Pollution API** | Real-time NO₂, PM2.5, AQI | Free API key |
| **WAQI (World Air Quality Index)** | Backup AQI / NO₂ data | Free |
| **Sentinel-5P** | Atmospheric NO₂ column data | Free via GEE |

**Data priority system** — the app always tries to get real satellite data first:
```
GEE (MODIS) → NASA POWER → OpenWeatherMap → WAQI → Realistic Simulation
```
If all APIs are unavailable, it falls back to a statistically accurate simulation based on known climate patterns for Gujarat.

---

## 🗺️ Cities & Areas Covered

### Ahmedabad — 24 Areas
Navrangpura, Satellite, Bopal, Maninagar, Naroda, Vatva, Chandkheda, Gota, Paldi, Vejalpur, Shahibaug, Nikol, Vastral, Thaltej, Prahlad Nagar, Iscon, Ghatlodia, Motera, Sabarmati, Odhav, Ranip, Naranpura, Ambawadi, Ellis Bridge

### Gandhinagar — 20 Areas
Sector 1, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 28, GIFT City, Infocity, Kudasan, Pethapur, Koba, Raysan, Vavol

Each area has its coordinates, land use type (industrial / commercial / residential / government / mixed), and Urban Heat Island offset applied to temperature readings.

---

## 🧠 AI Environment Action Plan

When a user clicks **"Get AI Advice"**, the app sends the full ML analysis to **Claude AI (Anthropic)** which generates a structured 30-day action plan covering:

1. **Current Situation** — plain language summary
2. **Immediate Actions Needed** — 3–4 specific steps
3. **30-Day Action Plan** — Week 1 / Week 2–3 / Week 4
4. **Who Should Act** — AMC, GPCB, RWAs, factories
5. **Expected Improvement** — measurable targets

If the Anthropic API is unavailable, a built-in rule-based fallback provides a detailed plan based on risk score thresholds. The plan is also included in the PDF report.

---

## 🗣️ Hindi Language Support

Designed specifically so **villagers and non-English speakers** can understand their environment.

Click **हिन्दी** → everything switches to Hindi instantly:
- All headings, labels, descriptions
- Badge values (सामान्य / उच्च / गंभीर)
- Pollution zone names on the map
- Anomaly table rows
- AI loading messages and button states

Click **English** → switches back. Works 100% offline, no Google Translate, no API needed.

---

## 📄 PDF Report — Ready for Government Submission

One click generates a professional PDF report containing:
- Area overview with all ML metrics
- Anomaly table with severity colours
- AI-generated 30-day action plan
- Methodology section (Isolation Forest, ARIMA, DBSCAN explained)
- Data sources and disclaimer
- Reference number, date, prepared-by footer

Designed to be handed directly to a **municipal commissioner** or **GPCB officer**.

---

## 🗂️ Project Structure

```
greengrid/
│
├── app.py                  # Flask backend — API routes, AI advice, PDF report
├── ml_engine.py            # All ML models + satellite data fetching
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore              # Prevents secrets from being uploaded
├── README.md               # This file
│
└── templates/
    ├── home.html           # Landing page — interactive city map
    └── dashboard.html      # ML dashboard — charts, map, Hindi, AI
```

---

## ⚙️ How to Run

### Step 1 — Clone
```bash
git clone https://github.com/YOUR_USERNAME/greengrid.git
cd greengrid
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set API keys
```bash
cp .env.example .env
```
Open `.env` and fill in:
- `OPENWEATHER_API_KEY` — free from https://openweathermap.org/api
- `ANTHROPIC_API_KEY` — free from https://console.anthropic.com

> Both are optional. The app works without them using fallback data and rule-based AI.

### Step 4 — Run
```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🌐 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Home page — city map |
| `GET /dashboard?city=X&area=Y` | ML Dashboard |
| `GET /api/analyze/Ahmedabad/Navrangpura` | Full ML analysis (JSON) |
| `GET /api/advice/Ahmedabad/Navrangpura` | AI action plan (JSON) |
| `GET /api/report/Ahmedabad/Navrangpura` | Download PDF report |
| `GET /api/areas/Ahmedabad` | List all areas for a city |
| `GET /api/health` | Server health check |
| `GET /api/refresh` | Clear ML cache manually |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, Flask, Flask-CORS |
| ML Models | scikit-learn (Isolation Forest, DBSCAN), statsmodels (ARIMA) |
| Satellite Data | NASA POWER API, Google Earth Engine, OpenWeatherMap |
| AI | Claude AI by Anthropic (with rule-based fallback) |
| Maps | Leaflet.js + Esri Satellite tiles |
| Charts | Chart.js |
| PDF | ReportLab |
| Caching | Python LRU Cache + APScheduler (midnight auto-refresh) |
| Frontend | Vanilla HTML/CSS/JS — no framework needed |
| Fonts | Noto Sans Devanagari (Hindi support) |

---

## 🏆 How We Address Every Point in the Problem Statement

| Problem Statement Requirement | How GreenGrid Solves It |
|---|---|
| Ingest data from multiple satellite missions | NASA POWER + MODIS (GEE) + Sentinel-5P + OpenWeatherMap |
| Harmonize to common spatial grid | lat/lng referenced grid for every area with consistent coordinate system |
| Time-series environmental database | 365 days of daily temperature per area, queryable by city and area |
| ML analytics — anomaly detection | ✅ Isolation Forest |
| ML analytics — trend analysis | ✅ ARIMA forecast + Rising/Falling/Stable label |
| ML analytics — hotspot identification | ✅ DBSCAN clustering → Red Alert / High Risk / Moderate / Safe zones |
| Interactive geospatial maps | ✅ Leaflet.js pollution hotspot map with satellite base layer |
| Urban Heat Island analysis | ✅ UHI offset applied per area type (industrial areas +4.5°C) |
| Practical Environment Action Plan | ✅ Claude AI generates 30-day plan linked to real satellite findings |
| Output for municipal use | ✅ PDF report ready for government submission |

---

## 📸 Screenshots


![WhatsApp Image 2026-03-22 at 8 53 48 AM](https://github.com/user-attachments/assets/a7fa1200-39d6-4a8d-8bef-bf15209ff41c)
![WhatsApp Image 2026-03-22 at 8 53 34 AM](https://github.com/user-attachments/assets/c8e28541-a2b1-4f49-950a-4d263f50458e)
![WhatsApp Image 2026-03-22 at 9 18 20 AM](https://github.com/user-attachments/assets/61e79efc-f32f-401f-9ac5-bcb690bcedf3)


---

## 👥 Team
Built for **Sustainable Environment — Problem Statement 4**
<h1>**Lakkad Vanshik**
**Harshil Gajera**
**Kathiriya Meet**
**Vamja Fenil**
**Kothari Divisha**
</h1>


---

## 📜 License

MIT License — free to use, modify, and distribute.

---

*"Every city in India has satellite data being collected over it daily — none of it is being used by municipal corporations. GreenGrid closes that gap."*
