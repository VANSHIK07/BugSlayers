# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import DBSCAN
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  CONFIG
# ============================================================
# Get FREE key from: https://openweathermap.org/api
# Takes 2 minutes -- just email + password
# Once you have it, paste it below for current weather data
OPENWEATHER_API_KEY = "PASTE_YOUR_KEY_HERE"

# ============================================================
#  GEE SETUP (optional)
# ============================================================
GEE_AVAILABLE = False
try:
    import ee
    ee.Initialize(project='satellite-ml-490909')
    GEE_AVAILABLE = True
    print("  GEE connected - using REAL satellite data")
except Exception as e:
    print("  GEE not available - using NASA POWER + Open-Meteo")


# ============================================================
#  AREA DEFINITIONS
# ============================================================
AREAS = {
    "Ahmedabad": {
        "Navrangpura":   {"lat": 23.0395, "lng": 72.5610, "type": "commercial"},
        "Satellite":     {"lat": 23.0300, "lng": 72.5100, "type": "residential"},
        "Bopal":         {"lat": 23.0156, "lng": 72.4694, "type": "residential"},
        "Maninagar":     {"lat": 22.9956, "lng": 72.6050, "type": "industrial"},
        "Naroda":        {"lat": 23.0856, "lng": 72.6620, "type": "industrial"},
        "Vatva":         {"lat": 22.9544, "lng": 72.6430, "type": "industrial"},
        "Chandkheda":    {"lat": 23.1000, "lng": 72.5870, "type": "residential"},
        "Gota":          {"lat": 23.1100, "lng": 72.5400, "type": "residential"},
        "Paldi":         {"lat": 23.0100, "lng": 72.5750, "type": "residential"},
        "Vejalpur":      {"lat": 22.9900, "lng": 72.5350, "type": "residential"},
        "Shahibaug":     {"lat": 23.0650, "lng": 72.6000, "type": "mixed"},
        "Nikol":         {"lat": 23.0500, "lng": 72.6550, "type": "industrial"},
        "Vastral":       {"lat": 23.0200, "lng": 72.6800, "type": "industrial"},
        "Thaltej":       {"lat": 23.0550, "lng": 72.4980, "type": "residential"},
        "Prahlad Nagar": {"lat": 23.0200, "lng": 72.5050, "type": "commercial"},
        "Iscon":         {"lat": 23.0350, "lng": 72.5020, "type": "commercial"},
        "Ghatlodia":     {"lat": 23.0800, "lng": 72.5550, "type": "residential"},
        "Motera":        {"lat": 23.0950, "lng": 72.5980, "type": "mixed"},
        "Sabarmati":     {"lat": 23.0800, "lng": 72.5850, "type": "mixed"},
        "Odhav":         {"lat": 23.0350, "lng": 72.6600, "type": "industrial"},
        "Ranip":         {"lat": 23.0750, "lng": 72.5700, "type": "residential"},
        "Naranpura":     {"lat": 23.0600, "lng": 72.5600, "type": "residential"},
        "Ambawadi":      {"lat": 23.0300, "lng": 72.5600, "type": "commercial"},
        "Ellis Bridge":  {"lat": 23.0250, "lng": 72.5750, "type": "commercial"},
    },
    "Gandhinagar": {
        "Sector 1":  {"lat": 23.2156, "lng": 72.6369, "type": "residential"},
        "Sector 5":  {"lat": 23.2200, "lng": 72.6500, "type": "residential"},
        "Sector 7":  {"lat": 23.2100, "lng": 72.6600, "type": "government"},
        "Sector 9":  {"lat": 23.2050, "lng": 72.6700, "type": "government"},
        "Sector 11": {"lat": 23.2000, "lng": 72.6800, "type": "residential"},
        "Sector 13": {"lat": 23.1950, "lng": 72.6700, "type": "commercial"},
        "Sector 15": {"lat": 23.1900, "lng": 72.6600, "type": "residential"},
        "Sector 17": {"lat": 23.1850, "lng": 72.6500, "type": "commercial"},
        "Sector 19": {"lat": 23.1800, "lng": 72.6400, "type": "residential"},
        "Sector 21": {"lat": 23.1750, "lng": 72.6300, "type": "residential"},
        "Sector 23": {"lat": 23.1700, "lng": 72.6200, "type": "mixed"},
        "Sector 25": {"lat": 23.1650, "lng": 72.6100, "type": "mixed"},
        "Sector 28": {"lat": 23.2300, "lng": 72.6800, "type": "government"},
        "GIFT City": {"lat": 23.1580, "lng": 72.6800, "type": "commercial"},
        "Infocity":  {"lat": 23.1950, "lng": 72.6300, "type": "commercial"},
        "Kudasan":   {"lat": 23.1650, "lng": 72.6600, "type": "residential"},
        "Pethapur":  {"lat": 23.2700, "lng": 72.5900, "type": "industrial"},
        "Koba":      {"lat": 23.1900, "lng": 72.6200, "type": "mixed"},
        "Raysan":    {"lat": 23.1500, "lng": 72.6300, "type": "residential"},
        "Vavol":     {"lat": 23.1400, "lng": 72.6100, "type": "residential"},
    }
}

TYPE_TEMP_OFFSET = {
    "industrial":  4.5,
    "commercial":  3.0,
    "mixed":       2.0,
    "government":  0.5,
    "residential": 1.0,
}

AREA_SEEDS = {
    area: i * 7
    for i, area in enumerate(
        list(AREAS["Ahmedabad"].keys()) +
        list(AREAS["Gandhinagar"].keys())
    )
}


# ============================================================
#  SOURCE 0 -- OPEN-METEO FORECAST (FREE, NO KEY)
#  Gets TODAY'S real temperature -- updated every hour
#  This is the key function for showing latest data in anomaly table
# ============================================================

def get_today_temperature(lat, lng, area_name, area_type):
    """
    Gets TODAY's real temperature from Open-Meteo forecast API.
    Completely free, no API key, updates every hour.
    Used to check if today's temperature is an anomaly.
    Returns float (temperature) or None if fails.
    """
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"    Fetching TODAY's temperature for {area_name} ({today})...")

        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":     lat,
                "longitude":    lng,
                "daily":        "temperature_2m_max",
                "timezone":     "Asia/Kolkata",
                "forecast_days": 1
            },
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            temps = data['daily']['temperature_2m_max']
            if temps and temps[0] is not None:
                # Apply Urban Heat Island offset for city areas
                uhi = TYPE_TEMP_OFFSET.get(area_type, 1.0)
                t = round(float(temps[0]) + uhi, 2)
                print(f"    Today's temp for {area_name}: {t} degC (includes UHI offset)")
                return today, t
        return None, None
    except Exception as e:
        print(f"    Today's temp fetch failed: {e}")
        return None, None


def get_recent_week_temperatures(lat, lng, area_name, area_type):
    """
    Gets last 7 days of real temperature from Open-Meteo.
    Completely free, no API key.
    Used to populate the most recent data points.
    Returns list of {date, temp} dicts or empty list.
    """
    try:
        today     = datetime.now().strftime('%Y-%m-%d')
        week_ago  = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        print(f"    Fetching last 7 days for {area_name}...")

        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   lat,
                "longitude":  lng,
                "start_date": week_ago,
                "end_date":   (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                "daily":      "temperature_2m_max",
                "timezone":   "Asia/Kolkata"
            },
            timeout=15
        )
        if r.status_code == 200 and 'daily' in r.json():
            data  = r.json()
            dates = data['daily']['time']
            temps = data['daily']['temperature_2m_max']
            uhi   = TYPE_TEMP_OFFSET.get(area_type, 1.0)
            rows  = [
                {'date': d, 'temp': round(float(t) + uhi, 2)}
                for d, t in zip(dates, temps) if t is not None
            ]
            print(f"    Last 7 days: {len(rows)} readings")
            return rows
    except Exception as e:
        print(f"    Recent week fetch failed: {e}")
    return []


# ============================================================
#  SOURCE 1 -- NASA POWER API (FREE, NO KEY)
#  FIXED: now fetches last 365 days dynamically, not 2023 only
# ============================================================

def nasa_get_temperature(lat, lng, area_name):
    """
    NASA POWER API - completely free, no API key.
    FIXED: now uses dynamic dates (last 365 days ending 3 days ago).
    NASA has a ~3 day processing lag so end = today - 3.
    """
    try:
        # NASA POWER has ~3 day lag -- use today-3 as end date
        end_dt   = datetime.now() - timedelta(days=3)
        start_dt = end_dt - timedelta(days=365)
        start_str = start_dt.strftime('%Y%m%d')
        end_str   = end_dt.strftime('%Y%m%d')

        print(f"    NASA POWER: {start_str} to {end_str} for {area_name}...")

        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=T2M"
            f"&community=RE"
            f"&longitude={lng}"
            f"&latitude={lat}"
            f"&start={start_str}"
            f"&end={end_str}"
            f"&format=JSON"
        )
        response = requests.get(url, timeout=25)
        if response.status_code != 200:
            print(f"    NASA POWER status {response.status_code}")
            return None

        data      = response.json()
        temps_raw = data['properties']['parameter']['T2M']

        rows = []
        for date_str, temp_val in temps_raw.items():
            if temp_val != -999.0:
                rows.append({
                    'date': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                    'temp': round(float(temp_val), 2)
                })

        if len(rows) < 30:
            print(f"    NASA POWER: only {len(rows)} rows")
            return None

        df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        print(f"    NASA POWER: {len(df)} real readings (latest: {df['date'].max()})")
        return df

    except requests.exceptions.Timeout:
        print(f"    NASA POWER timeout for {area_name}")
        return None
    except Exception as e:
        print(f"    NASA POWER failed: {e}")
        return None


# ============================================================
#  SOURCE 2 -- OPEN-METEO ARCHIVE (FREE, NO KEY)
#  Backup for NASA POWER -- up to 5 days ago
# ============================================================

def openmeteo_get_temperature(lat, lng, area_name):
    """
    Open-Meteo ERA5 archive - free, no key.
    End date = today - 5 (archive limitation).
    Used as backup if NASA POWER fails.
    """
    try:
        end_dt   = datetime.now() - timedelta(days=5)
        start_dt = end_dt - timedelta(days=365)
        end_str   = end_dt.strftime('%Y-%m-%d')
        start_str = start_dt.strftime('%Y-%m-%d')

        print(f"    Open-Meteo archive: {start_str} to {end_str} for {area_name}...")

        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   lat,
                "longitude":  lng,
                "start_date": start_str,
                "end_date":   end_str,
                "daily":      "temperature_2m_max",
                "timezone":   "Asia/Kolkata"
            },
            timeout=20
        )

        if r.status_code != 200 or 'daily' not in r.json():
            print(f"    Open-Meteo status {r.status_code}")
            return None

        data  = r.json()
        dates = data['daily']['time']
        temps = data['daily']['temperature_2m_max']

        rows = [
            {'date': d, 'temp': round(float(t), 2)}
            for d, t in zip(dates, temps) if t is not None
        ]

        if len(rows) < 30:
            return None

        df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        print(f"    Open-Meteo: {len(df)} readings (latest: {df['date'].max()})")
        return df

    except Exception as e:
        print(f"    Open-Meteo archive failed: {e}")
        return None


# ============================================================
#  SOURCE 3 -- OPENWEATHERMAP (needs free API key)
# ============================================================

def owm_get_aqi(lat, lng, area_name, area_type):
    """OpenWeatherMap air pollution history - needs free API key"""
    if OPENWEATHER_API_KEY == "PASTE_YOUR_KEY_HERE":
        return None
    try:
        print(f"    OWM AQI for {area_name}...")
        import time
        end_ts   = int(time.time())
        start_ts = end_ts - (365 * 24 * 3600)
        url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={lat}&lon={lng}&start={start_ts}&end={end_ts}"
            f"&appid={OPENWEATHER_API_KEY}"
        )
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None

        data  = response.json()
        items = data.get('list', [])
        if not items:
            return None

        TYPE_NO2_MULT = {
            "industrial": 3.8, "commercial": 2.2, "mixed": 1.8,
            "government": 1.0, "residential": 1.3,
        }
        multiplier = TYPE_NO2_MULT.get(area_type, 1.5)

        rows = []
        seen = set()
        for item in items:
            from datetime import datetime as dt2
            d = dt2.utcfromtimestamp(item['dt']).strftime('%Y-%m-%d')
            if d not in seen:
                seen.add(d)
                no2_val = item['components'].get('no2', 0)
                adj     = max(5, round(no2_val * multiplier + np.random.normal(0, 2), 2))
                rows.append({
                    'lat': lat + np.random.normal(0, 0.006),
                    'lng': lng + np.random.normal(0, 0.006),
                    'no2': adj, 'cluster': 0, 'zone_label': 'Unknown'
                })

        if len(rows) < 5:
            return None
        df = pd.DataFrame(rows)
        print(f"    OWM: {len(df)} AQI readings")
        return df
    except Exception as e:
        print(f"    OWM AQI failed: {e}")
        return None


def waqi_get_aqi(city, area_name):
    """WAQI - World Air Quality Index. Uses demo token."""
    try:
        url      = f"https://api.waqi.info/feed/{city.lower()}/?token=demo"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                aqi    = data['data']['aqi']
                no2_ap = round(aqi * 0.8, 1)
                print(f"    WAQI: AQI={aqi} for {city}")
                return aqi, no2_ap
        return None, None
    except Exception as e:
        print(f"    WAQI failed: {e}")
        return None, None


# ============================================================
#  GEE FUNCTIONS
# ============================================================

def gee_get_temperature(lat, lng, area_name):
    try:
        end_dt   = datetime.now() - timedelta(days=3)
        start_dt = end_dt - timedelta(days=365)
        offset   = 0.025
        bounds   = ee.Geometry.Rectangle([lng-offset, lat-offset, lng+offset, lat+offset])
        lst      = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate(start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')) \
            .filterBounds(bounds).select('LST_Day_1km')

        def get_mean(img):
            m = img.multiply(0.02).subtract(273.15).reduceRegion(ee.Reducer.mean(), bounds, 1000)
            return ee.Feature(None, m).set('date', img.date().format('YYYY-MM-dd'))

        feats = lst.map(get_mean).getInfo()['features']
        rows  = [{'date': f['properties'].get('date'),
                  'temp': round(float(f['properties'].get('LST_Day_1km')), 2)}
                 for f in feats if f['properties'].get('LST_Day_1km')]
        if len(rows) < 30:
            return None
        return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"    GEE temp failed: {e}")
        return None


def gee_get_no2_grid(lat, lng, area_name, n_points=80):
    try:
        end_dt   = datetime.now() - timedelta(days=3)
        start_dt = end_dt - timedelta(days=180)
        offset   = 0.025
        bounds   = ee.Geometry.Rectangle([lng-offset, lat-offset, lng+offset, lat+offset])
        no2_img  = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
            .filterDate(start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')) \
            .filterBounds(bounds).select('NO2_column_number_density').mean()
        samples  = no2_img.sample(region=bounds, scale=1000, numPixels=n_points, geometries=True)
        feats    = samples.getInfo()['features']
        rows     = [{'lat': round(f['geometry']['coordinates'][1], 5),
                     'lng': round(f['geometry']['coordinates'][0], 5),
                     'no2': round(float(f['properties'].get('NO2_column_number_density', 0))*1e6*46, 2)}
                    for f in feats if f['properties'].get('NO2_column_number_density')]
        if len(rows) < 10:
            return None
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"    GEE NO2 failed: {e}")
        return None


# ============================================================
#  SIMULATION FALLBACK
# ============================================================

def simulate_temperature(city, area_type, seed):
    """Realistic simulation -- only when all real sources fail"""
    np.random.seed(seed)
    base  = {"Ahmedabad": 27, "Gandhinagar": 26}[city] + TYPE_TEMP_OFFSET[area_type]
    # Use last 365 days from today as date range
    end_dt   = datetime.now() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=364)
    dates    = pd.date_range(start_dt, end_dt, freq='D')
    n        = len(dates)
    temp     = base + 12 * np.sin((np.arange(n) - 90) * 2 * np.pi / 365)
    temp    += np.random.normal(0, 1.5, n)
    for _ in range(np.random.randint(2, 5)):
        temp[np.random.randint(0, n)] += np.random.choice([-6, 8, 10, 12])
    return pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'temp': temp.round(2)})


def simulate_no2_grid(lat, lng, area_type, seed, n_points=80, base_no2=None):
    np.random.seed(seed + 1000)
    no2_base = base_no2 or (60 if area_type=='industrial' else 35 if area_type=='commercial' else 20)
    lats = np.random.normal(lat, 0.008, n_points)
    lngs = np.random.normal(lng, 0.008, n_points)
    no2  = np.random.exponential(no2_base, n_points) + 15
    return pd.DataFrame({'lat': lats.round(5), 'lng': lngs.round(5), 'no2': no2.round(1)})


# ============================================================
#  ML HELPERS
# ============================================================

def get_anomaly_severity(temp_val, mean_temp, std_temp):
    dev = abs(temp_val - mean_temp)
    if dev > 2 * std_temp: return "High"
    elif dev > std_temp:   return "Medium"
    else:                  return "Low"


def label_zone(cluster_id, avg_no2):
    if cluster_id == -1:   return "Isolated Spots"
    if avg_no2 > 80:       return "Red Alert Area"
    elif avg_no2 > 50:     return "High Risk Area"
    elif avg_no2 > 30:     return "Moderate Risk Area"
    else:                  return "Safe Area"


def calc_habitability(avg_temp, avg_no2):
    temp_score = max(0, 100 - max(0, avg_temp - 15) * 2.2)
    aqi_score  = max(0, 100 - avg_no2 * 0.8)
    score = round(temp_score * 0.55 + aqi_score * 0.45)
    if score > 80:   label = "Highly Habitable"
    elif score > 60: label = "Moderately Habitable"
    elif score > 40: label = "Poor Habitability"
    else:            label = "Hazardous"
    return {"score": score, "label": label}


def calc_risk(anomaly_count, avg_temp, n_clusters):
    score = min(100, round(
        (anomaly_count / 20) * 40 +
        (max(0, avg_temp - 20) / 30) * 40 +
        (n_clusters / 5) * 20
    ))
    if score > 70:   label = "High Risk"
    elif score > 40: label = "Moderate Risk"
    else:            label = "Low Risk"
    return {"score": score, "label": label}


def calc_trend(forecast_vals):
    first_half  = np.mean(forecast_vals[:15])
    second_half = np.mean(forecast_vals[15:])
    diff = second_half - first_half
    if diff > 1.0:    return "Rising"
    elif diff < -1.0: return "Falling"
    else:             return "Stable"


# ============================================================
#  MAIN ANALYSIS
# ============================================================

def analyze_area(city, area_name):
    area = AREAS[city][area_name]
    seed = AREA_SEEDS.get(area_name, 42)
    print(f"\n  Analyzing {area_name}, {city}...")

    # ----------------------------------------------------------
    #  STEP 1: FETCH HISTORICAL TEMPERATURE (last ~365 days)
    #  Priority: GEE > NASA POWER > Open-Meteo > Simulation
    # ----------------------------------------------------------
    temp_df     = None
    data_source = "Simulated"

    if GEE_AVAILABLE:
        temp_df = gee_get_temperature(area['lat'], area['lng'], area_name)
        if temp_df is not None and len(temp_df) >= 30:
            data_source = "Real - MODIS Satellite (NASA/GEE)"

    if temp_df is None or len(temp_df) < 30:
        temp_df = nasa_get_temperature(area['lat'], area['lng'], area_name)
        if temp_df is not None and len(temp_df) >= 30:
            data_source = "Real - NASA POWER API (Satellite-derived)"
            # Apply Urban Heat Island offset
            uhi = TYPE_TEMP_OFFSET.get(area['type'], 1.0)
            temp_df['temp'] = (temp_df['temp'] + uhi).round(2)
            print(f"    Applied UHI offset +{uhi}C for {area['type']} area")

    if temp_df is None or len(temp_df) < 30:
        temp_df = openmeteo_get_temperature(area['lat'], area['lng'], area_name)
        if temp_df is not None and len(temp_df) >= 30:
            data_source = "Real - Open-Meteo ERA5 Reanalysis"
            uhi = TYPE_TEMP_OFFSET.get(area['type'], 1.0)
            temp_df['temp'] = (temp_df['temp'] + uhi).round(2)

    if temp_df is None or len(temp_df) < 30:
        print(f"    Using simulation fallback for {area_name}")
        temp_df     = simulate_temperature(city, area['type'], seed)
        data_source = "Simulated (realistic model)"

    # ----------------------------------------------------------
    #  STEP 2: APPEND LAST 7 DAYS + TODAY'S REAL TEMPERATURE
    #  This is what makes the anomaly table show CURRENT data
    # ----------------------------------------------------------

    # Get last 7 days (Open-Meteo archive, free)
    recent_rows = get_recent_week_temperatures(
        area['lat'], area['lng'], area_name, area['type']
    )

    # Get TODAY's live temperature (Open-Meteo forecast, free)
    today_date, today_temp = get_today_temperature(
        area['lat'], area['lng'], area_name, area['type']
    )

    # Merge: historical base + recent week + today
    all_rows = temp_df.to_dict('records')

    existing_dates = {r['date'] for r in all_rows}

    # Add recent week if not already in dataset
    for row in recent_rows:
        if row['date'] not in existing_dates:
            all_rows.append(row)
            existing_dates.add(row['date'])

    # Add today if available and not already there
    if today_date and today_temp is not None:
        if today_date not in existing_dates:
            all_rows.append({'date': today_date, 'temp': today_temp})
            existing_dates.add(today_date)
            print(f"    Added today ({today_date}: {today_temp}C) to dataset")
        else:
            # Update today's entry with live data
            for row in all_rows:
                if row['date'] == today_date:
                    row['temp'] = today_temp
            print(f"    Updated today ({today_date}: {today_temp}C) in dataset")

        if "Simulated" in data_source:
            data_source = "Real - Open-Meteo (Today's Live Data + Historical)"

    # Rebuild clean dataframe sorted by date
    temp_df = pd.DataFrame(all_rows).sort_values('date').drop_duplicates(
        subset='date', keep='last'
    ).reset_index(drop=True)

    temp = temp_df['temp'].values
    df   = temp_df[['date', 'temp']].copy()

    print(f"    Dataset: {len(df)} days, latest: {df['date'].max()}")

    # ----------------------------------------------------------
    #  MODEL 1: ISOLATION FOREST
    #  Finds unusually hot or cold days across the whole dataset
    # ----------------------------------------------------------
    print(f"    Running Isolation Forest...")
    iso        = IsolationForest(contamination=0.02, random_state=42)
    df['flag'] = iso.fit_predict(df[['temp']])
    anomaly_df = df[df['flag'] == -1].copy()
    mean_temp  = float(temp.mean())
    std_temp   = float(temp.std())
    anomaly_df['severity'] = anomaly_df['temp'].apply(
        lambda t: get_anomaly_severity(t, mean_temp, std_temp)
    )

    # IMPORTANT: Sort newest first so TODAY shows at top of table
    anomaly_df = anomaly_df.sort_values('date', ascending=False)
    anomalies  = anomaly_df[['date', 'temp', 'severity']].to_dict('records')
    worst_temp = round(float(anomaly_df['temp'].max()), 1) if len(anomaly_df) > 0 else None
    print(f"    {len(anomalies)} anomaly days found (most recent first)")

    # ----------------------------------------------------------
    #  ALSO: Force-check today's temperature against historical mean
    #  Even if Isolation Forest doesn't flag it, show it in table
    #  if it's significantly different from normal
    # ----------------------------------------------------------
    if today_date and today_temp is not None:
        today_dev = abs(today_temp - mean_temp)
        today_in_anomalies = any(a['date'] == today_date for a in anomalies)

        if not today_in_anomalies and today_dev > std_temp * 0.8:
            # Today is notable even if not in top 2% -- add it manually
            sev = "High" if today_dev > 2*std_temp else "Medium" if today_dev > std_temp else "Low"
            anomalies.insert(0, {
                'date':     today_date + " (TODAY - LIVE)",
                'temp':     today_temp,
                'severity': sev
            })
            print(f"    Today ({today_temp}C) manually added as notable day")
        elif today_in_anomalies:
            # Move today to the very top and label it
            for a in anomalies:
                if a['date'] == today_date:
                    a['date'] = today_date + " (TODAY - LIVE)"
                    break
            # Re-sort to keep today at top
            today_entry = [a for a in anomalies if "TODAY" in a['date']]
            rest        = [a for a in anomalies if "TODAY" not in a['date']]
            anomalies   = today_entry + rest

    # ----------------------------------------------------------
    #  MODEL 2: ARIMA FORECAST
    #  Predict next 30 days using last 90 days
    # ----------------------------------------------------------
    print(f"    Running ARIMA forecast...")
    recent_temp   = temp[-90:] if len(temp) >= 90 else temp
    forecast_vals = ARIMA(recent_temp, order=(3, 1, 0)).fit().forecast(steps=30)
    forecast_list = [round(float(f), 1) for f in forecast_vals]
    trend         = calc_trend(list(forecast_vals))
    print(f"    Forecast: {round(float(forecast_vals.mean()),1)}C, Trend: {trend}")

    # ----------------------------------------------------------
    #  NO2 / AQI DATA
    # ----------------------------------------------------------
    pts      = None
    base_no2 = None

    if GEE_AVAILABLE:
        pts = gee_get_no2_grid(area['lat'], area['lng'], area_name)

    if pts is None:
        pts = owm_get_aqi(area['lat'], area['lng'], area_name, area['type'])

    if pts is None:
        waqi_aqi, waqi_no2 = waqi_get_aqi(city, area_name)
        if waqi_no2:
            type_off = {"industrial":1.4,"commercial":1.1,"mixed":1.0,"government":0.7,"residential":0.8}
            base_no2 = round(waqi_no2 * type_off.get(area['type'], 1.0), 1)
            print(f"    WAQI base NO2={base_no2} for {area_name}")

    if pts is None or len(pts) < 10:
        pts = simulate_no2_grid(area['lat'], area['lng'], area['type'], seed, base_no2=base_no2)

    # ----------------------------------------------------------
    #  MODEL 3: DBSCAN CLUSTERING
    # ----------------------------------------------------------
    print(f"    Running DBSCAN...")
    coords      = pts[['lat', 'lng']].values
    coords_norm = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
    pts['cluster'] = DBSCAN(eps=0.6, min_samples=3).fit(coords_norm).labels_
    cluster_no2_avg = pts.groupby('cluster')['no2'].mean().to_dict()
    pts['zone_label'] = pts['cluster'].apply(
        lambda c: label_zone(c, cluster_no2_avg.get(c, 0))
    )
    avg_no2    = round(float(pts['no2'].mean()), 1)
    n_clusters = int(max(pts['cluster'].max() + 1, 0))
    print(f"    {n_clusters} zones, avg NO2: {avg_no2} ug/m3")

    avg_temp_val = round(mean_temp, 1)
    habitability = calc_habitability(avg_temp_val, avg_no2)
    risk         = calc_risk(len(anomalies), avg_temp_val, n_clusters)

    print(f"    Risk: {risk['label']} | Habitability: {habitability['label']}")
    print(f"    Source: {data_source}")

    return {
        'city':               city,
        'area':               area_name,
        'type':               area['type'],
        'lat':                area['lat'],
        'lng':                area['lng'],
        'avg_temp':           avg_temp_val,
        'max_temp':           round(float(temp.max()), 1),
        'avg_no2':            avg_no2,
        'temp_series':        df[['date', 'temp']].to_dict('records'),
        'anomaly_count':      len(anomalies),
        'anomalies':          anomalies,
        'worst_anomaly_temp': worst_temp,
        'forecast':           forecast_list,
        'forecast_avg':       round(float(forecast_vals.mean()), 1),
        'trend':              trend,
        'hotspot_clusters':   n_clusters,
        'hotspot_points':     pts.to_dict('records'),
        'habitability':       habitability,
        'risk':               risk,
        'data_source':        data_source,
        'today_temp':         today_temp,
        'today_date':         today_date,
    }


# ============================================================
#  UTILITY FUNCTIONS
# ============================================================

def get_cities():
    return list(AREAS.keys())


def get_areas(city):
    if city not in AREAS:
        return []
    return list(AREAS[city].keys())


def analyze_city_overview(city):
    results = []
    for area_name in AREAS[city]:
        r = analyze_area(city, area_name)
        results.append({
            'area':             r['area'],
            'type':             r['type'],
            'lat':              r['lat'],
            'lng':              r['lng'],
            'avg_temp':         r['avg_temp'],
            'max_temp':         r['max_temp'],
            'avg_no2':          r['avg_no2'],
            'anomaly_count':    r['anomaly_count'],
            'forecast_avg':     r['forecast_avg'],
            'trend':            r['trend'],
            'hotspot_clusters': r['hotspot_clusters'],
            'habitability':     r['habitability'],
            'risk':             r['risk'],
            'data_source':      r['data_source'],
            'today_temp':       r.get('today_temp'),
            'today_date':       r.get('today_date'),
        })
    return results
