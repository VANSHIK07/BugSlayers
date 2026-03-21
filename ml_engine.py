import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
#  GEE SETUP — real satellite data with fallback
# ══════════════════════════════════════════════
GEE_AVAILABLE = False
try:
    import ee
    ee.Initialize(project='satellite-ml-490909')
    GEE_AVAILABLE = True
    print("✅ Google Earth Engine connected — using REAL satellite data")
except Exception as e:
    print(f"⚠️  GEE not available — using simulated data")


# ══════════════════════════════════════════════
#  AREA DEFINITIONS
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
#  GEE REAL DATA FUNCTIONS
# ══════════════════════════════════════════════

def gee_get_temperature(lat, lng, area_name):
    try:
        offset = 0.025
        bounds = ee.Geometry.Rectangle([lng-offset, lat-offset, lng+offset, lat+offset])
        lst = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate('2023-01-01', '2024-01-01') \
            .filterBounds(bounds).select('LST_Day_1km')
        def get_mean(img):
            mean = img.multiply(0.02).subtract(273.15) \
                      .reduceRegion(ee.Reducer.mean(), bounds, 1000)
            return ee.Feature(None, mean).set('date', img.date().format('YYYY-MM-dd'))
        features = lst.map(get_mean).getInfo()['features']
        rows = []
        for f in features:
            val = f['properties'].get('LST_Day_1km')
            dt  = f['properties'].get('date')
            if val is not None and dt:
                rows.append({'date': dt, 'temp': round(float(val), 2)})
        if len(rows) < 30:
            return None
        df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        print(f"  ✅ GEE: {len(df)} real MODIS readings for {area_name}")
        return df
    except Exception as e:
        print(f"  ⚠️  GEE temp failed: {e}")
        return None


def gee_get_no2_grid(lat, lng, area_name, n_points=80):
    try:
        offset = 0.025
        bounds = ee.Geometry.Rectangle([lng-offset, lat-offset, lng+offset, lat+offset])
        no2_img = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
            .filterDate('2023-04-01', '2023-10-01') \
            .filterBounds(bounds).select('NO2_column_number_density').mean()
        samples  = no2_img.sample(region=bounds, scale=1000, numPixels=n_points, geometries=True)
        features = samples.getInfo()['features']
        rows = []
        for f in features:
            val   = f['properties'].get('NO2_column_number_density')
            coord = f['geometry']['coordinates']
            if val is not None:
                rows.append({'lat': round(coord[1],5), 'lng': round(coord[0],5),
                             'no2': round(float(val)*1e6*46, 2)})
        if len(rows) < 10:
            return None
        print(f"  ✅ GEE: {len(rows)} real NO2 points for {area_name}")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠️  GEE NO2 failed: {e}")
        return None


# ══════════════════════════════════════════════
#  SIMULATION FALLBACK
# ══════════════════════════════════════════════

def simulate_temperature(city, area_type, seed):
    np.random.seed(seed)
    base  = {"Ahmedabad": 27, "Gandhinagar": 26}[city] + TYPE_TEMP_OFFSET[area_type]
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    temp  = base + 12 * np.sin((np.arange(365) - 90) * 2 * np.pi / 365)
    temp += np.random.normal(0, 1.5, 365)
    for _ in range(np.random.randint(2, 5)):
        temp[np.random.randint(0, 365)] += np.random.choice([-6, 8, 10, 12])
    return pd.DataFrame({'date': dates.astype(str), 'temp': temp.round(2)})


def simulate_no2_grid(lat, lng, area_type, seed, n_points=80):
    np.random.seed(seed + 1000)
    no2_base = 60 if area_type=='industrial' else 35 if area_type=='commercial' else 20
    lats = np.random.normal(lat, 0.008, n_points)
    lngs = np.random.normal(lng, 0.008, n_points)
    no2  = np.random.exponential(no2_base, n_points) + 15
    return pd.DataFrame({'lat': lats.round(5), 'lng': lngs.round(5), 'no2': no2.round(1)})


# ══════════════════════════════════════════════
#  ML HELPER FUNCTIONS
# ══════════════════════════════════════════════

def get_anomaly_severity(temp_val, mean_temp, std_temp):
    dev = abs(temp_val - mean_temp)
    if dev > 2 * std_temp: return "High"
    elif dev > std_temp:   return "Medium"
    else:                  return "Low"


def label_zone(cluster_id, avg_no2):
    if cluster_id == -1:
        return "Isolated Spots"
    if avg_no2 > 80:   return "Red Alert Area"
    elif avg_no2 > 50: return "High Risk Area"
    elif avg_no2 > 30: return "Moderate Risk Area"
    else:              return "Safe Area"


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


# ══════════════════════════════════════════════
#  MAIN ML ANALYSIS
# ══════════════════════════════════════════════

def analyze_area(city, area_name):
    area = AREAS[city][area_name]
    seed = AREA_SEEDS.get(area_name, 42)
    print(f"\n🔍 Analyzing {area_name}, {city}...")

    # Temperature data — real GEE or simulation
    temp_df     = None
    data_source = "Simulated"
    if GEE_AVAILABLE:
        temp_df = gee_get_temperature(area['lat'], area['lng'], area_name)
    if temp_df is not None and len(temp_df) >= 30:
        data_source = "Real — MODIS Satellite (NASA)"
        print(f"  📡 Using REAL temperature data")
    else:
        temp_df     = simulate_temperature(city, area['type'], seed)
        data_source = "Simulated (connect GEE for real satellite data)"
        print(f"  💻 Using simulated temperature data")

    temp = temp_df['temp'].values
    df   = temp_df[['date', 'temp']].copy()

    # MODEL 1 — Isolation Forest
    print(f"  🤖 Isolation Forest...")
    iso        = IsolationForest(contamination=0.02, random_state=42)
    df['flag'] = iso.fit_predict(df[['temp']])
    anomaly_df = df[df['flag'] == -1].copy()
    mean_temp  = float(temp.mean())
    std_temp   = float(temp.std())
    anomaly_df['severity'] = anomaly_df['temp'].apply(
        lambda t: get_anomaly_severity(t, mean_temp, std_temp)
    )
    anomalies  = anomaly_df[['date', 'temp', 'severity']].to_dict('records')
    worst_temp = round(float(anomaly_df['temp'].max()), 1) if len(anomaly_df) > 0 else None
    print(f"  ✅ {len(anomalies)} unusual days found")

    # MODEL 2 — ARIMA
    print(f"  🤖 ARIMA forecast...")
    forecast_vals = ARIMA(temp, order=(3, 1, 0)).fit().forecast(steps=30)
    forecast_list = [round(float(f), 1) for f in forecast_vals]
    trend         = calc_trend(list(forecast_vals))
    print(f"  ✅ Forecast: {round(float(forecast_vals.mean()),1)}°C, {trend}")

    # NO2 data — real GEE or simulation
    pts = None
    if GEE_AVAILABLE:
        pts = gee_get_no2_grid(area['lat'], area['lng'], area_name)
    if pts is None or len(pts) < 10:
        pts = simulate_no2_grid(area['lat'], area['lng'], area['type'], seed)

    # MODEL 3 — DBSCAN
    print(f"  🤖 DBSCAN clustering...")
    coords      = pts[['lat', 'lng']].values
    coords_norm = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
    pts['cluster'] = DBSCAN(eps=0.6, min_samples=3).fit(coords_norm).labels_
    cluster_no2_avg = pts.groupby('cluster')['no2'].mean().to_dict()
    pts['zone_label'] = pts['cluster'].apply(
        lambda c: label_zone(c, cluster_no2_avg.get(c, 0))
    )
    avg_no2    = round(float(pts['no2'].mean()), 1)
    n_clusters = int(max(pts['cluster'].max() + 1, 0))
    print(f"  ✅ {n_clusters} zones, avg NO2: {avg_no2}")

    avg_temp_val = round(mean_temp, 1)
    habitability = calc_habitability(avg_temp_val, avg_no2)
    risk         = calc_risk(len(anomalies), avg_temp_val, n_clusters)

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
    }


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
        })
    return results
