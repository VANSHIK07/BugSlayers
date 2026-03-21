import requests
import pandas as pd

def get_air_quality(city_name):
    # Free API - no key needed!
    # Delhi coordinates as example
    cities = {
        "Delhi": (28.6139, 77.209),
        "Mumbai": (19.0760, 72.8777),
        "Bengaluru": (12.9716, 77.5946),
        "Chennai": (13.0827, 80.2707)
    }
    
    lat, lon = cities[city_name]
    
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide&past_days=7"
    
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "pm2_5": data["hourly"]["pm2_5"],
        "nitrogen_dioxide": data["hourly"]["nitrogen_dioxide"],
        "pm10": data["hourly"]["pm10"]
    })
    return df