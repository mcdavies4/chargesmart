import requests
import json
from datetime import datetime
import time

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

REGIONS = {
    "uk":           (51.0, -2.0,  53.0,  0.5),
    "us_east":      (38.0,-77.0,  42.5, -70.0),
    "us_west":      (34.0,-122.5, 47.5,-117.0),
    "us_texas":     (25.5,-100.0, 36.5, -93.5),
    "us_midwest":   (40.0, -90.0, 43.5, -82.5),
    "norway":       (57.0,  4.0,  71.0,  31.0),
    "netherlands":  (50.7,  3.3,  53.6,   7.2),
    "germany":      (47.3,  5.9,  55.0,  15.0),
    "sweden":       (55.0, 10.9,  69.0,  24.2),
    "france":       (42.3, -4.8,  51.1,   8.2),
}

# Representative coords per region for weather lookup
REGION_WEATHER_COORDS = {
    "uk":          (51.5, -0.1),
    "us_east":     (40.7, -74.0),
    "us_west":     (37.7,-122.4),
    "us_texas":    (30.3, -97.7),
    "us_midwest":  (41.8, -87.6),
    "norway":      (59.9,  10.7),
    "netherlands": (52.4,   4.9),
    "germany":     (52.5,  13.4),
    "sweden":      (59.3,  18.1),
    "france":      (48.9,   2.3),
}

def get_weather(lat, lon):
    """Fetch current temperature and precipitation from Open-Meteo (free, no key)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,precipitation,weathercode"
            f"&timezone=auto"
        )
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            current = r.json().get("current", {})
            return {
                "temperature": current.get("temperature_2m", 15),
                "precipitation": current.get("precipitation", 0),
                "weathercode": current.get("weathercode", 0),
            }
    except Exception:
        pass
    return {"temperature": 15, "precipitation": 0, "weathercode": 0}

def get_calendar_features(dt):
    """Return calendar-based features for a datetime."""
    # UK bank holidays 2025-2026
    uk_bank_holidays = {
        "2025-01-01","2025-04-18","2025-04-21","2025-05-05",
        "2025-05-26","2025-08-25","2025-12-25","2025-12-26",
        "2026-01-01","2026-04-03","2026-04-06","2026-05-04",
        "2026-05-25","2026-08-31","2026-12-25","2026-12-28",
    }
    # US federal holidays 2025-2026
    us_holidays = {
        "2025-01-01","2025-01-20","2025-02-17","2025-05-26",
        "2025-06-19","2025-07-04","2025-09-01","2025-11-11",
        "2025-11-27","2025-12-25",
        "2026-01-01","2026-01-19","2026-02-16","2026-05-25",
        "2026-06-19","2026-07-04","2026-09-07","2026-11-11",
        "2026-11-26","2026-12-25",
    }

    date_str = dt.strftime("%Y-%m-%d")
    month = dt.month
    day = dt.day

    is_bank_holiday = 1 if (date_str in uk_bank_holidays or date_str in us_holidays) else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_december = 1 if month == 12 else 0
    is_school_holiday = 1 if (
        (month == 8) or
        (month == 7 and day >= 20) or
        (month == 12 and day >= 20) or
        (month == 4 and 1 <= day <= 14) or
        is_bank_holiday
    ) else 0

    return {
        "is_bank_holiday": is_bank_holiday,
        "is_school_holiday": is_school_holiday,
        "is_summer": is_summer,
        "is_december": is_december,
    }

def collect_region(name, bbox):
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:60];
    node["amenity"="charging_station"]({south},{west},{north},{east});
    out body;
    """
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=90)
        if response.status_code == 200:
            data = response.json()
            chargers = data.get("elements", [])
            print(f"  {name}: {len(chargers)} chargers")
            return chargers
        else:
            print(f"  {name}: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"  {name}: Error - {e}")
        return []

def get_country(name):
    if name.startswith('us_'):
        return 'US'
    elif name in ['norway', 'netherlands', 'germany', 'sweden', 'france']:
        return 'EU'
    return 'UK'

def collect_all():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date = now.strftime("%Y%m%d")
    print(f"\n[{timestamp}] Collecting all regions...")

    # Fetch weather per region
    weather_cache = {}
    print("  Fetching weather data...")
    for name, coords in REGION_WEATHER_COORDS.items():
        weather_cache[name] = get_weather(coords[0], coords[1])
        time.sleep(0.5)

    calendar = get_calendar_features(now)

    all_chargers = []
    for name, bbox in REGIONS.items():
        chargers = collect_region(name, bbox)
        weather = weather_cache.get(name, {"temperature": 15, "precipitation": 0, "weathercode": 0})
        for c in chargers:
            c['region'] = name
            c['country'] = get_country(name)
            c['temperature'] = weather['temperature']
            c['precipitation'] = weather['precipitation']
            c['weathercode'] = weather['weathercode']
            c['is_bank_holiday'] = calendar['is_bank_holiday']
            c['is_school_holiday'] = calendar['is_school_holiday']
            c['is_summer'] = calendar['is_summer']
            c['is_december'] = calendar['is_december']
        all_chargers.extend(chargers)
        time.sleep(3)

    snapshot = {"timestamp": timestamp, "chargers": all_chargers}
    filename = f"data_{date}.json"
    with open(filename, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    print(f"  Total: {len(all_chargers)} chargers saved to {filename}")
    w = weather_cache.get('uk', {})
    print(f"  UK Weather: {w.get('temperature')}°C, {w.get('precipitation')}mm precip")
    cal_flags = [k for k, v in calendar.items() if v == 1]
    print(f"  Calendar flags: {cal_flags if cal_flags else 'none'}")

if __name__ == "__main__":
    print("ChargeSmart Collector - UK + US + Europe")
    print("Now with weather + calendar data")
    print("Collecting every 10 minutes. Press Ctrl+C to stop.\n")
    while True:
        collect_all()
        print("  Waiting 10 minutes...")
        time.sleep(600)
