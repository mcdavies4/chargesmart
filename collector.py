import requests
import json
from datetime import datetime
import time

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

REGIONS = {
    # ── EXISTING ──────────────────────────────────────────────
    "uk":              (51.0,  -2.0,  53.0,   0.5),
    "us_east":         (38.0, -77.0,  42.5, -70.0),
    "us_west":         (34.0,-122.5,  47.5,-117.0),
    "us_texas":        (25.5,-100.0,  36.5, -93.5),
    "us_midwest":      (40.0, -90.0,  43.5, -82.5),
    "norway":          (57.0,   4.0,  71.0,  31.0),
    "netherlands":     (50.7,   3.3,  53.6,   7.2),
    "germany":         (47.3,   5.9,  55.0,  15.0),
    "sweden":          (55.0,  10.9,  69.0,  24.2),
    "france":          (42.3,  -4.8,  51.1,   8.2),

    # ── AFRICA ────────────────────────────────────────────────
    "south_africa":    (-34.8,  16.5, -22.1,  33.0),  # Full SA
    "kenya":           ( -4.7,  33.9,   4.6,  41.9),  # Full Kenya
    "nigeria":         (  4.3,   2.7,  13.9,  14.7),  # Full Nigeria
    "egypt":           ( 22.0,  24.7,  31.7,  37.1),  # Full Egypt
    "ethiopia":        (  3.4,  33.0,  15.0,  48.0),  # Full Ethiopia
    "morocco":         ( 27.6, -13.2,  35.9,  -1.1),  # Full Morocco
    "ghana":           (  4.7,  -3.3,  11.2,   1.2),  # Full Ghana
    "rwanda":          ( -2.9,  28.8,  -1.0,  30.9),  # Full Rwanda
    "tanzania":        (-11.7,  29.3,  -0.9,  40.5),  # Full Tanzania

    # ── MIDDLE EAST ───────────────────────────────────────────
    "uae":             ( 22.6,  51.6,  26.1,  56.4),  # UAE
    "saudi_arabia":    ( 16.4,  36.5,  32.2,  55.7),  # Saudi Arabia
    "israel":          ( 29.5,  34.2,  33.3,  35.9),  # Israel

    # ── SOUTH / SOUTHEAST ASIA ────────────────────────────────
    "india_south":     (  8.0,  76.0,  15.0,  80.5),  # Bangalore/Chennai corridor
    "india_west":      ( 18.9,  72.8,  23.1,  73.2),  # Mumbai/Ahmedabad
    "indonesia":       ( -8.8, 106.6,  -6.1, 112.0),  # Java island
    "thailand":        ( 13.6, 100.3,  14.1, 100.9),  # Bangkok region
    "vietnam":         ( 10.6, 106.5,  21.1, 107.2),  # Ho Chi Minh + Hanoi

    # ── LATIN AMERICA ─────────────────────────────────────────
    "brazil_southeast":(-23.8, -46.9, -19.8, -43.1),  # São Paulo/Rio
    "chile":           (-36.0, -72.5, -29.9, -69.5),  # Santiago corridor
    "colombia":        (  3.8, -77.2,   7.2, -72.5),  # Bogotá region
}

REGION_META = {
    # country code, weather coords, continent
    "uk":               ("UK",  (51.5,  -0.1), "Europe"),
    "us_east":          ("US",  (40.7, -74.0), "Americas"),
    "us_west":          ("US",  (37.7,-122.4), "Americas"),
    "us_texas":         ("US",  (30.3, -97.7), "Americas"),
    "us_midwest":       ("US",  (41.8, -87.6), "Americas"),
    "norway":           ("EU",  (59.9,  10.7), "Europe"),
    "netherlands":      ("EU",  (52.4,   4.9), "Europe"),
    "germany":          ("EU",  (52.5,  13.4), "Europe"),
    "sweden":           ("EU",  (59.3,  18.1), "Europe"),
    "france":           ("EU",  (48.9,   2.3), "Europe"),
    "south_africa":     ("ZA",  (-26.2,  28.0), "Africa"),
    "kenya":            ("KE",  ( -1.3,  36.8), "Africa"),
    "nigeria":          ("NG",  (  6.5,   3.4), "Africa"),
    "egypt":            ("EG",  ( 30.0,  31.2), "Africa"),
    "ethiopia":         ("ET",  (  9.0,  38.7), "Africa"),
    "morocco":          ("MA",  ( 33.9,  -6.9), "Africa"),
    "ghana":            ("GH",  (  5.6,  -0.2), "Africa"),
    "rwanda":           ("RW",  ( -1.9,  30.1), "Africa"),
    "tanzania":         ("TZ",  ( -6.8,  39.3), "Africa"),
    "uae":              ("AE",  ( 25.2,  55.3), "Middle East"),
    "saudi_arabia":     ("SA",  ( 24.7,  46.7), "Middle East"),
    "israel":           ("IL",  ( 31.8,  35.2), "Middle East"),
    "india_south":      ("IN",  ( 12.9,  77.6), "Asia"),
    "india_west":       ("IN",  ( 19.1,  72.9), "Asia"),
    "indonesia":        ("ID",  ( -6.2, 106.8), "Asia"),
    "thailand":         ("TH",  ( 13.8, 100.5), "Asia"),
    "vietnam":          ("VN",  ( 10.8, 106.7), "Asia"),
    "brazil_southeast": ("BR",  (-23.5, -46.6), "Americas"),
    "chile":            ("CL",  (-33.5, -70.6), "Americas"),
    "colombia":         ("CO",  (  4.7, -74.1), "Americas"),
}

def get_weather(lat, lon):
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
                "temperature":   current.get("temperature_2m", 20),
                "precipitation": current.get("precipitation", 0),
                "weathercode":   current.get("weathercode", 0),
            }
    except Exception:
        pass
    return {"temperature": 20, "precipitation": 0, "weathercode": 0}

def get_calendar_features(dt):
    uk_bank_holidays = {
        "2025-01-01","2025-04-18","2025-04-21","2025-05-05",
        "2025-05-26","2025-08-25","2025-12-25","2025-12-26",
        "2026-01-01","2026-04-03","2026-04-06","2026-05-04",
        "2026-05-25","2026-08-31","2026-12-25","2026-12-28",
    }
    date_str = dt.strftime("%Y-%m-%d")
    month    = dt.month
    day      = dt.day
    is_bank_holiday   = 1 if date_str in uk_bank_holidays else 0
    is_summer         = 1 if month in [6, 7, 8] else 0
    is_december       = 1 if month == 12 else 0
    is_school_holiday = 1 if (
        month == 8 or (month == 7 and day >= 20) or
        (month == 12 and day >= 20) or
        (month == 4 and 1 <= day <= 14) or is_bank_holiday
    ) else 0
    return {
        "is_bank_holiday":   is_bank_holiday,
        "is_school_holiday": is_school_holiday,
        "is_summer":         is_summer,
        "is_december":       is_december,
    }

def collect_region(name, bbox):
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:90];
    node["amenity"="charging_station"]({south},{west},{north},{east});
    out body;
    """
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
        if response.status_code == 200:
            chargers = response.json().get("elements", [])
            print(f"  {name:20s}: {len(chargers):4d} chargers")
            return chargers
        else:
            print(f"  {name:20s}: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"  {name:20s}: Error — {e}")
        return []

def collect_all(regions_to_run=None):
    now       = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date      = now.strftime("%Y%m%d")
    targets   = regions_to_run or list(REGIONS.keys())

    print(f"\n[{timestamp}] Collecting {len(targets)} regions...")

    # Fetch weather (only for regions being collected)
    weather_cache = {}
    print("  Fetching weather data...")
    for name in targets:
        meta = REGION_META.get(name, {})
        coords = meta[1] if meta else (0, 0)
        weather_cache[name] = get_weather(coords[0], coords[1])
        time.sleep(0.3)

    calendar    = get_calendar_features(now)
    all_chargers = []

    for name in targets:
        bbox    = REGIONS[name]
        meta    = REGION_META.get(name, ("XX", (0,0), "Unknown"))
        country = meta[0]
        continent = meta[2]
        chargers = collect_region(name, bbox)
        weather  = weather_cache.get(name, {"temperature": 20, "precipitation": 0, "weathercode": 0})

        for c in chargers:
            c['region']            = name
            c['country']           = country
            c['continent']         = continent
            c['temperature']       = weather['temperature']
            c['precipitation']     = weather['precipitation']
            c['weathercode']       = weather['weathercode']
            c['is_bank_holiday']   = calendar['is_bank_holiday']
            c['is_school_holiday'] = calendar['is_school_holiday']
            c['is_summer']         = calendar['is_summer']
            c['is_december']       = calendar['is_december']

        all_chargers.extend(chargers)
        time.sleep(4)  # Be respectful to Overpass API

    snapshot = {"timestamp": timestamp, "chargers": all_chargers}
    filename = f"data_{date}.json"
    with open(filename, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    # Summary by continent
    from collections import Counter
    continents = Counter(c.get('continent','?') for c in all_chargers)
    print(f"\n  ✅ Total: {len(all_chargers):,} chargers saved to {filename}")
    for cont, count in sorted(continents.items()):
        print(f"     {cont:15s}: {count:,}")

    return len(all_chargers)

# ── TARGETED COLLECTION MODES ────────────────────────────────
AFRICA_REGIONS       = ["south_africa","kenya","nigeria","egypt","ethiopia","morocco","ghana","rwanda","tanzania"]
MIDDLE_EAST_REGIONS  = ["uae","saudi_arabia","israel"]
ASIA_REGIONS         = ["india_south","india_west","indonesia","thailand","vietnam"]
LATAM_REGIONS        = ["brazil_southeast","chile","colombia"]
EMERGING_MARKETS     = AFRICA_REGIONS + MIDDLE_EAST_REGIONS + ASIA_REGIONS + LATAM_REGIONS
EXISTING_REGIONS     = ["uk","us_east","us_west","us_texas","us_midwest","norway","netherlands","germany","sweden","france"]
ALL_REGIONS          = EXISTING_REGIONS + EMERGING_MARKETS

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    modes = {
        "all":           ALL_REGIONS,
        "existing":      EXISTING_REGIONS,
        "emerging":      EMERGING_MARKETS,
        "africa":        AFRICA_REGIONS,
        "middle_east":   MIDDLE_EAST_REGIONS,
        "asia":          ASIA_REGIONS,
        "latam":         LATAM_REGIONS,
    }

    target = modes.get(mode, ALL_REGIONS)

    print(f"ChargeSmart Collector — mode: {mode} ({len(target)} regions)")
    print(f"Regions: {', '.join(target)}")
    print("Collecting every 10 minutes. Press Ctrl+C to stop.\n")

    while True:
        collect_all(target)
        print("  Waiting 10 minutes...\n")
        time.sleep(600)
