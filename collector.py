import requests
import json
from datetime import datetime
import time

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

REGIONS = {
    "uk": (51.0, -2.0, 53.0, 0.5),
    "us_east": (38.0, -77.0, 42.5, -70.0),
    "us_west": (34.0, -122.5, 47.5, -117.0),
    "us_texas": (25.5, -100.0, 36.5, -93.5),
    "us_midwest": (40.0, -90.0, 43.5, -82.5),
    "norway": (57.0, 4.0, 71.0, 31.0),
    "netherlands": (50.7, 3.3, 53.6, 7.2),
    "germany": (47.3, 5.9, 55.0, 15.0),
    "sweden": (55.0, 10.9, 69.0, 24.2),
    "france": (42.3, -4.8, 51.1, 8.2),
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
    else:
        return 'UK'

def collect_all():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.now().strftime("%Y%m%d")
    print(f"\n[{timestamp}] Collecting all regions...")

    all_chargers = []
    for name, bbox in REGIONS.items():
        chargers = collect_region(name, bbox)
        for c in chargers:
            c['region'] = name
            c['country'] = get_country(name)
        all_chargers.extend(chargers)
        time.sleep(3)

    snapshot = {
        "timestamp": timestamp,
        "chargers": all_chargers
    }

    filename = f"data_{date}.json"
    with open(filename, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    print(f"  Total: {len(all_chargers)} chargers saved to {filename}")

if __name__ == "__main__":
    print("ChargeSmart Collector - UK + US + Europe")
    print("Collecting every 10 minutes. Press Ctrl+C to stop.\n")
    while True:
        collect_all()
        print("  Waiting 10 minutes...")
        time.sleep(600)
