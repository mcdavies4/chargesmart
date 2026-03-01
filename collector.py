import requests
import schedule
import time
import json
from datetime import datetime

API_URL = "https://overpass-api.de/api/interpreter"

QUERY = """
[out:json];
node["amenity"="charging_station"]
(51.0,-2.0,53.0,0.5);
out body;
"""

def collect_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Collecting data at {timestamp}...")
    
    try:
        response = requests.post(API_URL, data=QUERY)
        data = response.json()
        
        filename = f"data_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, "a") as f:
            entry = {"timestamp": timestamp, "chargers": data["elements"]}
            f.write(json.dumps(entry) + "\n")
        
        print(f"Saved {len(data['elements'])} chargers successfully!")
    
    except Exception as e:
        print(f"Something went wrong: {e}")
        print(f"Response text: {response.text[:200]}")

collect_data()
schedule.every(10).minutes.do(collect_data)

print("Data collector is running... Press CTRL+C to stop")
while True:
    schedule.run_pending()
    time.sleep(1)