import json
import csv
from datetime import datetime

def get_time_features(timestamp):
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return {
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "is_weekend": dt.weekday() >= 5
    }

def get_location_type(tags):
    name = tags.get("name", "").lower()
    operator = tags.get("operator", "").lower()
    if "motorway" in name or "services" in name:
        return "motorway"
    elif "supermarket" in name or "tesco" in operator or "sainsbury" in operator:
        return "supermarket"
    elif "council" in operator or "city" in operator:
        return "council"
    else:
        return "other"

def process_data():
    input_file = "data_20260301.json"
    output_file = "enriched_data.csv"
    fieldnames = ["timestamp", "hour", "day_of_week",
                  "is_weekend", "location_type", "capacity",
                  "lat", "lon", "operator"]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with open(input_file, "r") as f:
            for line in f:
                snapshot = json.loads(line)
                time_features = get_time_features(snapshot["timestamp"])

                for charger in snapshot["chargers"]:
                    tags = charger.get("tags", {})
                    row = {
                        "timestamp": snapshot["timestamp"],
                        "hour": time_features["hour"],
                        "day_of_week": time_features["day_of_week"],
                        "is_weekend": time_features["is_weekend"],
                        "location_type": get_location_type(tags),
                        "capacity": tags.get("capacity", 1),
                        "lat": charger["lat"],
                        "lon": charger["lon"],
                        "operator": tags.get("operator", "unknown")
                    }
                    writer.writerow(row)

    print("Enriched data saved to enriched_data.csv!")

process_data()
print("Done!")