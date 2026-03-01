import json

with open("data_20260301.json", "r") as f:
    first_line = f.readline()

snapshot = json.loads(first_line)
chargers = snapshot["chargers"]

print(f"Total chargers: {len(chargers)}")
print(f"Timestamp: {snapshot['timestamp']}")
print("")
print("Sample of 5 chargers:")

for c in chargers[:5]:
    print(f"  - {c['id']} | {c['lat']} | {c['lon']} | {c['tags']}")