import json
import csv
import glob
from datetime import datetime

UK_OPERATORS = ['bp pulse', 'pod point', 'osprey', 'gridserve', 'zap-map', 'char.gy',
                'ubitricity', 'geo', 'engenie', 'instavolt', 'evolt', 'mod', 'tesla']

US_OPERATORS = ['tesla', 'supercharger', 'chargepoint', 'evgo', 'blink', 
                'electrify america', 'volta', 'semacharge', 'greenlots', 
                'chargehub', 'francis', 'clipper creek']

def classify_location(tags, country):
    name = tags.get('name', '').lower()
    operator = tags.get('operator', '').lower()
    combined = name + ' ' + operator

    if any(k in combined for k in ['motorway', 'services', 'highway', 'freeway', 'rest stop', 'truck stop']):
        return 'motorway'
    if any(k in combined for k in ['tesco', 'sainsbury', 'asda', 'morrisons', 'walmart', 'target', 'costco', 'kroger', 'supermarket', 'grocery']):
        return 'supermarket'
    if any(k in combined for k in ['council', 'city', 'borough', 'county', 'municipal', 'government']):
        return 'council'
    if any(k in combined for k in ['tesla', 'supercharger']):
        return 'tesla'
    return 'other'

def get_operator(tags, country):
    operator = tags.get('operator', '').strip()
    name = tags.get('name', '').strip()
    
    operators = US_OPERATORS if country == 'US' else UK_OPERATORS
    combined = (operator + ' ' + name).lower()
    
    for op in operators:
        if op in combined:
            return op.title()
    
    if operator:
        return operator[:30]
    if name:
        return name[:30]
    return 'unknown'

def enrich():
    files = glob.glob('data_*.json')
    if not files:
        print("No data files found!")
        return

    rows = []
    for filepath in files:
        print(f"Processing {filepath}...")
        content = open(filepath, 'r').read()
        
        decoder = json.JSONDecoder()
        pos = 0
        snapshot_count = 0
        
        while pos < len(content):
            try:
                snapshot, pos = decoder.raw_decode(content, pos)
                snapshot_count += 1
                
                timestamp = snapshot.get('timestamp', '')
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    hour = dt.hour
                    day_of_week = dt.weekday()
                    is_weekend = 1 if day_of_week >= 5 else 0
                except:
                    hour, day_of_week, is_weekend = 12, 0, 0

                for c in snapshot.get('chargers', []):
                    tags = c.get('tags', {})
                    country = c.get('country', 'UK')
                    
                    try:
                        capacity = int(tags.get('capacity', 1))
                    except:
                        capacity = 1

                    rows.append({
                        'id': c.get('id'),
                        'timestamp': timestamp,
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'is_weekend': is_weekend,
                        'location_type': classify_location(tags, country),
                        'capacity': min(capacity, 20),
                        'lat': c.get('lat', 0),
                        'lon': c.get('lon', 0),
                        'operator': get_operator(tags, country),
                        'country': country,
                    })

                pos = content.find('{', pos)
                if pos == -1:
                    break
            except:
                break
        
        print(f"  {snapshot_count} snapshots processed")

    if not rows:
        print("No data to enrich!")
        return

    with open('enriched_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    uk_count = sum(1 for r in rows if r['country'] == 'UK')
    us_count = sum(1 for r in rows if r['country'] == 'US')
    print(f"\nDone! {len(rows)} total rows")
    print(f"  UK: {uk_count} rows")
    print(f"  US: {us_count} rows")
    print(f"Saved to enriched_data.csv")

if __name__ == "__main__":
    enrich()
