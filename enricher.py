import json
import csv
import glob
from datetime import datetime

UK_OPERATORS = ['bp pulse','pod point','osprey','gridserve','zap-map','char.gy',
                'ubitricity','geo','engenie','instavolt','evolt','mod','tesla']
US_OPERATORS = ['tesla','supercharger','chargepoint','evgo','blink',
                'electrify america','volta','semacharge','greenlots','chargehub','clipper creek']
EU_OPERATORS = ['ionity','allego','fastned','recharge','enbw','tesla',
                'supercharger','charge4europe','newmotion','vattenfall']

# Africa operators
AFRICA_OPERATORS = ['gridcars','charge.africa','axgrid','ev-zone','powerx',
                    'ampersand','spiro','roam','BasiGo','zero carbon charge',
                    'african clean energy','kabisa','byd','nawiri',
                    'national oil','kenol','total energies africa']

# Middle East operators
MIDDLE_EAST_OPERATORS = ['charge & go','eviq','adnoc','charge master',
                         'tesla','pod point','dubai electricity']

# Asia operators  
ASIA_OPERATORS = ['tata power','ather','ola electric','charge+','evolt',
                  'pea volta','ea anywhere','vinfast','evn','charge spot',
                  'pertamina','star charge','byd']

# Latin America operators
LATAM_OPERATORS = ['eletroposto','zletric','voltbras','enel x','copec',
                   'blink','tesla','terpel','charge here']

ALL_EMERGING_OPERATORS = (AFRICA_OPERATORS + MIDDLE_EAST_OPERATORS + 
                           ASIA_OPERATORS + LATAM_OPERATORS)

UK_BANK_HOLIDAYS = {
    "2025-01-01","2025-04-18","2025-04-21","2025-05-05","2025-05-26",
    "2025-08-25","2025-12-25","2025-12-26",
    "2026-01-01","2026-04-03","2026-04-06","2026-05-04",
    "2026-05-25","2026-08-31","2026-12-25","2026-12-28",
}

def classify_location(tags, country):
    name = tags.get('name', '').lower()
    operator = tags.get('operator', '').lower()
    combined = name + ' ' + operator
    if any(k in combined for k in ['motorway','services','highway','freeway','autobahn','autoroute']):
        return 'motorway'
    if any(k in combined for k in ['tesco','sainsbury','asda','morrisons','walmart','target',
                                    'costco','supermarket','carrefour','lidl','aldi']):
        return 'supermarket'
    if any(k in combined for k in ['council','city','borough','county','municipal','gemeente']):
        return 'council'
    if any(k in combined for k in ['tesla','supercharger']):
        return 'tesla'
    return 'other'

def get_operator(tags, country):
    operator = tags.get('operator', '').strip()
    name = tags.get('name', '').strip()
    ops = US_OPERATORS if country == 'US' else EU_OPERATORS if country == 'EU' else UK_OPERATORS
    combined = (operator + ' ' + name).lower()
    for op in ops:
        if op in combined:
            return op.title()
    return (operator or name or 'unknown')[:30]

def get_calendar(dt):
    date_str = dt.strftime("%Y-%m-%d")
    month, day = dt.month, dt.day
    is_bank_holiday = 1 if date_str in UK_BANK_HOLIDAYS else 0
    is_school_holiday = 1 if (
        month == 8 or (month == 7 and day >= 20) or
        (month == 12 and day >= 20) or
        (month == 4 and 1 <= day <= 14) or
        is_bank_holiday
    ) else 0
    is_summer   = 1 if month in [6, 7, 8] else 0
    is_december = 1 if month == 12 else 0
    return is_bank_holiday, is_school_holiday, is_summer, is_december

def enrich():
    files = sorted(glob.glob('data_*.json'))
    if not files:
        print("No data files found!")
        return

    rows = []
    for filepath in files:
        print(f"Processing {filepath}...")
        content = open(filepath, 'r', encoding='utf-8', errors='ignore').read()
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
                    is_bank_holiday, is_school_holiday, is_summer, is_december = get_calendar(dt)
                except Exception:
                    hour, day_of_week, is_weekend = 12, 0, 0
                    is_bank_holiday = is_school_holiday = is_summer = is_december = 0

                for c in snapshot.get('chargers', []):
                    tags = c.get('tags', {})
                    country = c.get('country', 'UK')
                    try:
                        capacity = min(int(tags.get('capacity', 1)), 20)
                    except Exception:
                        capacity = 1

                    rows.append({
                        'id':               c.get('id'),
                        'timestamp':        timestamp,
                        'hour':             hour,
                        'day_of_week':      day_of_week,
                        'is_weekend':       is_weekend,
                        'is_bank_holiday':  is_bank_holiday,
                        'is_school_holiday':is_school_holiday,
                        'is_summer':        is_summer,
                        'is_december':      is_december,
                        'temperature':      c.get('temperature', 15),
                        'precipitation':    c.get('precipitation', 0),
                        'location_type':    classify_location(tags, country),
                        'capacity':         capacity,
                        'lat':              c.get('lat', 0),
                        'lon':              c.get('lon', 0),
                        'operator':         get_operator(tags, country),
                        'country':          country,
                    })

                pos = content.find('{', pos)
                if pos == -1:
                    break
            except Exception:
                break

        print(f"  {snapshot_count} snapshots processed")

    if not rows:
        print("No data to enrich!")
        return

    with open('enriched_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    uk = sum(1 for r in rows if r['country'] == 'UK')
    us = sum(1 for r in rows if r['country'] == 'US')
    eu = sum(1 for r in rows if r['country'] == 'EU')
    print(f"\nDone! {len(rows):,} total rows")
    print(f"  UK: {uk:,}  US: {us:,}  EU: {eu:,}")
    print(f"Saved to enriched_data.csv")

if __name__ == "__main__":
    enrich()
