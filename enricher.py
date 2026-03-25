import json
import csv
import glob
import os
from datetime import datetime

# Use Railway Volume path if set, otherwise current directory
DATA_DIR = os.environ.get('DATA_DIR', '.')
os.makedirs(DATA_DIR, exist_ok=True)

UK_OPERATORS = ['bp pulse','pod point','osprey','gridserve','zap-map','char.gy',
                'ubitricity','geo','engenie','instavolt','evolt','mod','tesla']
US_OPERATORS = ['tesla','supercharger','chargepoint','evgo','blink',
                'electrify america','volta','semacharge','greenlots','chargehub','clipper creek']
EU_OPERATORS = ['ionity','allego','fastned','recharge','enbw','tesla',
                'supercharger','charge4europe','newmotion','vattenfall']
AFRICA_OPERATORS = ['gridcars','charge.africa','axgrid','ev-zone','powerx',
                    'ampersand','spiro','roam','basigo','zero carbon charge',
                    'african clean energy','kabisa','byd','nawiri',
                    'national oil','kenol','total energies africa']
MIDDLE_EAST_OPERATORS = ['charge & go','eviq','adnoc','charge master',
                         'tesla','pod point','dubai electricity']
ASIA_OPERATORS = ['tata power','ather','ola electric','charge+','evolt',
                  'pea volta','ea anywhere','vinfast','evn','charge spot',
                  'pertamina','star charge','byd']
LATAM_OPERATORS = ['eletroposto','zletric','voltbras','enel x','copec',
                   'blink','tesla','terpel','charge here']
ALL_OPERATORS = (UK_OPERATORS + US_OPERATORS + EU_OPERATORS + AFRICA_OPERATORS +
                 MIDDLE_EAST_OPERATORS + ASIA_OPERATORS + LATAM_OPERATORS)

UK_BANK_HOLIDAYS = {
    "2025-01-01","2025-04-18","2025-04-21","2025-05-05","2025-05-26",
    "2025-08-25","2025-12-25","2025-12-26",
    "2026-01-01","2026-04-03","2026-04-06","2026-05-04",
    "2026-05-25","2026-08-31","2026-12-25","2026-12-28",
}

def classify_location(tags, operator, country):
    """Works for both OSM tags and OCM operator strings."""
    name     = tags.get('name', '').lower() if tags else ''
    op       = (tags.get('operator', '') if tags else operator or '').lower()
    combined = name + ' ' + op
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

def get_operator(tags, ocm_operator, country):
    """
    For OCM records, operator is already clean.
    For OSM records, extract from tags.
    """
    if ocm_operator and ocm_operator != 'Unknown':
        return ocm_operator[:30]
    tags = tags or {}
    operator = tags.get('operator', '').strip()
    name     = tags.get('name', '').strip()
    combined = (operator + ' ' + name).lower()
    for op in ALL_OPERATORS:
        if op in combined:
            return op.title()
    return (operator or name or 'unknown')[:30]

def get_calendar(dt):
    date_str = dt.strftime("%Y-%m-%d")
    month, day = dt.month, dt.day
    is_bank_holiday   = 1 if date_str in UK_BANK_HOLIDAYS else 0
    is_school_holiday = 1 if (
        month == 8 or (month == 7 and day >= 20) or
        (month == 12 and day >= 20) or
        (month == 4 and 1 <= day <= 14) or is_bank_holiday
    ) else 0
    is_summer   = 1 if month in [6, 7, 8] else 0
    is_december = 1 if month == 12 else 0
    return is_bank_holiday, is_school_holiday, is_summer, is_december

def enrich():
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_*.json')))
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
                    hour        = dt.hour
                    day_of_week = dt.weekday()
                    is_weekend  = 1 if day_of_week >= 5 else 0
                    is_bank_holiday, is_school_holiday, is_summer, is_december = get_calendar(dt)
                except Exception:
                    hour = day_of_week = is_weekend = 12
                    is_bank_holiday = is_school_holiday = is_summer = is_december = 0

                for c in snapshot.get('chargers', []):
                    tags         = c.get('tags') or {}
                    country      = c.get('country', 'UK')
                    source       = c.get('source', 'osm')   # 'ocm' or 'osm'
                    ocm_operator = c.get('operator', '')     # pre-parsed by OCM collector
                    max_kw       = c.get('max_kw', 0)        # OCM power rating
                    is_free      = c.get('is_free', 0)       # OCM free usage flag

                    # Capacity: OCM provides it directly, OSM via tags
                    if source == 'ocm':
                        capacity = min(int(c.get('capacity', 1)), 20)
                    else:
                        try:
                            capacity = min(int(tags.get('capacity', 1)), 20)
                        except Exception:
                            capacity = 1

                    rows.append({
                        'id':               c.get('id'),
                        'source':           source,
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
                        'location_type':    classify_location(tags, ocm_operator, country),
                        'capacity':         capacity,
                        'max_kw':           max_kw,
                        'is_free':          is_free,
                        'lat':              c.get('lat', 0),
                        'lon':              c.get('lon', 0),
                        'operator':         get_operator(tags, ocm_operator, country),
                        'country':          country,
                        'continent':        c.get('continent', 'Unknown'),
                    })

                pos = content.find('{', pos)
                if pos == -1:
                    break
            except Exception:
                break

        print(f"  {snapshot_count} snapshots, {len(rows):,} rows so far")

    if not rows:
        print("No data to enrich!")
        return

    out_path = os.path.join(DATA_DIR, 'enriched_data.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    sources   = Counter(r['source']    for r in rows)
    countries = Counter(r['country']   for r in rows)
    print(f"\nDone! {len(rows):,} total rows")
    print(f"  OCM: {sources['ocm']:,}  OSM: {sources['osm']:,}")
    print(f"  Top countries: {dict(countries.most_common(5))}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    enrich()
