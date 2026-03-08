"""
ChargeSmart — Data Collector
==============================
Collects EV charger data from TWO sources:
  1. OpenChargeMap (OCM) — primary, 50,000+ verified chargers globally
  2. OpenStreetMap via Overpass API — secondary, fills gaps OCM misses

Both are merged and deduplicated before saving.
OCM gives richer data (operator, connection type, live status).
OSM gives broader coverage in emerging markets.
"""

import requests
import json
import math
from datetime import datetime
from collections import Counter
import time

# ── API CONFIG ────────────────────────────────────────────────
OCM_API_KEY  = 'd29bb079-0234-40d8-b0af-a2bb55a7d399'
OCM_URL      = 'https://api.openchargemap.io/v3/poi'
OVERPASS_URL = 'http://overpass-api.de/api/interpreter'

# ── REGION DEFINITIONS ────────────────────────────────────────
REGIONS = {
    'uk':              (51.0,  -2.0,  53.0,   0.5),
    'us_east':         (38.0, -77.0,  42.5, -70.0),
    'us_west':         (34.0,-122.5,  47.5,-117.0),
    'us_texas':        (25.5,-100.0,  36.5, -93.5),
    'us_midwest':      (40.0, -90.0,  43.5, -82.5),
    'norway':          (57.0,   4.0,  71.0,  31.0),
    'netherlands':     (50.7,   3.3,  53.6,   7.2),
    'germany':         (47.3,   5.9,  55.0,  15.0),
    'sweden':          (55.0,  10.9,  69.0,  24.2),
    'france':          (42.3,  -4.8,  51.1,   8.2),
    'south_africa':    (-34.8,  16.5, -22.1,  33.0),
    'kenya':           ( -4.7,  33.9,   4.6,  41.9),
    'nigeria':         (  4.3,   2.7,  13.9,  14.7),
    'egypt':           ( 22.0,  24.7,  31.7,  37.1),
    'ethiopia':        (  3.4,  33.0,  15.0,  48.0),
    'morocco':         ( 27.6, -13.2,  35.9,  -1.1),
    'ghana':           (  4.7,  -3.3,  11.2,   1.2),
    'rwanda':          ( -2.9,  28.8,  -1.0,  30.9),
    'tanzania':        (-11.7,  29.3,  -0.9,  40.5),
    'uae':             ( 22.6,  51.6,  26.1,  56.4),
    'saudi_arabia':    ( 16.4,  36.5,  32.2,  55.7),
    'israel':          ( 29.5,  34.2,  33.3,  35.9),
    'india_south':     (  8.0,  76.0,  15.0,  80.5),
    'india_west':      ( 18.9,  72.8,  23.1,  73.2),
    'indonesia':       ( -8.8, 106.6,  -6.1, 112.0),
    'thailand':        ( 13.6, 100.3,  14.1, 100.9),
    'vietnam':         ( 10.6, 106.5,  21.1, 107.2),
    'brazil_southeast':(-23.8, -46.9, -19.8, -43.1),
    'chile':           (-36.0, -72.5, -29.9, -69.5),
    'colombia':        (  3.8, -77.2,   7.2, -72.5),
}

# country_code, weather_coords, continent, ocm_country_code
REGION_META = {
    'uk':               ('UK',  (51.5,  -0.1), 'Europe',       'GB'),
    'us_east':          ('US',  (40.7, -74.0), 'Americas',     'US'),
    'us_west':          ('US',  (37.7,-122.4), 'Americas',     'US'),
    'us_texas':         ('US',  (30.3, -97.7), 'Americas',     'US'),
    'us_midwest':       ('US',  (41.8, -87.6), 'Americas',     'US'),
    'norway':           ('EU',  (59.9,  10.7), 'Europe',       'NO'),
    'netherlands':      ('EU',  (52.4,   4.9), 'Europe',       'NL'),
    'germany':          ('EU',  (52.5,  13.4), 'Europe',       'DE'),
    'sweden':           ('EU',  (59.3,  18.1), 'Europe',       'SE'),
    'france':           ('EU',  (48.9,   2.3), 'Europe',       'FR'),
    'south_africa':     ('ZA',  (-26.2,  28.0), 'Africa',      'ZA'),
    'kenya':            ('KE',  ( -1.3,  36.8), 'Africa',      'KE'),
    'nigeria':          ('NG',  (  6.5,   3.4), 'Africa',      'NG'),
    'egypt':            ('EG',  ( 30.0,  31.2), 'Africa',      'EG'),
    'ethiopia':         ('ET',  (  9.0,  38.7), 'Africa',      'ET'),
    'morocco':          ('MA',  ( 33.9,  -6.9), 'Africa',      'MA'),
    'ghana':            ('GH',  (  5.6,  -0.2), 'Africa',      'GH'),
    'rwanda':           ('RW',  ( -1.9,  30.1), 'Africa',      'RW'),
    'tanzania':         ('TZ',  ( -6.8,  39.3), 'Africa',      'TZ'),
    'uae':              ('AE',  ( 25.2,  55.3), 'Middle East', 'AE'),
    'saudi_arabia':     ('SA',  ( 24.7,  46.7), 'Middle East', 'SA'),
    'israel':           ('IL',  ( 31.8,  35.2), 'Middle East', 'IL'),
    'india_south':      ('IN',  ( 12.9,  77.6), 'Asia',        'IN'),
    'india_west':       ('IN',  ( 19.1,  72.9), 'Asia',        'IN'),
    'indonesia':        ('ID',  ( -6.2, 106.8), 'Asia',        'ID'),
    'thailand':         ('TH',  ( 13.8, 100.5), 'Asia',        'TH'),
    'vietnam':          ('VN',  ( 10.8, 106.7), 'Asia',        'VN'),
    'brazil_southeast': ('BR',  (-23.5, -46.6), 'Americas',    'BR'),
    'chile':            ('CL',  (-33.5, -70.6), 'Americas',    'CL'),
    'colombia':         ('CO',  (  4.7, -74.1), 'Americas',    'CO'),
}


# ── WEATHER ───────────────────────────────────────────────────
def get_weather(lat, lon):
    try:
        url = (
            f'https://api.open-meteo.com/v1/forecast'
            f'?latitude={lat}&longitude={lon}'
            f'&current=temperature_2m,precipitation,weathercode'
            f'&timezone=auto'
        )
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            current = r.json().get('current', {})
            return {
                'temperature':   current.get('temperature_2m', 20),
                'precipitation': current.get('precipitation', 0),
                'weathercode':   current.get('weathercode', 0),
            }
    except Exception:
        pass
    return {'temperature': 20, 'precipitation': 0, 'weathercode': 0}


# ── CALENDAR ──────────────────────────────────────────────────
def get_calendar_features(dt):
    uk_bank_holidays = {
        '2025-01-01','2025-04-18','2025-04-21','2025-05-05',
        '2025-05-26','2025-08-25','2025-12-25','2025-12-26',
        '2026-01-01','2026-04-03','2026-04-06','2026-05-04',
        '2026-05-25','2026-08-31','2026-12-25','2026-12-28',
    }
    date_str = dt.strftime('%Y-%m-%d')
    month = dt.month
    day   = dt.day
    is_bank_holiday   = 1 if date_str in uk_bank_holidays else 0
    is_summer         = 1 if month in [6, 7, 8] else 0
    is_december       = 1 if month == 12 else 0
    is_school_holiday = 1 if (
        month == 8 or (month == 7 and day >= 20) or
        (month == 12 and day >= 20) or
        (month == 4 and 1 <= day <= 14) or is_bank_holiday
    ) else 0
    return {
        'is_bank_holiday':   is_bank_holiday,
        'is_school_holiday': is_school_holiday,
        'is_summer':         is_summer,
        'is_december':       is_december,
    }


# ── OCM COLLECTOR ─────────────────────────────────────────────
def collect_ocm(name, bbox, ocm_country_code, max_results=500):
    south, west, north, east = bbox
    lat_centre = (south + north) / 2
    lon_centre = (west  + east)  / 2
    lat_diff   = abs(north - south) / 2
    lon_diff   = abs(east  - west)  / 2
    radius_km  = math.sqrt(lat_diff**2 + lon_diff**2) * 111

    params = {
        'key':          OCM_API_KEY,
        'latitude':     lat_centre,
        'longitude':    lon_centre,
        'distance':     min(radius_km, 500),
        'distanceunit': 'KM',
        'maxresults':   max_results,
        'compact':      True,
        'verbose':      False,
        'countrycode':  ocm_country_code,
    }

    try:
        response = requests.get(OCM_URL, params=params, timeout=30)
        if response.status_code == 429:
            print(f'  {name:20s}: OCM rate limited, skipping')
            return []
        if response.status_code != 200:
            print(f'  {name:20s}: OCM HTTP {response.status_code}')
            return []

        data = response.json()
        if not isinstance(data, list):
            return []

        chargers = []
        for poi in data:
            try:
                addr  = poi.get('AddressInfo') or {}
                c_lat = addr.get('Latitude')
                c_lon = addr.get('Longitude')
                if not c_lat or not c_lon:
                    continue

                # Filter to bounding box
                if not (south <= float(c_lat) <= north and west <= float(c_lon) <= east):
                    continue

                op_info  = poi.get('OperatorInfo') or {}
                operator = (op_info.get('Title') or addr.get('Title') or 'Unknown')[:40]

                connections = poi.get('Connections') or []
                capacity    = max(len(connections), 1)

                max_kw = 0
                for conn in connections:
                    kw = conn.get('PowerKW') or 0
                    if kw > max_kw:
                        max_kw = kw

                status_type = poi.get('StatusType') or {}
                status_id   = status_type.get('ID', 0)
                live_status = 'free' if status_id == 50 else 'busy' if status_id == 210 else None

                usage_type = poi.get('UsageType') or {}
                is_free    = 1 if (not usage_type.get('IsMembershipRequired') and
                                   not usage_type.get('IsPayAtLocation')) else 0

                chargers.append({
                    'id':          f"ocm_{poi.get('ID', 0)}",
                    'source':      'ocm',
                    'lat':         float(c_lat),
                    'lon':         float(c_lon),
                    'operator':    operator,
                    'capacity':    capacity,
                    'max_kw':      max_kw,
                    'live_status': live_status,
                    'is_free':     is_free,
                    'tags':        {},
                })
            except Exception:
                continue

        return chargers

    except Exception as e:
        print(f'  {name:20s}: OCM error — {e}')
        return []


# ── OSM COLLECTOR ─────────────────────────────────────────────
def collect_osm(name, bbox):
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:90];
    node["amenity"="charging_station"]({south},{west},{north},{east});
    out body;
    """
    try:
        response = requests.post(OVERPASS_URL, data={'data': query}, timeout=120)
        if response.status_code == 429:
            print(f'  {name:20s}: OSM rate limited')
            return []
        if response.status_code != 200:
            return []

        elements = response.json().get('elements', [])
        chargers = []

        for el in elements:
            tags     = el.get('tags', {})
            operator = (tags.get('operator') or tags.get('name') or 'Unknown')[:40]
            try:
                capacity = int(tags.get('capacity', 1))
            except Exception:
                capacity = 1

            chargers.append({
                'id':          f"osm_{el.get('id', 0)}",
                'source':      'osm',
                'lat':         float(el.get('lat', 0)),
                'lon':         float(el.get('lon', 0)),
                'operator':    operator,
                'capacity':    capacity,
                'max_kw':      0,
                'live_status': None,
                'is_free':     0,
                'tags':        tags,
            })

        return chargers

    except Exception as e:
        print(f'  {name:20s}: OSM error — {e}')
        return []


# ── DEDUPLICATION ─────────────────────────────────────────────
def deduplicate(chargers, threshold_m=50):
    if not chargers:
        return []
    # OCM wins over OSM when same location
    chargers = sorted(chargers, key=lambda x: 0 if x['source'] == 'ocm' else 1)
    kept  = []
    thresh = threshold_m / 111000  # metres to degrees
    for c in chargers:
        dup = False
        for k in kept:
            if abs(c['lat'] - k['lat']) < thresh and abs(c['lon'] - k['lon']) < thresh:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# ── REGION COLLECTOR ──────────────────────────────────────────
def collect_region(name, bbox, ocm_country_code):
    ocm_chargers = collect_ocm(name, bbox, ocm_country_code)
    time.sleep(1)
    osm_chargers = collect_osm(name, bbox)
    combined     = deduplicate(ocm_chargers + osm_chargers)
    ocm_n = sum(1 for c in combined if c['source'] == 'ocm')
    osm_n = sum(1 for c in combined if c['source'] == 'osm')
    print(f'  {name:20s}: {len(combined):4d} total  (OCM:{ocm_n} + OSM:{osm_n})')
    return combined


# ── MAIN COLLECT ──────────────────────────────────────────────
def collect_all(regions_to_run=None):
    now       = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    date      = now.strftime('%Y%m%d')
    targets   = regions_to_run or list(REGIONS.keys())

    print(f'\n[{timestamp}] Collecting {len(targets)} regions...')
    print(f'  Sources: OpenChargeMap (primary) + OpenStreetMap (secondary)\n')

    weather_cache = {}
    print('  Fetching weather data...')
    for name in targets:
        meta   = REGION_META.get(name, ('XX', (0, 0), 'Unknown', 'XX'))
        coords = meta[1]
        weather_cache[name] = get_weather(coords[0], coords[1])
        time.sleep(0.3)

    calendar     = get_calendar_features(now)
    all_chargers = []

    for name in targets:
        bbox             = REGIONS[name]
        meta             = REGION_META.get(name, ('XX', (0, 0), 'Unknown', 'XX'))
        country          = meta[0]
        continent        = meta[2]
        ocm_country_code = meta[3] if len(meta) > 3 else 'XX'
        weather          = weather_cache.get(name, {'temperature': 20, 'precipitation': 0, 'weathercode': 0})

        chargers = collect_region(name, bbox, ocm_country_code)

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
        time.sleep(3)

    snapshot = {'timestamp': timestamp, 'chargers': all_chargers}
    filename = f'data_{date}.json'
    with open(filename, 'a') as f:
        f.write(json.dumps(snapshot) + '\n')

    continents = Counter(c.get('continent', '?') for c in all_chargers)
    sources    = Counter(c.get('source', '?')    for c in all_chargers)

    print(f'\n  Total: {len(all_chargers):,} chargers saved to {filename}')
    for cont, count in sorted(continents.items()):
        print(f'     {cont:15s}: {count:,}')
    print(f'  Sources: OCM={sources["ocm"]:,}  OSM={sources["osm"]:,}')

    return len(all_chargers)


# ── MODES ────────────────────────────────────────────────────
AFRICA_REGIONS       = ['south_africa','kenya','nigeria','egypt','ethiopia','morocco','ghana','rwanda','tanzania']
MIDDLE_EAST_REGIONS  = ['uae','saudi_arabia','israel']
ASIA_REGIONS         = ['india_south','india_west','indonesia','thailand','vietnam']
LATAM_REGIONS        = ['brazil_southeast','chile','colombia']
EMERGING_MARKETS     = AFRICA_REGIONS + MIDDLE_EAST_REGIONS + ASIA_REGIONS + LATAM_REGIONS
EXISTING_REGIONS     = ['uk','us_east','us_west','us_texas','us_midwest','norway','netherlands','germany','sweden','france']
ALL_REGIONS          = EXISTING_REGIONS + EMERGING_MARKETS


if __name__ == '__main__':
    import sys
    mode   = sys.argv[1] if len(sys.argv) > 1 else 'emerging'
    modes  = {
        'all':         ALL_REGIONS,
        'existing':    EXISTING_REGIONS,
        'emerging':    EMERGING_MARKETS,
        'africa':      AFRICA_REGIONS,
        'middle_east': MIDDLE_EAST_REGIONS,
        'asia':        ASIA_REGIONS,
        'latam':       LATAM_REGIONS,
    }
    target = modes.get(mode, EMERGING_MARKETS)
    print(f'ChargeSmart Dual-Source Collector')
    print(f'Mode: {mode} | Regions: {len(target)} | Sources: OCM + OSM')
    collect_all(target)
