from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
import requests
from geopy.distance import geodesic
import pgeocode

app = Flask(**name**, static_folder='static')

OCM_API_KEY = 'd29bb079-0234-40d8-b0af-a2bb55a7d399'
OCM_URL = 'https://api.openchargemap.io/v3/poi'

def load_chargers():
if os.path.exists('enriched_data.csv'):
df = pd.read_csv('enriched_data.csv')
chargers = df.groupby('id').agg({
'lat': 'first', 'lon': 'first', 'capacity': 'first',
'operator': 'first', 'location_type': 'first', 'country': 'first'
}).reset_index()
print(f"Loaded {len(chargers)} unique chargers")
return chargers
else:
print("No enriched_data.csv found, generating synthetic data…")
return generate_synthetic_chargers()

def generate_synthetic_chargers():
np.random.seed(42)
uk = pd.DataFrame({
'id': range(1, 101),
'lat': np.random.uniform(51.0, 53.0, 100),
'lon': np.random.uniform(-2.0, 0.5, 100),
'capacity': np.random.choice([1, 2, 4, 6], 100),
'operator': np.random.choice(['BP Pulse', 'Pod Point', 'Osprey', 'Unknown'], 100),
'location_type': np.random.choice(['motorway', 'supermarket', 'council', 'other'], 100),
'country': 'UK'
})
us = pd.DataFrame({
'id': range(101, 201),
'lat': np.random.uniform(34.0, 47.0, 100),
'lon': np.random.uniform(-122.5, -70.0, 100),
'capacity': np.random.choice([2, 4, 8, 12], 100),
'operator': np.random.choice(['Tesla', 'ChargePoint', 'EVgo', 'Blink', 'Electrify America'], 100),
'location_type': np.random.choice(['motorway', 'supermarket', 'tesla', 'other'], 100),
'country': 'US'
})
return pd.concat([uk, us], ignore_index=True)

def train_model(chargers):
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
np.random.seed(42)
n = len(chargers) * 20
hours = np.random.randint(0, 24, n)
days = np.random.randint(0, 7, n)
is_weekend = (days >= 5).astype(int)
capacities = np.random.choice(chargers['capacity'].values, n)
lats = np.random.choice(chargers['lat'].values, n)
lons = np.random.choice(chargers['lon'].values, n)
score = np.zeros(n)
score += np.where((hours >= 8) & (hours <= 9), 3, 0)
score += np.where((hours >= 17) & (hours <= 19), 3, 0)
score += np.where((hours >= 12) & (hours <= 13), 1, 0)
score += is_weekend * 2
score += np.where(capacities == 1, 2, 0)
score += np.random.randint(0, 4, n)
busy = (score >= 5).astype(int)
X = np.column_stack([hours, days, is_weekend, capacities, lats, lons])
X_train, X_test, y_train, y_test = train_test_split(X, busy, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Model trained! Accuracy: {model.score(X_test, y_test):.2%}")
with open('charger_model.pkl', 'wb') as f:
pickle.dump(model, f)
return model

def get_live_chargers(lat, lon, radius_miles, country):
"""Fetch live charger data from OpenChargeMap"""
try:
radius_km = radius_miles * 1.60934
country_code = 'GB' if country == 'UK' else 'US'
params = {
'key': OCM_API_KEY,
'latitude': lat,
'longitude': lon,
'distance': radius_km,
'distanceunit': 'KM',
'countrycode': country_code,
'maxresults': 30,
'compact': True,
'verbose': False,
}
response = requests.get(OCM_URL, params=params, timeout=8)
if response.status_code != 200:
return []
data = response.json()
chargers = []
for poi in data:
try:
addr = poi.get('AddressInfo', {})
c_lat = addr.get('Latitude')
c_lon = addr.get('Longitude')
if not c_lat or not c_lon:
continue
operator = 'Unknown'
op_info = poi.get('OperatorInfo')
if op_info and op_info.get('Title'):
operator = op_info['Title'][:30]
connections = poi.get('Connections', [])
capacity = max(len(connections), 1)
status_type = poi.get('StatusType')
live_status = None
if status_type:
status_id = status_type.get('ID', 0)
if status_id == 50:
live_status = 'free'
elif status_id == 210:
live_status = 'busy'
dist = geodesic((lat, lon), (c_lat, c_lon)).miles
chargers.append({
'id': poi.get('ID', 0),
'lat': float(c_lat),
'lon': float(c_lon),
'operator': operator,
'capacity': capacity,
'distance_miles': round(dist, 2),
'live_status': live_status,
'country': country,
})
except:
continue
return sorted(chargers, key=lambda x: x['distance_miles'])
except Exception as e:
print(f"OCM API error: {e}")
return []

# Load model and chargers

chargers = load_chargers()
if os.path.exists('charger_model.pkl'):
with open('charger_model.pkl', 'rb') as f:
model = pickle.load(f)
print("Model loaded from file")
else:
print("Training model…")
model = train_model(chargers)

nomi_uk = pgeocode.Nominatim('GB')
nomi_us = pgeocode.Nominatim('US')

def lookup_location(query):
query = query.strip().upper()
result = nomi_uk.query_postal_code(query)
if result is not None and not pd.isna(result.latitude):
return float(result.latitude), float(result.longitude), 'UK'
result = nomi_us.query_postal_code(query)
if result is not None and not pd.isna(result.latitude):
return float(result.latitude), float(result.longitude), 'US'
return None, None, None

@app.route('/')
def index():
return render_template('index.html')

@app.route('/predict')
def predict():
try:
hour = int(request.args.get('hour', 12))
day_of_week = int(request.args.get('day_of_week', 0))
capacity = int(request.args.get('capacity', 2))
lat = float(request.args.get('lat', 51.5))
lon = float(request.args.get('lon', -0.1))
features = np.array([[hour, day_of_week, 1 if day_of_week >= 5 else 0, capacity, lat, lon]])
prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0]
return jsonify({
'prediction': 'busy' if prediction == 1 else 'free',
'probability_busy': round(float(proba[1]) * 100, 1),
'probability_free': round(float(proba[0]) * 100, 1)
})
except Exception as e:
return jsonify({'error': str(e)}), 400

@app.route('/nearby')
def nearby():
try:
hour = int(request.args.get('hour', 12))
day_of_week = int(request.args.get('day_of_week', 0))
radius_miles = float(request.args.get('radius', 1.0))

```
    if request.args.get('lat') and request.args.get('lon'):
        user_lat = float(request.args.get('lat'))
        user_lon = float(request.args.get('lon'))
        country = request.args.get('country', 'UK')
    elif request.args.get('postcode'):
        query = request.args.get('postcode')
        user_lat, user_lon, country = lookup_location(query)
        if user_lat is None:
            return jsonify({'error': f'Could not find location: {query}. Please check and try again.'}), 400
    else:
        return jsonify({'error': 'Please provide a postcode, ZIP code, or coordinates'}), 400

    is_weekend = 1 if day_of_week >= 5 else 0
    results = []

    # Try live OCM data first
    live_chargers = get_live_chargers(user_lat, user_lon, radius_miles, country)

    if live_chargers:
        for c in live_chargers[:20]:
            features = np.array([[hour, day_of_week, is_weekend, c['capacity'], c['lat'], c['lon']]])
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            if c['live_status']:
                final_prediction = c['live_status']
                prob_free = 95.0 if c['live_status'] == 'free' else 5.0
                source = 'live'
            else:
                final_prediction = 'busy' if prediction == 1 else 'free'
                prob_free = round(float(proba[0]) * 100, 1)
                source = 'ai'

            results.append({
                'id': c['id'],
                'lat': c['lat'],
                'lon': c['lon'],
                'operator': c['operator'],
                'capacity': c['capacity'],
                'distance_miles': c['distance_miles'],
                'prediction': final_prediction,
                'probability_free': prob_free,
                'probability_busy': round(100 - prob_free, 1),
                'country': c['country'],
                'source': source
            })
    else:
        # Fall back to our database + AI
        local_chargers = chargers.copy()
        if country and 'country' in local_chargers.columns:
            local_chargers = local_chargers[local_chargers['country'] == country]

        nearby_list = []
        for _, c in local_chargers.iterrows():
            dist = geodesic((user_lat, user_lon), (c['lat'], c['lon'])).miles
            if dist <= radius_miles:
                nearby_list.append((dist, c))

        for dist, c in sorted(nearby_list)[:20]:
            features = np.array([[hour, day_of_week, is_weekend, c['capacity'], c['lat'], c['lon']]])
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            results.append({
                'id': int(c['id']),
                'lat': round(float(c['lat']), 6),
                'lon': round(float(c['lon']), 6),
                'operator': str(c['operator']),
                'capacity': int(c['capacity']),
                'distance_miles': round(dist, 2),
                'prediction': 'busy' if prediction == 1 else 'free',
                'probability_free': round(float(proba[0]) * 100, 1),
                'probability_busy': round(float(proba[1]) * 100, 1),
                'country': str(c.get('country', 'UK')),
                'source': 'ai'
            })

    return jsonify({'total': len(results), 'chargers': results})

except Exception as e:
    return jsonify({'error': str(e)}), 400
```

@app.route('/static/manifest.json')
def manifest():
return app.send_static_file('manifest.json')

@app.route('/static/service-worker.js')
def service_worker():
response = app.send_static_file('service-worker.js')
response.headers['Service-Worker-Allowed'] = '/'
return response

@app.route('/sitemap.xml')
def sitemap():
return app.send_static_file('sitemap.xml')

@app.route('/robots.txt')
def robots():
return app.send_static_file('robots.txt')

if **name** == '**main**':
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=False)
