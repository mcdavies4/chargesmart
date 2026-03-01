from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
from geopy.distance import geodesic
import pgeocode

app = Flask(__name__, static_folder='static')

# Load charger data
def load_chargers():
    if os.path.exists('enriched_data.csv'):
        df = pd.read_csv('enriched_data.csv')
        chargers = df.groupby('id').agg({
            'lat': 'first',
            'lon': 'first',
            'capacity': 'first',
            'operator': 'first',
            'location_type': 'first',
            'country': 'first'
        }).reset_index()
        print(f"Loaded {len(chargers)} unique chargers")
        uk = len(chargers[chargers['country'] == 'UK']) if 'country' in chargers.columns else len(chargers)
        us = len(chargers[chargers['country'] == 'US']) if 'country' in chargers.columns else 0
        print(f"  UK: {uk}, US: {us}")
        return chargers
    else:
        print("No enriched_data.csv found, generating synthetic data...")
        return generate_synthetic_chargers()

def generate_synthetic_chargers():
    np.random.seed(42)
    n = 200

    # UK chargers
    uk_chargers = pd.DataFrame({
        'id': range(1, 101),
        'lat': np.random.uniform(51.0, 53.0, 100),
        'lon': np.random.uniform(-2.0, 0.5, 100),
        'capacity': np.random.choice([1, 2, 4, 6], 100),
        'operator': np.random.choice(['BP Pulse', 'Pod Point', 'Osprey', 'Unknown'], 100),
        'location_type': np.random.choice(['motorway', 'supermarket', 'council', 'other'], 100),
        'country': 'UK'
    })

    # US chargers
    us_chargers = pd.DataFrame({
        'id': range(101, 201),
        'lat': np.random.uniform(34.0, 47.0, 100),
        'lon': np.random.uniform(-122.5, -70.0, 100),
        'capacity': np.random.choice([2, 4, 8, 12], 100),
        'operator': np.random.choice(['Tesla', 'ChargePoint', 'EVgo', 'Blink', 'Electrify America'], 100),
        'location_type': np.random.choice(['motorway', 'supermarket', 'tesla', 'other'], 100),
        'country': 'US'
    })

    return pd.concat([uk_chargers, us_chargers], ignore_index=True)

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
    accuracy = model.score(X_test, y_test)
    print(f"Model trained! Accuracy: {accuracy:.2%}")

    with open('charger_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

# Load or train model
chargers = load_chargers()

if os.path.exists('charger_model.pkl'):
    with open('charger_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded from file")
else:
    print("Training model...")
    model = train_model(chargers)

# Postcode lookup
nomi_uk = pgeocode.Nominatim('GB')
nomi_us = pgeocode.Nominatim('US')

def lookup_location(query):
    """Try UK postcode first, then US ZIP code"""
    query = query.strip().upper()
    
    # Try UK first
    result = nomi_uk.query_postal_code(query)
    if result is not None and not pd.isna(result.latitude):
        return float(result.latitude), float(result.longitude), 'UK'
    
    # Try US ZIP
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

        # Get location
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

        # Filter chargers by country if we know it
        local_chargers = chargers.copy()
        if country and 'country' in local_chargers.columns:
            local_chargers = local_chargers[local_chargers['country'] == country]

        # Find nearby chargers
        nearby_list = []
        for _, c in local_chargers.iterrows():
            dist = geodesic((user_lat, user_lon), (c['lat'], c['lon'])).miles
            if dist <= radius_miles:
                nearby_list.append((dist, c))

        nearby_list.sort(key=lambda x: x[0])
        nearby_list = nearby_list[:20]

        results = []
        is_weekend = 1 if day_of_week >= 5 else 0
        for dist, c in nearby_list:
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
                'country': str(c.get('country', 'UK'))
            })

        return jsonify({'total': len(results), 'chargers': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
