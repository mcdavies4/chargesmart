from flask import Flask, request, jsonify, render_template, redirect, make_response, session, url_for
import pandas as pd
import numpy as np
import pickle
import os

# Global intelligence database
try:
    from global_data import get_country, all_countries, by_continent, needs_investment, score as readiness_score
    GLOBAL_DATA_LOADED = True
except ImportError:
    GLOBAL_DATA_LOADED = False

# Railway Volume mount point — set DATA_DIR env var in Railway to /data
# Both web and worker services must mount the same volume at /data
DATA_DIR = os.environ.get('DATA_DIR', '.')

def data_path(filename):
    """Returns path to a data file, using DATA_DIR if set."""
    return os.path.join(DATA_DIR, filename)

import requests
import json
try:
    import stripe
except ImportError:
    stripe = None
try:
    import pgeocode
    from geopy.distance import geodesic
except ImportError:
    pgeocode = None
    geodesic = None

app = Flask(__name__, static_folder='static')

if stripe: stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
# Consumer plan price IDs
PRO_PRICE_ID   = os.environ.get('STRIPE_PRO_PRICE_ID', '')
FLEET_PRICE_ID = os.environ.get('STRIPE_FLEET_PRICE_ID', '')

# API tier price IDs
API_DEVELOPER_PRICE_ID = os.environ.get('STRIPE_API_DEVELOPER_PRICE_ID', '')
API_BUSINESS_PRICE_ID  = os.environ.get('STRIPE_API_BUSINESS_PRICE_ID', '')
API_ENTERPRISE_PRICE_ID= os.environ.get('STRIPE_API_ENTERPRISE_PRICE_ID', '')

OCM_API_KEY = 'd29bb079-0234-40d8-b0af-a2bb55a7d399'
OCM_URL = 'https://api.openchargemap.io/v3/poi'

SUBSCRIBERS = {}

def load_subscribers():
    global SUBSCRIBERS
    if os.path.exists('subscribers.json'):
        with open('subscribers.json') as f:
            SUBSCRIBERS = json.load(f)

def save_subscribers():
    with open('subscribers.json', 'w') as f:
        json.dump(SUBSCRIBERS, f)

def get_plan(email):
    if not email:
        return 'free'
    return SUBSCRIBERS.get(email.lower(), 'free')

load_subscribers()

def load_chargers():
    if os.path.exists(data_path('enriched_data.csv')):
        df = pd.read_csv(data_path('enriched_data.csv'))
        chargers = df.groupby('id').agg({
            'lat': 'first', 'lon': 'first', 'capacity': 'first',
            'operator': 'first', 'location_type': 'first', 'country': 'first'
        }).reset_index()
        print(f"Loaded {len(chargers)} unique chargers")
        return chargers
    else:
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
    with open(data_path('charger_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    return model

def get_live_chargers(lat, lon, radius_miles, country):
    try:
        radius_km = radius_miles * 1.60934
        country_code = 'GB' if country == 'UK' else ('US' if country == 'US' else None)
        params = {
            'key': OCM_API_KEY,
            'latitude': lat,
            'longitude': lon,
            'distance': radius_km,
            'distanceunit': 'KM',
            'maxresults': 30,
            'compact': True,
            'verbose': False,
        }
        if country_code:
            params['countrycode'] = country_code
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
                elif addr.get('Title'):
                    operator = addr['Title'][:30]
                elif addr.get('AddressLine1'):
                    operator = addr['AddressLine1'][:30]
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
                dist = geodesic((lat, lon), (c_lat, c_lon)).miles if geodesic else ((abs(lat-c_lat)**2 + abs(lon-c_lon)**2)**0.5 * 69)
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

chargers = load_chargers()

if os.path.exists(data_path('charger_model.pkl')):
    try:
        with open(data_path('charger_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from file")
    except Exception as e:
        print(f"Model load failed: {e}, retraining...")
        model = train_model(chargers) if chargers is not None and len(chargers) > 0 else None
else:
    print("Training model...")
    model = train_model(chargers) if chargers is not None and len(chargers) > 0 else None

try:
    nomi_uk = pgeocode.Nominatim('GB')
    nomi_us = pgeocode.Nominatim('US')
    nomi_de = pgeocode.Nominatim('DE')
    nomi_fr = pgeocode.Nominatim('FR')
    nomi_nl = pgeocode.Nominatim('NL')
    nomi_no = pgeocode.Nominatim('NO')
    nomi_se = pgeocode.Nominatim('SE')
except Exception:
    nomi_uk = nomi_us = nomi_de = nomi_fr = nomi_nl = nomi_no = nomi_se = None

def lookup_location(query):
    query = query.strip().upper()
    for nomi, country in [
        (nomi_uk, 'UK'),
        (nomi_us, 'US'),
        (nomi_de, 'EU'),
        (nomi_fr, 'EU'),
        (nomi_nl, 'EU'),
        (nomi_no, 'EU'),
        (nomi_se, 'EU'),
    ]:
        try:
            result = nomi.query_postal_code(query)
            if result is not None and not pd.isna(result.latitude):
                return float(result.latitude), float(result.longitude), country
        except:
            continue
    return None, None, None

@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/predict')
def predict():
    try:
        if model is None:
            return jsonify({'prediction':'free','probability_free':65,'probability_busy':35,'note':'Model initialising'})
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
        if model is None:
            return jsonify({'error': 'Model not available'}), 503
        hour = int(request.args.get('hour', 12))
        day_of_week = int(request.args.get('day_of_week', 0))
        radius_miles = float(request.args.get('radius', 1.0))
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

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        data     = request.get_json()
        plan     = data.get('plan', 'pro')
        email    = data.get('email', '')
        api_key  = data.get('api_key', '')
        base_url = request.host_url.rstrip('/')

        # Map plan to Stripe price ID
        price_map = {
            'pro':            PRO_PRICE_ID,
            'fleet':          FLEET_PRICE_ID,
            'api_developer':  API_DEVELOPER_PRICE_ID,
            'api_business':   API_BUSINESS_PRICE_ID,
            'api_enterprise': API_ENTERPRISE_PRICE_ID,
        }
        price_id = price_map.get(plan, PRO_PRICE_ID)
        if not price_id:
            return jsonify({'error': f'Price ID for plan "{plan}" not configured. Set env var in Railway.'}), 400

        # 14-day free trial for consumer plans, no trial for API plans
        is_api_plan = plan.startswith('api_')
        sub_data = {} if is_api_plan else {'trial_period_days': 14}

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            subscription_data=sub_data,
            customer_email=email or None,
            success_url=base_url + f'/payment-success?plan={plan}&api_key={api_key}',
            cancel_url=base_url + ('/' if not is_api_plan else '/developers'),
            metadata={'plan': plan, 'api_key': api_key},
        )
        return jsonify({'url': session.url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/payment-success')
def payment_success():
    plan    = request.args.get('plan', '')
    api_key = request.args.get('api_key', '')

    # Upgrade API key tier if this was an API plan purchase
    if plan.startswith('api_') and api_key:
        from api_system import load_keys, save_keys
        tier_map = {
            'api_developer':  'developer',
            'api_business':   'business',
            'api_enterprise': 'enterprise',
        }
        new_tier = tier_map.get(plan, 'developer')
        keys = load_keys()
        if api_key in keys:
            keys[api_key]['tier'] = new_tier
            save_keys(keys)

    return render_template('index.html')


@app.route('/check-subscription', methods=['POST'])
def check_subscription():
    data = request.get_json()
    email = data.get('email', '').lower()
    plan = get_plan(email)
    return jsonify({'plan': plan, 'email': email})

@app.route('/stripe-key')
def stripe_key():
    key = os.environ.get('STRIPE_SECRET_KEY', 'NOT FOUND')
    pub = os.environ.get('STRIPE_PUBLISHABLE_KEY', 'NOT FOUND')
    pro = os.environ.get('STRIPE_PRO_PRICE_ID', 'NOT FOUND')
    fleet = os.environ.get('STRIPE_FLEET_PRICE_ID', 'NOT FOUND')
    return jsonify({
        'secret_key_found': key != 'NOT FOUND',
        'secret_key_prefix': key[:12] if key != 'NOT FOUND' else 'NOT FOUND',
        'publishable_key_found': pub != 'NOT FOUND',
        'pro_price_found': pro != 'NOT FOUND',
        'fleet_price_found': fleet != 'NOT FOUND',
    })

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


# ── REVIEWS ─────────────────────────────────────────────────
import datetime

REVIEWS_FILE = 'reviews.json'

def load_reviews():
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE) as f:
            return json.load(f)
    return {}

def save_reviews(reviews):
    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews, f)

@app.route('/submit-review', methods=['POST'])
def submit_review():
    try:
        data = request.get_json()
        charger_id = str(data.get('charger_id', ''))
        rating = int(data.get('rating', 0))
        comment = str(data.get('comment', ''))[:200]
        operator = str(data.get('operator', 'Unknown'))[:40]
        if not charger_id or rating < 1 or rating > 5:
            return jsonify({'error': 'Invalid review'}), 400
        reviews = load_reviews()
        if charger_id not in reviews:
            reviews[charger_id] = {'operator': operator, 'ratings': [], 'comments': []}
        reviews[charger_id]['ratings'].append(rating)
        reviews[charger_id]['comments'].append({
            'rating': rating,
            'comment': comment,
            'date': datetime.datetime.now().strftime('%Y-%m-%d')
        })
        save_reviews(reviews)
        avg = sum(reviews[charger_id]['ratings']) / len(reviews[charger_id]['ratings'])
        return jsonify({'success': True, 'avg_rating': round(avg, 1), 'total': len(reviews[charger_id]['ratings'])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get-reviews/<charger_id>')
def get_reviews(charger_id):
    reviews = load_reviews()
    data = reviews.get(str(charger_id), {})
    if not data:
        return jsonify({'avg_rating': None, 'total': 0, 'comments': []})
    ratings = data.get('ratings', [])
    avg = sum(ratings) / len(ratings) if ratings else 0
    return jsonify({
        'avg_rating': round(avg, 1),
        'total': len(ratings),
        'comments': data.get('comments', [])[-5:]
    })

# ── FAULT REPORTER ───────────────────────────────────────────
FAULTS_FILE = 'faults.json'

def load_faults():
    if os.path.exists(FAULTS_FILE):
        with open(FAULTS_FILE) as f:
            return json.load(f)
    return {}

@app.route('/report-fault', methods=['POST'])
def report_fault():
    try:
        data = request.get_json()
        charger_id = str(data.get('charger_id', ''))
        fault_type = str(data.get('fault_type', 'other'))
        operator = str(data.get('operator', 'Unknown'))[:40]
        if not charger_id:
            return jsonify({'error': 'Invalid'}), 400
        faults = load_faults()
        if charger_id not in faults:
            faults[charger_id] = {'operator': operator, 'reports': []}
        faults[charger_id]['reports'].append({
            'fault_type': fault_type,
            'date': datetime.datetime.now().strftime('%Y-%m-%d')
        })
        with open(FAULTS_FILE, 'w') as f:
            json.dump(faults, f)
        return jsonify({'success': True, 'total_reports': len(faults[charger_id]['reports'])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── JOURNEY COST CALCULATOR ──────────────────────────────────
@app.route('/journey-cost', methods=['POST'])
def journey_cost():
    try:
        data = request.get_json()
        miles = float(data.get('miles', 0))
        car_model = data.get('car_model', 'average')
        currency = data.get('currency', 'gbp')
        EV_EFFICIENCY = {'nissan_leaf': 3.5, 'tesla_model3': 4.0, 'tesla_modelx': 2.8, 'vw_id3': 3.8, 'average': 3.5}
        ELEC_COST = {'gbp': 0.28, 'usd': 0.16, 'eur': 0.22}
        PETROL_COST = {'gbp': 0.155, 'usd': 0.12, 'eur': 0.14}
        PETROL_MPG = 35
        sym = {'gbp': '£', 'usd': '$', 'eur': '€'}[currency]
        mpkwh = EV_EFFICIENCY.get(car_model, 3.5)
        kwh_needed = miles / mpkwh
        ev_cost = kwh_needed * ELEC_COST[currency]
        petrol_cost = (miles / PETROL_MPG) * PETROL_COST[currency] * 4.546
        saving = petrol_cost - ev_cost
        co2_saved = (0.404 - 0.069) * miles
        return jsonify({
            'ev_cost': round(ev_cost, 2),
            'petrol_cost': round(petrol_cost, 2),
            'saving': round(saving, 2),
            'kwh_needed': round(kwh_needed, 1),
            'co2_saved': round(co2_saved, 1),
            'symbol': sym
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ── CHARGER DESERT MAP ──────────────────────────────────────
@app.route('/charger-deserts')
def charger_deserts():
    return render_template('index.html')

@app.route('/api/charger-deserts')
def api_charger_deserts():
    try:
        region = request.args.get('region', 'uk')

        # Grid definitions per region
        regions = {
            'uk': {
                'lat_min': 49.9, 'lat_max': 58.7,
                'lon_min': -6.0, 'lon_max': 1.8,
                'grid_step': 0.3,
                'name': 'United Kingdom',
                'country': 'UK'
            },
            'us_east': {
                'lat_min': 24.5, 'lat_max': 47.5,
                'lon_min': -85.0, 'lon_max': -66.0,
                'grid_step': 0.5,
                'name': 'US East Coast',
                'country': 'US'
            },
            'us_west': {
                'lat_min': 32.0, 'lat_max': 49.0,
                'lon_min': -124.5, 'lon_max': -104.0,
                'grid_step': 0.5,
                'name': 'US West Coast',
                'country': 'US'
            },
            'germany': {
                'lat_min': 47.2, 'lat_max': 55.1,
                'lon_min': 5.8, 'lon_max': 15.1,
                'grid_step': 0.3,
                'name': 'Germany',
                'country': 'EU'
            },
            'france': {
                'lat_min': 41.3, 'lat_max': 51.1,
                'lon_min': -5.2, 'lon_max': 9.6,
                'grid_step': 0.3,
                'name': 'France',
                'country': 'EU'
            },
            'netherlands': {
                'lat_min': 50.7, 'lat_max': 53.6,
                'lon_min': 3.3, 'lon_max': 7.3,
                'grid_step': 0.2,
                'name': 'Netherlands',
                'country': 'EU'
            },
            # ── AFRICA ──────────────────────────────────────
            'south_africa': {
                'lat_min': -34.8, 'lat_max': -22.1,
                'lon_min': 16.5,  'lon_max': 33.0,
                'grid_step': 0.5,
                'name': 'South Africa',
                'country': 'ZA'
            },
            'kenya': {
                'lat_min': -4.7, 'lat_max': 4.6,
                'lon_min': 33.9, 'lon_max': 41.9,
                'grid_step': 0.4,
                'name': 'Kenya',
                'country': 'KE'
            },
            'nigeria': {
                'lat_min': 4.3, 'lat_max': 13.9,
                'lon_min': 2.7, 'lon_max': 14.7,
                'grid_step': 0.5,
                'name': 'Nigeria',
                'country': 'NG'
            },
            'egypt': {
                'lat_min': 22.0, 'lat_max': 31.7,
                'lon_min': 24.7, 'lon_max': 37.1,
                'grid_step': 0.5,
                'name': 'Egypt',
                'country': 'EG'
            },
            'ethiopia': {
                'lat_min': 3.4, 'lat_max': 15.0,
                'lon_min': 33.0, 'lon_max': 48.0,
                'grid_step': 0.5,
                'name': 'Ethiopia',
                'country': 'ET'
            },
            'morocco': {
                'lat_min': 27.6, 'lat_max': 35.9,
                'lon_min': -13.2, 'lon_max': -1.1,
                'grid_step': 0.4,
                'name': 'Morocco',
                'country': 'MA'
            },
            # ── MIDDLE EAST ─────────────────────────────────
            'uae': {
                'lat_min': 22.6, 'lat_max': 26.1,
                'lon_min': 51.6, 'lon_max': 56.4,
                'grid_step': 0.2,
                'name': 'UAE',
                'country': 'AE'
            },
            # ── ASIA ────────────────────────────────────────
            'india': {
                'lat_min': 8.0, 'lat_max': 23.1,
                'lon_min': 72.8, 'lon_max': 80.5,
                'grid_step': 0.5,
                'name': 'India',
                'country': 'IN'
            },
            # ── LATIN AMERICA ───────────────────────────────
            'brazil': {
                'lat_min': -23.8, 'lat_max': -19.8,
                'lon_min': -46.9, 'lon_max': -43.1,
                'grid_step': 0.3,
                'name': 'Brazil',
                'country': 'BR'
            },
        }

        if region not in regions:
            region = 'uk'

        reg = regions[region]

        # Load charger data — sample for speed (still accurate)
        if os.path.exists(data_path('enriched_data.csv')):
            import pandas as pd
            import numpy as np
            df = pd.read_csv(data_path('enriched_data.csv'), usecols=['lat','lon','country'])
            if reg['country'] != 'ALL':
                df = df[df['country'] == reg['country']]
            # Sample max 50k rows — enough for accurate desert detection
            if len(df) > 50000:
                df = df.sample(50000, random_state=42)
            clats = df['lat'].values
            clons = df['lon'].values
        else:
            clats = np.array([])
            clons = np.array([])

        # Build grid using numpy — vectorised, 1000x faster
        DESERT_RADIUS = 0.08  # ~5 miles in degrees
        step = reg['grid_step']

        grid_lats = np.arange(reg['lat_min'], reg['lat_max'], step)
        grid_lons = np.arange(reg['lon_min'], reg['lon_max'], step)

        deserts = []
        covered = []

        for lat in grid_lats:
            for lon in grid_lons:
                if len(clats) > 0:
                    has_charger = bool(np.any(
                        (np.abs(clats - lat) < DESERT_RADIUS) &
                        (np.abs(clons - lon) < DESERT_RADIUS)
                    ))
                else:
                    has_charger = False

                pt = {'lat': round(float(lat), 3), 'lon': round(float(lon), 3)}
                if not has_charger:
                    deserts.append(pt)
                else:
                    covered.append(pt)

        total = len(deserts) + len(covered)
        desert_pct = round(len(deserts) / total * 100, 1) if total > 0 else 0

        # Build sub-region breakdown (quadrants)
        lat_mid = (reg['lat_min'] + reg['lat_max']) / 2
        lon_mid = (reg['lon_min'] + reg['lon_max']) / 2

        quadrants = {
            'North West': {'d': 0, 'c': 0},
            'North East': {'d': 0, 'c': 0},
            'South West': {'d': 0, 'c': 0},
            'South East': {'d': 0, 'c': 0},
        }
        for pt in deserts:
            ns = 'North' if pt['lat'] >= lat_mid else 'South'
            ew = 'West' if pt['lon'] < lon_mid else 'East'
            quadrants[f'{ns} {ew}']['d'] += 1
        for pt in covered:
            ns = 'North' if pt['lat'] >= lat_mid else 'South'
            ew = 'West' if pt['lon'] < lon_mid else 'East'
            quadrants[f'{ns} {ew}']['c'] += 1

        breakdown = []
        for name, vals in quadrants.items():
            total_q = vals['d'] + vals['c']
            pct_q = round(vals['d'] / total_q * 100, 1) if total_q > 0 else 0
            breakdown.append({
                'name': name,
                'desert_zones': vals['d'],
                'covered_zones': vals['c'],
                'desert_pct': pct_q
            })
        breakdown.sort(key=lambda x: -x['desert_pct'])

        return jsonify({
            'region': reg['name'],
            'deserts': deserts[:2000],
            'total_desert_zones': len(deserts),
            'total_covered_zones': len(covered),
            'desert_percentage': desert_pct,
            'charger_count': len(clats),
            'breakdown': breakdown
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ═══════════════════════════════════════════════════════════════
# CHARGESMART PUBLIC API v1
# ═══════════════════════════════════════════════════════════════
from api_system import create_api_key, validate_key, record_usage, get_key_stats, TIER_LIMITS, TIER_PRICES
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        api_key = auth.replace('Bearer ', '').strip()
        if not api_key:
            api_key = request.args.get('key', '')
        valid, tier, remaining, error = validate_key(api_key)
        if not valid:
            return jsonify({
                'error': error,
                'docs': 'https://chargesmart.online/developers'
            }), 401
        # Check endpoint access for this tier
        from api_system import check_endpoint_access
        allowed, access_error = check_endpoint_access(tier, request.path)
        if not allowed:
            return jsonify({
                'error': access_error,
                'your_tier': tier,
                'upgrade': 'https://chargesmart.online/developers#pricing'
            }), 403
        record_usage(api_key)
        request.api_tier = tier
        request.api_remaining = remaining
        return f(*args, **kwargs)
    return decorated

def api_response(data, api_key=''):
    """Wrap response with metadata"""
    try:
        resp = {
            'success': True,
            'data': data,
            'meta': {
                'version': 'v1',
                'calls_remaining_today': getattr(request, 'api_remaining', None),
                'docs': 'https://chargesmart.online/developers'
            }
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({'success': True, 'data': data})

# ── GET API KEY ──────────────────────────────────────────────
@app.route('/api/v1/keys/register', methods=['POST'])
def register_api_key():
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip().lower()
        if not email or '@' not in email:
            return jsonify({'error': 'Valid email required'}), 400

        # Ensure api_keys.json exists in DATA_DIR
        import json as _json
        if not os.path.exists(data_path('api_keys.json')):
            with open(data_path('api_keys.json'), 'w') as f:
                f.write('{}')

        result = create_api_key(email, tier='free')
        if result['existing']:
            msg = 'Your existing API key has been returned'
        else:
            msg = 'API key created successfully. Welcome to ChargeSmart API!'

        # Also record in signups.json as backup
        try:
            signups = _json.load(open('signups.json')) if os.path.exists('signups.json') else []
            if not any(s.get('email') == email for s in signups):
                import datetime as _dt
                signups.append({'email': email, 'source': 'api_key', 
                                'date': _dt.datetime.now().isoformat()})
                with open('signups.json', 'w') as f:
                    _json.dump(signups, f, indent=2)
        except Exception:
            pass

        return jsonify({
            'success':     True,
            'message':     msg,
            'api_key':     result['key'],
            'tier':        result['tier'],
            'daily_limit': TIER_LIMITS['free'],
            'docs':        'https://chargesmart.online/developers'
        })
    except Exception as e:
        import traceback
        print(f"register_api_key error: {traceback.format_exc()}")
        return jsonify({'error': 'Could not generate key. Please try again.', 'detail': str(e)}), 500

# ── USAGE STATS ──────────────────────────────────────────────
@app.route('/api/v1/keys/stats')
def api_key_stats():
    try:
        auth = request.headers.get('Authorization', '')
        api_key = auth.replace('Bearer ', '').strip() or request.args.get('key', '')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        stats = get_key_stats(api_key)
        if not stats:
            return jsonify({'error': 'Invalid API key'}), 401
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 1. PREDICT CHARGER AVAILABILITY ─────────────────────────
@app.route('/api/v1/predict')
@require_api_key
def api_predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not available'}), 503
        lat  = float(request.args.get('lat', 51.5))
        lon  = float(request.args.get('lon', -0.1))
        hour = int(request.args.get('hour', datetime.datetime.now().hour))
        day  = int(request.args.get('day', datetime.datetime.now().weekday()))
        cap  = int(request.args.get('capacity', 2))
        country = request.args.get('country', 'UK').upper()

        country_map = {'UK': 0, 'US': 1, 'EU': 2}
        loc_map_r = {'motorway': 3, 'supermarket': 2, 'council': 1, 'tesla': 2, 'other': 1}
        loc_type = request.args.get('location_type', 'other')

        features_vals = [[
            hour, day,
            1 if day >= 5 else 0,
            min(cap, 20),
            lat, lon,
            country_map.get(country, 0),
            loc_map_r.get(loc_type, 1)
        ]]

        import pickle, os
        if os.path.exists(data_path('charger_model.pkl')):
            model = pickle.load(open(data_path('charger_model.pkl'), 'rb'))
            proba = model.predict_proba(features_vals)[0]
            prob_free = round(float(proba[0]) * 100, 1)
            prediction = 'free' if prob_free >= 50 else 'busy'
        else:
            prob_free = 50.0
            prediction = 'unknown'

        # Find best time today
        best_hour = hour
        best_prob = prob_free
        for h in range(24):
            fv = [[h, day, 1 if day >= 5 else 0, min(cap,20), lat, lon, country_map.get(country,0), loc_map_r.get(loc_type,1)]]
            if os.path.exists(data_path('charger_model.pkl')):
                p = model.predict_proba(fv)[0]
                pf = round(float(p[0]) * 100, 1)
                if pf > best_prob:
                    best_prob = pf
                    best_hour = h

        return api_response({
            'prediction':      prediction,
            'probability_free': prob_free,
            'probability_busy': round(100 - prob_free, 1),
            'best_time_today': f"{best_hour:02d}:00",
            'best_time_probability': best_prob,
            'inputs': {
                'lat': lat, 'lon': lon,
                'hour': hour, 'day_of_week': day,
                'capacity': cap, 'country': country,
                'location_type': loc_type
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 2. CHARGER DESERTS ───────────────────────────────────────
@app.route('/api/v1/deserts')
@require_api_key
def api_deserts():
    try:
        region = request.args.get('region', 'uk')
        # Reuse existing desert logic
        with app.test_request_context(f'/api/charger-deserts?region={region}'):
            pass
        resp = api_charger_deserts.__wrapped__() if hasattr(api_charger_deserts, '__wrapped__') else api_charger_deserts()
        data = resp.get_json()
        return api_response(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 3. JOURNEY COST ──────────────────────────────────────────
@app.route('/api/v1/journey-cost')
@require_api_key
def api_journey_cost():
    try:
        miles    = float(request.args.get('miles', 0))
        car      = request.args.get('car', 'average')
        currency = request.args.get('currency', 'gbp').lower()
        if miles <= 0:
            return jsonify({'error': 'miles must be greater than 0'}), 400

        EV_EFF  = {'average':3.5,'nissan_leaf':3.5,'tesla_model3':4.0,'tesla_modelx':2.8,'vw_id3':3.8}
        ELEC    = {'gbp':0.28,'usd':0.16,'eur':0.22}
        PETROL  = {'gbp':0.155,'usd':0.12,'eur':0.14}
        SYMS    = {'gbp':'£','usd':'$','eur':'€'}

        kwh       = miles / (EV_EFF.get(car, 3.5))
        ev_cost   = kwh * ELEC.get(currency, 0.28)
        pet_cost  = (miles / 35) * PETROL.get(currency, 0.155) * 4.546
        saving    = pet_cost - ev_cost
        co2_saved = 0.335 * miles

        return api_response({
            'miles':          miles,
            'car_model':      car,
            'currency':       currency,
            'symbol':         SYMS.get(currency, '£'),
            'ev_cost':        round(ev_cost, 2),
            'petrol_cost':    round(pet_cost, 2),
            'saving':         round(saving, 2),
            'saving_pct':     round((saving / pet_cost) * 100, 1),
            'kwh_needed':     round(kwh, 1),
            'co2_saved_kg':   round(co2_saved, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 4. CARBON SAVINGS ────────────────────────────────────────
@app.route('/api/v1/carbon')
@require_api_key
def api_carbon():
    try:
        miles          = float(request.args.get('miles', 0))
        vehicle        = request.args.get('vehicle', 'medium')
        trips_per_week = int(request.args.get('trips_per_week', 1))
        currency       = request.args.get('currency', 'gbp').lower()
        if miles <= 0:
            return jsonify({'error': 'miles must be greater than 0'}), 400

        EV_CO2   = {'small':0.053,'medium':0.069,'large':0.098,'van':0.135}
        PETROL_CO2 = 0.404
        FUEL_COST  = {'gbp':0.155,'usd':0.193,'eur':0.175}
        EV_COST    = {'gbp':0.052,'usd':0.065,'eur':0.058}
        SYMS       = {'gbp':'£','usd':'$','eur':'€'}

        co2_per_trip   = (PETROL_CO2 - EV_CO2.get(vehicle, 0.069)) * miles
        money_per_trip = (FUEL_COST.get(currency,0.155) - EV_COST.get(currency,0.052)) * miles

        return api_response({
            'miles_per_trip':        miles,
            'vehicle_type':          vehicle,
            'trips_per_week':        trips_per_week,
            'currency':              currency,
            'symbol':                SYMS.get(currency,'£'),
            'co2_saved_per_trip_kg': round(co2_per_trip, 2),
            'co2_saved_weekly_kg':   round(co2_per_trip * trips_per_week, 2),
            'co2_saved_annual_kg':   round(co2_per_trip * trips_per_week * 52, 1),
            'trees_equivalent':      round(co2_per_trip * trips_per_week * 52 / 21.77, 1),
            'money_saved_per_trip':  round(money_per_trip, 2),
            'money_saved_weekly':    round(money_per_trip * trips_per_week, 2),
            'money_saved_annual':    round(money_per_trip * trips_per_week * 52, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 5. CHARGER REVIEWS ───────────────────────────────────────
@app.route('/api/v1/reviews')
@require_api_key
def api_reviews():
    try:
        lat    = float(request.args.get('lat', 51.5))
        lon    = float(request.args.get('lon', -0.1))
        radius = float(request.args.get('radius', 5))

        reviews = load_reviews()
        faults  = load_faults()
        results = []
        for charger_id, data in reviews.items():
            ratings = data.get('ratings', [])
            if not ratings:
                continue
            avg = round(sum(ratings) / len(ratings), 1)
            fault_count = len(faults.get(charger_id, {}).get('reports', []))
            results.append({
                'charger_id':    charger_id,
                'operator':      data.get('operator', 'Unknown'),
                'avg_rating':    avg,
                'total_reviews': len(ratings),
                'recent_faults': fault_count,
                'comments':      data.get('comments', [])[-3:],
            })
        results.sort(key=lambda x: -x['total_reviews'])
        return api_response({
            'total_chargers_with_reviews': len(results),
            'chargers': results[:50]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── DEVELOPER DASHBOARD PAGE ─────────────────────────────────
@app.route('/developers')
def developers_page():
    resp = make_response(render_template('developers.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# ── NETWORK UPTIME LEADERBOARD ───────────────────────────────
@app.route('/leaderboard')
def leaderboard_page():
    return render_template('leaderboard.html')

@app.route('/api/leaderboard')
def api_leaderboard():
    try:
        faults  = load_faults()
        reviews = load_reviews()

        # Known networks with base reliability scores
        networks = {
            'Tesla Supercharger': {'base': 96, 'country': 'UK/US/EU', 'type': 'DC Fast'},
            'Pod Point':          {'base': 88, 'country': 'UK',       'type': 'AC/DC'},
            'BP Pulse':           {'base': 71, 'country': 'UK',       'type': 'AC/DC'},
            'Osprey':             {'base': 85, 'country': 'UK',       'type': 'DC Fast'},
            'InstaVolt':          {'base': 89, 'country': 'UK',       'type': 'DC Fast'},
            'Gridserve':          {'base': 92, 'country': 'UK',       'type': 'DC Fast'},
            'ChargePoint':        {'base': 84, 'country': 'US',       'type': 'AC/DC'},
            'EVgo':               {'base': 79, 'country': 'US',       'type': 'DC Fast'},
            'Electrify America':  {'base': 76, 'country': 'US',       'type': 'DC Fast'},
            'Blink':              {'base': 68, 'country': 'US',       'type': 'AC/DC'},
            'Ionity':             {'base': 91, 'country': 'EU',       'type': 'DC Fast'},
            'Allego':             {'base': 83, 'country': 'EU',       'type': 'AC/DC'},
            'Fastned':            {'base': 94, 'country': 'EU',       'type': 'DC Fast'},
            'Vattenfall':         {'base': 87, 'country': 'EU',       'type': 'AC/DC'},
        }

        results = []
        for name, info in networks.items():
            # Count community fault reports for this network
            fault_count = 0
            review_count = 0
            total_rating = 0
            for cid, fdata in faults.items():
                op = fdata.get('operator', '').lower()
                if any(n.lower() in op for n in name.lower().split()):
                    fault_count += len(fdata.get('reports', []))
            for cid, rdata in reviews.items():
                op = rdata.get('operator', '').lower()
                if any(n.lower() in op for n in name.lower().split()):
                    ratings = rdata.get('ratings', [])
                    review_count += len(ratings)
                    total_rating += sum(ratings)

            # Adjust uptime based on faults
            uptime = info['base']
            uptime -= min(fault_count * 2, 15)
            uptime = max(min(uptime, 99), 40)

            avg_rating = round(total_rating / review_count, 1) if review_count > 0 else None

            results.append({
                'name':         name,
                'uptime':       uptime,
                'country':      info['country'],
                'type':         info['type'],
                'fault_reports': fault_count,
                'review_count': review_count,
                'avg_rating':   avg_rating,
                'trend':        'up' if uptime >= info['base'] else 'down',
            })

        results.sort(key=lambda x: -x['uptime'])
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return jsonify({
            'success': True,
            'networks': results,
            'total_networks': len(results),
            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── FLEET PDF REPORT ─────────────────────────────────────────
@app.route('/fleet-report-pdf', methods=['POST'])
def fleet_report_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io

        data = request.get_json() or {}
        fleet_name    = data.get('fleet_name', 'My Fleet')
        vehicles      = data.get('vehicles', [])
        month         = data.get('month', datetime.datetime.now().strftime('%B %Y'))
        total_miles   = sum(v.get('miles', 0) for v in vehicles)
        total_sessions= sum(v.get('sessions', 0) for v in vehicles)

        # Calculations
        CO2_PER_MILE  = 0.335
        COST_PER_MILE_EV     = 0.052
        COST_PER_MILE_PETROL = 0.155
        total_co2     = round(total_miles * CO2_PER_MILE, 1)
        total_saving  = round(total_miles * (COST_PER_MILE_PETROL - COST_PER_MILE_EV), 2)
        trees_eq      = round(total_co2 / 21.77, 1)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=20*mm, leftMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)

        styles = getSampleStyleSheet()
        GREEN  = colors.HexColor('#00C851')
        DARK   = colors.HexColor('#0a0a0f')
        GREY   = colors.HexColor('#6b6b80')

        title_style = ParagraphStyle('Title', fontSize=24, fontName='Helvetica-Bold',
                                     textColor=DARK, spaceAfter=4)
        sub_style   = ParagraphStyle('Sub',   fontSize=12, fontName='Helvetica',
                                     textColor=GREY, spaceAfter=20)
        label_style = ParagraphStyle('Label', fontSize=9,  fontName='Helvetica',
                                     textColor=GREY)
        val_style   = ParagraphStyle('Val',   fontSize=22, fontName='Helvetica-Bold',
                                     textColor=GREEN)
        section_style = ParagraphStyle('Section', fontSize=13, fontName='Helvetica-Bold',
                                        textColor=DARK, spaceBefore=16, spaceAfter=8)

        story = []

        # Header
        story.append(Paragraph(f'⚡ ChargeSmart Fleet Report', title_style))
        story.append(Paragraph(f'{fleet_name} · {month}', sub_style))
        story.append(Spacer(1, 8))

        # Summary stats table
        stats_data = [
            [Paragraph('TOTAL MILES', label_style), Paragraph('CHARGING SESSIONS', label_style),
             Paragraph('CO2 SAVED', label_style), Paragraph('MONEY SAVED', label_style)],
            [Paragraph(f'{total_miles:,}', val_style), Paragraph(f'{total_sessions}', val_style),
             Paragraph(f'{total_co2}kg', val_style), Paragraph(f'£{total_saving:,.2f}', val_style)],
        ]
        stats_table = Table(stats_data, colWidths=[42*mm, 50*mm, 42*mm, 42*mm])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fff8')),
            ('BOX',        (0,0), (-1,-1), 1, colors.HexColor('#e0ffe0')),
            ('INNERGRID',  (0,0), (-1,-1), 0.5, colors.HexColor('#e0ffe0')),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f0fff0'), colors.white]),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 8))

        # Trees equivalent callout
        story.append(Paragraph(f'🌳  Equivalent to planting {trees_eq} trees this month', sub_style))

        # Vehicle breakdown
        if vehicles:
            story.append(Paragraph('Vehicle Breakdown', section_style))
            veh_data = [['Vehicle', 'Driver', 'Miles', 'Sessions', 'CO2 Saved', 'Cost Saved']]
            for v in vehicles:
                m = v.get('miles', 0)
                s = v.get('sessions', 0)
                co2 = round(m * CO2_PER_MILE, 1)
                sav = round(m * (COST_PER_MILE_PETROL - COST_PER_MILE_EV), 2)
                veh_data.append([
                    v.get('plate', 'Unknown'),
                    v.get('driver', 'Unknown'),
                    f'{m:,}',
                    str(s),
                    f'{co2}kg',
                    f'£{sav:.2f}',
                ])
            veh_table = Table(veh_data, colWidths=[30*mm, 35*mm, 25*mm, 25*mm, 28*mm, 28*mm])
            veh_table.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#00C851')),
                ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
                ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0,0), (-1,-1), 9),
                ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')]),
                ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
                ('INNERGRID',     (0,0), (-1,-1), 0.25, colors.HexColor('#eeeeee')),
                ('TOPPADDING',    (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(veh_table)

        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f'Generated by ChargeSmart · chargesmart.online · {datetime.datetime.now().strftime("%d %b %Y")}',
            ParagraphStyle('Footer', fontSize=8, textColor=GREY, alignment=TA_CENTER)
        ))

        doc.build(story)
        buf.seek(0)

        from flask import send_file
        return send_file(
            buf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'ChargeSmart_Fleet_Report_{month.replace(" ","_")}.pdf'
        )
    except ImportError:
        return jsonify({'error': 'PDF generation requires reportlab. Run: pip install reportlab'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ═══════════════════════════════════════════════════════════════
# SIGNUP TRACKING
# ═══════════════════════════════════════════════════════════════
SIGNUPS_FILE = 'signups.json'

def load_signups():
    if os.path.exists(SIGNUPS_FILE):
        with open(SIGNUPS_FILE) as f:
            return json.load(f)
    return []

def save_signups(signups):
    with open(SIGNUPS_FILE, 'w') as f:
        json.dump(signups, f, indent=2)

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json() or {}
        name   = data.get('name', '').strip()
        email  = data.get('email', '').strip().lower()
        source = data.get('source', '').strip()

        if not email or '@' not in email:
            return jsonify({'error': 'Valid email required'}), 400
        if not name:
            return jsonify({'error': 'Name required'}), 400

        signups = load_signups()

        # Check duplicate
        for s in signups:
            if s.get('email') == email:
                return jsonify({'success': True, 'message': 'You are already signed up!', 'duplicate': True})

        signups.append({
            'id':        len(signups) + 1,
            'name':      name,
            'email':     email,
            'source':    source or 'Not specified',
            'date':      datetime.datetime.now().strftime('%Y-%m-%d'),
            'time':      datetime.datetime.now().strftime('%H:%M'),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        save_signups(signups)

        return jsonify({
            'success': True,
            'message': f'Welcome aboard {name}! You are user #{len(signups)}.',
            'user_number': len(signups)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/admin/signups')
def admin_signups():
    # Simple password check via query param
    pw = request.args.get('pw', '')
    if pw != os.environ.get('ADMIN_PW', 'chargesmart2026'):
        return jsonify({'error': 'Unauthorised'}), 401
    signups = load_signups()
    # Analytics
    from collections import Counter
    sources = Counter(s.get('source','Unknown') for s in signups)
    dates   = Counter(s.get('date','') for s in signups)
    return jsonify({
        'total_signups':  len(signups),
        'signups':        signups,
        'by_source':      dict(sources.most_common()),
        'by_date':        dict(sorted(dates.items())),
        'today':          sum(1 for s in signups if s.get('date') == datetime.datetime.now().strftime('%Y-%m-%d')),
        'this_week':      sum(1 for s in signups if s.get('date', '') >= (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')),
    })

@app.route('/admin')
def admin_dashboard():
    pw = request.args.get('pw', '')
    if pw != os.environ.get('ADMIN_PW', 'chargesmart2026'):
        return '''<!DOCTYPE html><html><body style="background:#0a0a0f;color:#fff;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column;gap:16px;">
        <div style="font-size:32px;">⚡</div>
        <form method="GET">
            <input name="pw" type="password" placeholder="Admin password" style="padding:10px;background:#1a1a2a;border:1px solid #333;color:#fff;border-radius:6px;font-family:monospace;">
            <button type="submit" style="padding:10px 20px;background:#00ff87;border:none;border-radius:6px;color:#000;font-family:monospace;font-weight:700;cursor:pointer;margin-left:8px;">ENTER</button>
        </form></body></html>''', 401
    return render_template('admin.html')


@app.route('/version')
def version():
    import datetime
    return jsonify({
        'version': '2.1.0',
        'deployed': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'endpoints': 85,
        'status': 'live'
    })

@app.route('/debug/test-key', methods=['GET','POST'])
def debug_test_key():
    """Test key registration directly"""
    import traceback
    try:
        from api_system import create_api_key, KEYS_FILE, load_keys
        result = create_api_key('test@test.com', tier='free')
        keys = load_keys()
        return jsonify({
            'success': True,
            'key': result['key'],
            'existing': result['existing'],
            'total_keys': len(keys),
            'KEYS_FILE': KEYS_FILE,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/debug/paths')
def debug_paths():
    import traceback
    data_dir = os.environ.get('DATA_DIR', '.')

    # Try ACTUALLY creating a key end-to-end
    create_result = None
    create_error  = None
    try:
        from api_system import create_api_key, load_keys, KEYS_FILE
        result = create_api_key('debug@chargesmart.online', tier='free')
        keys   = load_keys()
        create_result = {
            'result':       result,
            'KEYS_FILE':    KEYS_FILE,
            'total_keys':   len(keys),
            'key_preview':  result['key'][:20] + '...',
        }
    except Exception as e:
        create_error = traceback.format_exc()

    # Try auth - create a user
    auth_result = None
    auth_error  = None
    try:
        from auth import get_or_create_user, USERS_FILE, create_magic_token
        user  = get_or_create_user('debug@chargesmart.online')
        token = create_magic_token('debug@chargesmart.online')
        auth_result = {
            'USERS_FILE':  USERS_FILE,
            'user_uid':    user.get('uid'),
            'user_plan':   user.get('plan'),
            'token_len':   len(token),
        }
    except Exception as e:
        auth_error = traceback.format_exc()

    return jsonify({
        'DATA_DIR':          data_dir,
        'create_key':        create_result,
        'create_key_error':  create_error,
        'auth':              auth_result,
        'auth_error':        auth_error,
        'env_vars': {
            'DATA_DIR':      os.environ.get('DATA_DIR', 'NOT SET'),
            'BREVO_API_KEY': 'SET' if os.environ.get('BREVO_API_KEY') else 'NOT SET',
        }
    })


@app.route('/explorer')
def explorer_page():
    resp = make_response(render_template('explorer.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

# ═══════════════════════════════════════════════════════════════
# CHARGESMART API v1 — EXTENDED ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def load_charger_data(country=None):
    """Load enriched_data.csv, generating synthetic data if missing."""
    import pandas as pd
    if not os.path.exists(data_path('enriched_data.csv')):
        # Generate and save synthetic data so all endpoints work
        print("enriched_data.csv missing - generating synthetic data...")
        base = generate_synthetic_chargers()
        np.random.seed(42)
        n = len(base)
        rows = []
        for _ in range(50):  # 50 time samples per charger
            tmp = base.copy()
            tmp['hour']        = np.random.randint(0, 24, n)
            tmp['day_of_week'] = np.random.randint(0, 7, n)
            rows.append(tmp)
        df_full = pd.concat(rows, ignore_index=True)
        df_full.to_csv(data_path('enriched_data.csv'), index=False)
        print(f"Generated {len(df_full)} synthetic rows -> enriched_data.csv")
    cols = ['lat','lon','hour','day_of_week','capacity','location_type','operator','country']
    # Only load cols that exist
    available = pd.read_csv(data_path('enriched_data.csv'), nrows=0).columns.tolist()
    load_cols = [c for c in cols if c in available]
    df = pd.read_csv(data_path('enriched_data.csv'), usecols=load_cols)
    if country and country != 'ALL':
        df = df[df['country'] == country.upper()]
    return df if len(df) > 0 else None

# ── 1. PEAK HOURS ────────────────────────────────────────────
@app.route('/api/v1/peak-hours')
@require_api_key
def api_peak_hours():
    try:
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        country = request.args.get('country', 'UK').upper()
        radius  = float(request.args.get('radius', 0.5))

        import pandas as pd
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Filter to nearby chargers
        nearby = df[
            (abs(df['lat'] - lat) < radius) &
            (abs(df['lon'] - lon) < radius)
        ]
        if len(nearby) == 0:
            nearby = df  # fallback to country-wide patterns

        hour_counts = nearby.groupby('hour').size()
        total = hour_counts.sum()
        hours_data = []
        for h in range(24):
            count = int(hour_counts.get(h, 0))
            pct   = round(count / total * 100, 1) if total > 0 else 0
            hours_data.append({
                'hour':       h,
                'label':      f'{h:02d}:00',
                'demand_pct': pct,
                'demand':     'high' if pct > 6 else 'medium' if pct > 3 else 'low'
            })

        sorted_hours = sorted(hours_data, key=lambda x: -x['demand_pct'])
        return api_response({
            'lat': lat, 'lon': lon, 'country': country,
            'peak_hours':    sorted_hours[:3],
            'quietest_hours':list(reversed(sorted_hours))[:3],
            'best_time_to_charge': sorted_hours[-1]['label'],
            'worst_time_to_charge': sorted_hours[0]['label'],
            'hourly_breakdown': hours_data,
            'data_points': len(nearby)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 2. OPERATOR INTELLIGENCE ─────────────────────────────────
@app.route('/api/v1/operators')
@require_api_key
def api_operators():
    try:
        country = request.args.get('country', 'UK').upper()
        import pandas as pd
        df = load_charger_data(country if country != 'ALL' else None)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        ops = []
        for op_name, grp in df.groupby('operator'):
            if op_name.lower() in ['unknown','']:
                continue
            loc_dist = grp['location_type'].value_counts().to_dict()
            ops.append({
                'operator':        op_name,
                'total_chargers':  len(grp['lat'].drop_duplicates()),
                'avg_capacity':    round(float(grp['capacity'].mean()), 1),
                'max_capacity':    int(grp['capacity'].max()),
                'location_types':  loc_dist,
                'primary_location':grp['location_type'].mode()[0] if len(grp) > 0 else 'unknown',
                'countries':       list(grp['country'].unique()) if 'country' in grp.columns else [country],
                'lat_range':       [round(float(grp['lat'].min()),2), round(float(grp['lat'].max()),2)],
                'lon_range':       [round(float(grp['lon'].min()),2), round(float(grp['lon'].max()),2)],
            })

        ops.sort(key=lambda x: -x['total_chargers'])
        return api_response({
            'country':        country,
            'total_operators': len(ops),
            'operators':      ops[:50],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 3. COVERAGE SCORE ────────────────────────────────────────
@app.route('/api/v1/coverage')
@require_api_key
def api_coverage():
    try:
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        radius  = float(request.args.get('radius', 10))
        country = request.args.get('country', 'UK').upper()

        import pandas as pd
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Convert radius miles to degrees (~0.014 deg per mile)
        deg = radius * 0.014
        nearby = df[
            (abs(df['lat'] - lat) < deg) &
            (abs(df['lon'] - lon) < deg)
        ]

        unique_chargers = len(nearby[['lat','lon']].drop_duplicates())
        avg_capacity    = float(nearby['capacity'].mean()) if len(nearby) > 0 else 0
        operators       = nearby['operator'].nunique() if len(nearby) > 0 else 0
        has_fast        = 'motorway' in nearby['location_type'].values or avg_capacity >= 4

        # Score 0-100
        score = 0
        score += min(unique_chargers * 3, 40)  # up to 40 pts for charger count
        score += min(avg_capacity * 5, 20)      # up to 20 pts for capacity
        score += min(operators * 5, 20)         # up to 20 pts for network diversity
        score += 20 if has_fast else 0          # 20 pts for fast charging
        score = min(int(score), 100)

        grade = 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D' if score >= 20 else 'F'

        return api_response({
            'lat': lat, 'lon': lon,
            'radius_miles':     radius,
            'coverage_score':   score,
            'grade':            grade,
            'verdict':          'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Average' if score >= 40 else 'Poor' if score >= 20 else 'Charging Desert',
            'chargers_nearby':  unique_chargers,
            'avg_capacity':     round(avg_capacity, 1),
            'networks_present': operators,
            'has_fast_charging':has_fast,
            'recommendation':   'Well served area' if score >= 60 else 'Limited coverage — plan charging in advance',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 4. NEAREST CHARGERS + PREDICTION ────────────────────────
@app.route('/api/v1/nearest')
@require_api_key
def api_nearest():
    try:
        lat         = float(request.args.get('lat', 51.5))
        lon         = float(request.args.get('lon', -0.1))
        max_results = min(int(request.args.get('max_results', 5)), 20)
        country     = request.args.get('country', 'UK').upper()

        import pandas as pd, pickle, math
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        unique = df[['lat','lon','operator','capacity','location_type']].drop_duplicates(['lat','lon'])
        unique = unique.copy()
        unique['dist'] = unique.apply(
            lambda r: math.sqrt((r['lat']-lat)**2 + (r['lon']-lon)**2), axis=1
        )
        nearest = unique.nsmallest(max_results, 'dist')

        country_map  = {'UK':0,'US':1,'EU':2}
        loc_map      = {'motorway':3,'supermarket':2,'council':1,'tesla':2,'other':1}
        hour         = int(request.args.get('hour', datetime.datetime.now().hour))
        day          = int(request.args.get('day',  datetime.datetime.now().weekday()))
        c_code       = country_map.get(country, 0)

        model = None
        if os.path.exists(data_path('charger_model.pkl')):
            model = pickle.load(open(data_path('charger_model.pkl'),'rb'))

        results = []
        for _, row in nearest.iterrows():
            dist_miles = round(row['dist'] * 69, 2)
            loc_code   = loc_map.get(row['location_type'], 1)
            cap        = min(int(row['capacity']), 20)

            prob_free = 50.0
            if model:
                fv = [[hour, day, 1 if day>=5 else 0, cap,
                        row['lat'], row['lon'], c_code, loc_code]]
                prob_free = round(float(model.predict_proba(fv)[0][0]) * 100, 1)

            results.append({
                'lat':           round(float(row['lat']), 5),
                'lon':           round(float(row['lon']), 5),
                'operator':      str(row['operator']),
                'capacity':      cap,
                'location_type': str(row['location_type']),
                'distance_miles':dist_miles,
                'probability_free': prob_free,
                'prediction':    'free' if prob_free >= 50 else 'busy',
            })

        results.sort(key=lambda x: x['distance_miles'])
        return api_response({
            'lat': lat, 'lon': lon,
            'hour': hour, 'day': day,
            'total_found': len(results),
            'chargers': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 5. HEATMAP DATA ──────────────────────────────────────────
@app.route('/api/v1/heatmap')
@require_api_key
def api_heatmap():
    try:
        region = request.args.get('region', 'uk').lower()
        res    = request.args.get('resolution', 'low')

        import pandas as pd
        regions = {
            'uk':  {'country':'UK',  'lat':(49.9,58.7), 'lon':(-6.0,1.8)},
            'us':  {'country':'US',  'lat':(24.5,49.0), 'lon':(-125.0,-66.0)},
            'eu':  {'country':'EU',  'lat':(35.0,72.0), 'lon':(-10.0,32.0)},
        }
        reg = regions.get(region, regions['uk'])
        df  = load_charger_data(reg['country'])
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        step = 0.2 if res == 'high' else 0.5
        df_f = df[
            (df['lat'].between(*reg['lat'])) &
            (df['lon'].between(*reg['lon']))
        ]

        df_f = df_f.copy()
        df_f['grid_lat'] = (df_f['lat'] / step).round() * step
        df_f['grid_lon'] = (df_f['lon'] / step).round() * step
        grid = df_f.groupby(['grid_lat','grid_lon']).size().reset_index(name='count')
        max_count = int(grid['count'].max()) if len(grid) > 0 else 1

        points = []
        for _, row in grid.iterrows():
            points.append({
                'lat':       round(float(row['grid_lat']), 3),
                'lon':       round(float(row['grid_lon']), 3),
                'count':     int(row['count']),
                'intensity': round(float(row['count']) / max_count, 3),
            })

        return api_response({
            'region':      region,
            'resolution':  res,
            'grid_step':   step,
            'total_points':len(points),
            'max_density': max_count,
            'points':      points[:3000],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 6. COUNTRY COMPARISON ────────────────────────────────────
@app.route('/api/v1/compare')
@require_api_key
def api_compare():
    try:
        countries_param = request.args.get('countries', 'UK,US,EU')
        countries = [c.strip().upper() for c in countries_param.split(',')]

        import pandas as pd
        df_all = load_charger_data(None)
        if df_all is None:
            return jsonify({'error': 'Data not available'}), 503

        # Population for density calc
        populations = {'UK':67_000_000,'US':331_000_000,'EU':450_000_000}
        results = []
        for country in countries:
            df_c = df_all[df_all['country'] == country]
            if len(df_c) == 0:
                continue
            unique = df_c[['lat','lon']].drop_duplicates()
            pop    = populations.get(country, 100_000_000)
            results.append({
                'country':           country,
                'total_chargers':    len(unique),
                'chargers_per_100k': round(len(unique) / pop * 100_000, 2),
                'avg_capacity':      round(float(df_c['capacity'].mean()), 1),
                'top_operator':      df_c['operator'].mode()[0] if len(df_c) > 0 else 'Unknown',
                'top_location_type': df_c['location_type'].mode()[0] if len(df_c) > 0 else 'Unknown',
                'data_points':       len(df_c),
                'operators_count':   df_c['operator'].nunique(),
            })

        results.sort(key=lambda x: -x['chargers_per_100k'])
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return api_response({
            'countries_compared': countries,
            'winner':   results[0]['country'] if results else None,
            'results':  results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 7. OPERATOR COMPARISON ───────────────────────────────────
@app.route('/api/v1/operators/compare')
@require_api_key
def api_operators_compare():
    try:
        names_param = request.args.get('names', 'Tesla,BP Pulse')
        names = [n.strip() for n in names_param.split(',')]

        import pandas as pd
        df = load_charger_data(None)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        results = []
        for name in names:
            mask = df['operator'].str.lower().str.contains(name.lower(), na=False)
            grp  = df[mask]
            if len(grp) == 0:
                results.append({'operator': name, 'found': False})
                continue
            unique = grp[['lat','lon']].drop_duplicates()
            results.append({
                'operator':        name,
                'found':           True,
                'total_chargers':  len(unique),
                'avg_capacity':    round(float(grp['capacity'].mean()), 1),
                'countries':       list(grp['country'].unique()),
                'location_types':  grp['location_type'].value_counts().to_dict(),
                'peak_hour':       int(grp['hour'].mode()[0]),
                'data_points':     len(grp),
            })

        results.sort(key=lambda x: -(x.get('total_chargers', 0)))
        return api_response({
            'operators_compared': names,
            'winner': results[0]['operator'] if results and results[0].get('found') else None,
            'results': results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 8. 24HR AVAILABILITY FORECAST ───────────────────────────
@app.route('/api/v1/forecast')
@require_api_key
def api_forecast():
    try:
        lat      = float(request.args.get('lat', 51.5))
        lon      = float(request.args.get('lon', -0.1))
        capacity = int(request.args.get('capacity', 2))
        country  = request.args.get('country', 'UK').upper()
        loc_type = request.args.get('location_type', 'other')

        import pickle
        country_map = {'UK':0,'US':1,'EU':2}
        loc_map     = {'motorway':3,'supermarket':2,'council':1,'tesla':2,'other':1}
        c_code      = country_map.get(country, 0)
        loc_code    = loc_map.get(loc_type, 1)
        today       = datetime.datetime.now().weekday()
        cap         = min(capacity, 20)

        model = None
        if os.path.exists(data_path('charger_model.pkl')):
            model = pickle.load(open(data_path('charger_model.pkl'),'rb'))

        forecast = []
        for h in range(24):
            prob_free = 50.0
            if model:
                fv = [[h, today, 1 if today>=5 else 0, cap, lat, lon, c_code, loc_code]]
                prob_free = round(float(model.predict_proba(fv)[0][0]) * 100, 1)
            forecast.append({
                'hour':            h,
                'time':            f'{h:02d}:00',
                'probability_free': prob_free,
                'probability_busy': round(100 - prob_free, 1),
                'prediction':      'free' if prob_free >= 50 else 'busy',
                'confidence':      'high' if abs(prob_free - 50) > 25 else 'medium' if abs(prob_free - 50) > 10 else 'low',
            })

        best  = max(forecast, key=lambda x: x['probability_free'])
        worst = min(forecast, key=lambda x: x['probability_free'])

        return api_response({
            'lat': lat, 'lon': lon,
            'country': country,
            'forecast_day': datetime.datetime.now().strftime('%A'),
            'best_time':  best['time'],
            'worst_time': worst['time'],
            'best_probability_free':  best['probability_free'],
            'worst_probability_free': worst['probability_free'],
            'hourly_forecast': forecast,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 9. INFRASTRUCTURE GAP SCORE ──────────────────────────────
@app.route('/api/v1/gap-score')
@require_api_key
def api_gap_score():
    try:
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        country = request.args.get('country', 'UK').upper()

        import pandas as pd
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Check chargers within 1, 3, 5, 10 miles
        def count_within(miles):
            deg = miles * 0.014
            return len(df[
                (abs(df['lat'] - lat) < deg) &
                (abs(df['lon'] - lon) < deg)
            ][['lat','lon']].drop_duplicates())

        c1  = count_within(1)
        c3  = count_within(3)
        c5  = count_within(5)
        c10 = count_within(10)

        # Gap score — higher = bigger gap = more need
        gap = 100
        gap -= min(c1 * 20, 40)
        gap -= min(c3 * 5,  25)
        gap -= min(c5 * 2,  20)
        gap -= min(c10 * 1, 15)
        gap = max(gap, 0)

        priority = 'Critical' if gap >= 80 else 'High' if gap >= 60 else 'Medium' if gap >= 40 else 'Low' if gap >= 20 else 'Well Served'

        return api_response({
            'lat': lat, 'lon': lon,
            'country': country,
            'gap_score':          gap,
            'infrastructure_priority': priority,
            'chargers_within_1_mile':  c1,
            'chargers_within_3_miles': c3,
            'chargers_within_5_miles': c5,
            'chargers_within_10_miles':c10,
            'recommendation': 'Urgent infrastructure investment needed' if gap >= 80
                         else 'New chargers would significantly help this area' if gap >= 60
                         else 'Moderate coverage — additional chargers beneficial' if gap >= 40
                         else 'Adequate coverage for most EV drivers',
            'useful_for': ['Local councils','Energy companies','EV network operators','Urban planners'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 10. BATCH PREDICT ────────────────────────────────────────
@app.route('/api/v1/predict/batch', methods=['POST'])
@require_api_key
def api_batch_predict():
    try:
        import pickle
        data     = request.get_json() or {}
        chargers = data.get('chargers', [])

        if not chargers:
            return jsonify({'error': 'Provide chargers array in request body'}), 400
        if len(chargers) > 50:
            return jsonify({'error': 'Maximum 50 chargers per batch request'}), 400

        country_map = {'UK':0,'US':1,'EU':2}
        loc_map     = {'motorway':3,'supermarket':2,'council':1,'tesla':2,'other':1}
        now         = datetime.datetime.now()

        model = None
        if os.path.exists(data_path('charger_model.pkl')):
            model = pickle.load(open(data_path('charger_model.pkl'),'rb'))

        results = []
        for ch in chargers:
            lat      = float(ch.get('lat', 51.5))
            lon      = float(ch.get('lon', -0.1))
            hour     = int(ch.get('hour', now.hour))
            day      = int(ch.get('day',  now.weekday()))
            cap      = min(int(ch.get('capacity', 2)), 20)
            country  = ch.get('country', 'UK').upper()
            loc_type = ch.get('location_type', 'other')
            c_code   = country_map.get(country, 0)
            loc_code = loc_map.get(loc_type, 1)

            prob_free = 50.0
            if model:
                fv = [[hour, day, 1 if day>=5 else 0, cap, lat, lon, c_code, loc_code]]
                prob_free = round(float(model.predict_proba(fv)[0][0]) * 100, 1)

            results.append({
                'id':               ch.get('id', f'{lat},{lon}'),
                'lat':              lat,
                'lon':              lon,
                'probability_free': prob_free,
                'probability_busy': round(100 - prob_free, 1),
                'prediction':       'free' if prob_free >= 50 else 'busy',
            })

        free_count = sum(1 for r in results if r['prediction'] == 'free')
        return api_response({
            'total_chargers':   len(results),
            'predicted_free':   free_count,
            'predicted_busy':   len(results) - free_count,
            'results':          results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ═══════════════════════════════════════════════════════════════
# GOVERNMENT & BUSINESS ENDPOINTS
# ═══════════════════════════════════════════════════════════════



@app.route('/report-builder')
def report_builder():
    return render_template('report_builder.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/global')
def global_dashboard():
    return render_template('global_dashboard.html')


# ══════════════════════════════════════════════════════════════
# GLOBAL INTELLIGENCE API  ·  /api/v1/global/*
# ══════════════════════════════════════════════════════════════

@app.route('/api/v1/global/index')
@require_api_key
def global_index():
    """
    Full ranked list of all 106 countries by EV readiness score.
    Optional: ?continent=Africa&tier=HIGH&limit=20
    """
    try:
        if not GLOBAL_DATA_LOADED:
            return jsonify({'error': 'Global data module not loaded'}), 500

        countries = all_countries()

        continent = request.args.get('continent', '').strip()
        tier      = request.args.get('tier', '').strip().upper()
        limit     = int(request.args.get('limit', 200))

        if continent:
            countries = [c for c in countries if c['continent'].lower() == continent.lower()]
        if tier:
            countries = [c for c in countries if c['tier'] == tier]

        countries = countries[:limit]

        return jsonify({
            'total':      len(countries),
            'filters':    {'continent': continent or 'all', 'tier': tier or 'all'},
            'avg_score':  round(sum(c['score'] for c in countries) / len(countries), 1) if countries else 0,
            'countries':  countries,
            'source':     'ChargeSmart Global Intelligence Database · March 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/global/country/<code>')
@require_api_key
def global_country(code):
    """
    Full EV readiness profile for a single country.
    Returns score, tier, breakdown, corridors, and $1M ROI estimate.
    """
    try:
        if not GLOBAL_DATA_LOADED:
            return jsonify({'error': 'Global data module not loaded'}), 500

        c = get_country(code.upper())
        if not c:
            return jsonify({'error': f'Country code {code.upper()} not found'}), 404

        # Cost multiplier by GDP
        gdp = c.get('gdp_per_cap', 5000)
        cost_mult = 0.70 if gdp < 3000 else 0.80 if gdp < 8000 else 0.95 if gdp < 30000 else 1.10
        cost_per_charger = 8000 * cost_mult

        budget = 1_000_000
        chargers     = int(budget / cost_per_charger)
        vehicles_day = chargers * 18
        carbon_year  = round(chargers * 22 * 0.8 * 365 * 0.00021)
        jobs         = chargers * 2
        revenue_year = round(chargers * 22 * 0.6 * 365 * 0.15)
        payback_yrs  = round(budget / revenue_year, 1) if revenue_year else None

        import_score = max(0, 100 - c.get('tariff', 25) * 2)

        return jsonify({
            'code':       c['code'],
            'name':       c['name'],
            'continent':  c['continent'],
            'score':      c['score'],
            'tier':       c['tier'],
            'breakdown': {
                'renewable_energy':   {'score': c['renewable'],  'weight': '30%'},
                'policy_strength':    {'score': c['policy'],     'weight': '30%'},
                'grid_reliability':   {'score': c['grid'],       'weight': '25%'},
                'import_conditions':  {'score': import_score,    'weight': '15%', 'tariff_pct': c['tariff']},
            },
            'infrastructure': {
                'existing_chargers': c['chargers'],
                'population_m':      c['population'],
                'gdp_per_capita':    gdp,
                'chargers_per_100k': round(c['chargers'] / (c['population'] * 10), 1) if c['population'] else 0,
            },
            'corridors':  c.get('corridors', []),
            'roi_1m_usd': {
                'chargers_deployed':  chargers,
                'vehicles_served_day': vehicles_day,
                'co2_saved_year_t':   carbon_year,
                'jobs_created':       jobs,
                'annual_revenue_usd': revenue_year,
                'payback_years':      payback_yrs,
                'cost_per_charger_usd': int(cost_per_charger),
            },
            'source': 'ChargeSmart Global Intelligence Database · March 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/global/compare')
@require_api_key
def global_compare():
    """
    Compare up to 5 countries side by side.
    ?codes=KE,ET,RW,ZA,NG
    """
    try:
        if not GLOBAL_DATA_LOADED:
            return jsonify({'error': 'Global data module not loaded'}), 500

        codes_param = request.args.get('codes', '')
        if not codes_param:
            return jsonify({'error': 'Provide ?codes=KE,ET,RW (up to 5)'}), 400

        codes = [c.strip().upper() for c in codes_param.split(',')][:5]
        results = []
        not_found = []

        for code in codes:
            c = get_country(code)
            if c:
                results.append({
                    'code':      c['code'],
                    'name':      c['name'],
                    'score':     c['score'],
                    'tier':      c['tier'],
                    'renewable': c['renewable'],
                    'policy':    c['policy'],
                    'grid':      c['grid'],
                    'tariff':    c['tariff'],
                    'chargers':  c['chargers'],
                    'continent': c['continent'],
                })
            else:
                not_found.append(code)

        if not results:
            return jsonify({'error': 'No valid country codes provided'}), 400

        results.sort(key=lambda x: x['score'], reverse=True)
        winner = results[0]

        return jsonify({
            'countries':   results,
            'winner':      {'code': winner['code'], 'name': winner['name'], 'score': winner['score']},
            'not_found':   not_found,
            'source':      'ChargeSmart Global Intelligence Database · March 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/global/investment-gaps')
@require_api_key
def global_investment_gaps():
    """
    Countries ranked by infrastructure urgency — most chargers needed first.
    Ideal for DFI and government funding allocation decisions.
    ?min_gap=70&continent=Africa&limit=20
    """
    try:
        if not GLOBAL_DATA_LOADED:
            return jsonify({'error': 'Global data module not loaded'}), 500

        min_gap   = int(request.args.get('min_gap', 70))
        continent = request.args.get('continent', '').strip()
        limit     = int(request.args.get('limit', 50))

        gaps = needs_investment(min_gap=min_gap)

        if continent:
            gaps = [g for g in gaps if g['continent'].lower() == continent.lower()]

        gaps = gaps[:limit]

        total_chargers_needed = sum(g['total_chargers_needed'] for g in gaps)
        total_corridors       = sum(len(g['critical_corridors']) for g in gaps)

        return jsonify({
            'summary': {
                'countries_with_gaps':     len(gaps),
                'total_chargers_needed':   total_chargers_needed,
                'total_critical_corridors':total_corridors,
                'min_gap_threshold':       min_gap,
                'estimated_investment_usd': total_chargers_needed * 8000,
            },
            'countries': gaps,
            'source':    'ChargeSmart Global Intelligence Database · March 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/global/leaderboard')
def global_leaderboard_api():
    """
    Public leaderboard — top 20 countries by readiness score.
    No API key required (used on public leaderboard page).
    Optional: ?continent=Africa
    """
    try:
        if not GLOBAL_DATA_LOADED:
            return jsonify({'error': 'Global data module not loaded'}), 500

        continent = request.args.get('continent', '').strip()
        countries = all_countries()

        if continent:
            countries = [c for c in countries if c['continent'].lower() == continent.lower()]

        top = [{
            'rank':      i + 1,
            'code':      c['code'],
            'name':      c['name'],
            'score':     c['score'],
            'tier':      c['tier'],
            'continent': c['continent'],
            'renewable': c['renewable'],
            'chargers':  c['chargers'],
        } for i, c in enumerate(countries[:20])]

        return jsonify({
            'leaderboard': top,
            'total_countries': len(countries),
            'continent_filter': continent or 'global',
            'source': 'ChargeSmart Global Intelligence Database · March 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/global/continent-summary')
@require_api_key
def global_continent_summary():
    """
    Aggregate EV readiness stats per continent.
    Good for regional reports and investor briefings.
    """
    try:
        if not GLOBAL_DATA_LOADED:
            return jsonify({'error': 'Global data module not loaded'}), 500

        grouped = by_continent()
        summary = []

        for continent, countries in grouped.items():
            scores    = [c['score'] for c in countries]
            chargers  = sum(c['chargers'] for c in countries)
            high      = sum(1 for c in countries if c['tier'] == 'HIGH')
            medium    = sum(1 for c in countries if c['tier'] == 'MEDIUM')
            low       = sum(1 for c in countries if c['tier'] == 'LOW')
            top       = max(countries, key=lambda x: x['score'])
            bottom    = min(countries, key=lambda x: x['score'])

            summary.append({
                'continent':      continent,
                'countries':      len(countries),
                'avg_score':      round(sum(scores) / len(scores), 1),
                'top_score':      max(scores),
                'bottom_score':   min(scores),
                'tier_breakdown': {'HIGH': high, 'MEDIUM': medium, 'LOW': low},
                'total_chargers': chargers,
                'leader':         {'code': top['code'], 'name': top['name'], 'score': top['score']},
                'needs_most_help':{'code': bottom['code'], 'name': bottom['name'], 'score': bottom['score']},
            })

        summary.sort(key=lambda x: x['avg_score'], reverse=True)

        return jsonify({
            'continents': summary,
            'global_avg': round(sum(c['avg_score'] for c in summary) / len(summary), 1),
            'source':     'ChargeSmart Global Intelligence Database · March 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/solutions')
def solutions_page():
    return render_template('solutions.html')

# ── GOV 1. INVESTMENT PRIORITY REPORT ───────────────────────
@app.route('/api/v1/gov/investment-priority')
@require_api_key
def api_gov_investment_priority():
    try:
        import pandas as pd
        country = request.args.get('country', 'UK').upper()
        top_n   = min(int(request.args.get('top', 20)), 50)

        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Build grid of areas and score each by gap
        step = 0.2
        df['grid_lat'] = (df['lat'] / step).round() * step
        df['grid_lon'] = (df['lon'] / step).round() * step
        grid = df.groupby(['grid_lat','grid_lon']).agg(
            charger_count=('lat','count'),
            avg_capacity=('capacity','mean'),
            operators=('operator','nunique'),
        ).reset_index()

        # Areas with very few chargers = highest priority
        grid = grid[grid['charger_count'] < 20].copy()
        grid['gap_score'] = 100 - (
            (grid['charger_count'] * 3).clip(0,40) +
            (grid['avg_capacity']  * 5).clip(0,20) +
            (grid['operators']     * 5).clip(0,20)
        )
        grid['gap_score'] = grid['gap_score'].clip(0, 100).round(1)
        grid['priority'] = grid['gap_score'].apply(
            lambda s: 'Critical' if s>=80 else 'High' if s>=60 else 'Medium' if s>=40 else 'Low'
        )
        grid = grid.nlargest(top_n, 'gap_score')

        areas = []
        for _, row in grid.iterrows():
            areas.append({
                'lat':           round(float(row['grid_lat']), 3),
                'lon':           round(float(row['grid_lon']), 3),
                'gap_score':     float(row['gap_score']),
                'priority':      row['priority'],
                'chargers_found':int(row['charger_count']),
                'avg_capacity':  round(float(row['avg_capacity']), 1),
                'networks_present': int(row['operators']),
                'recommended_action': 'Install 4+ bay DC fast hub' if row['gap_score'] >= 80
                                 else 'Install 2-4 AC chargers' if row['gap_score'] >= 60
                                 else 'Monitor and review in 12 months',
            })

        critical = sum(1 for a in areas if a['priority'] == 'Critical')
        return api_response({
            'country':          country,
            'total_priority_areas': len(areas),
            'critical_areas':   critical,
            'high_areas':       sum(1 for a in areas if a['priority'] == 'High'),
            'estimated_chargers_needed': critical * 4,
            'areas':            areas,
            'report_generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'suitable_for':     ['Planning applications','Budget allocation','Net zero reporting','Press releases'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── GOV 2. CHARGER DENSITY ───────────────────────────────────
@app.route('/api/v1/gov/charger-density')
@require_api_key
def api_gov_charger_density():
    try:
        import pandas as pd
        country = request.args.get('country', 'UK').upper()

        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Regional populations (approximate)
        uk_regions = {
            'London':       {'lat':(51.3,51.7),  'lon':(-0.5,0.3),   'pop':9_000_000},
            'South East':   {'lat':(50.8,51.4),  'lon':(-1.5,1.4),   'pop':9_200_000},
            'South West':   {'lat':(49.9,51.7),  'lon':(-5.7,-1.8),  'pop':5_700_000},
            'East England': {'lat':(51.5,52.9),  'lon':(-0.5,1.8),   'pop':6_300_000},
            'Midlands':     {'lat':(52.0,53.0),  'lon':(-2.2,-0.5),  'pop':5_600_000},
            'North West':   {'lat':(53.3,54.0),  'lon':(-3.2,-1.8),  'pop':7_400_000},
            'North East':   {'lat':(54.5,55.8),  'lon':(-2.5,-0.9),  'pop':2_600_000},
            'Yorkshire':    {'lat':(53.3,54.5),  'lon':(-2.0,-0.1),  'pop':5_500_000},
            'Wales':        {'lat':(51.3,53.5),  'lon':(-5.3,-2.6),  'pop':3_200_000},
            'Scotland':     {'lat':(54.6,58.7),  'lon':(-6.0,-0.7),  'pop':5_500_000},
        }

        results = []
        for region, info in uk_regions.items():
            mask = (
                df['lat'].between(*info['lat']) &
                df['lon'].between(*info['lon'])
            )
            count = len(df[mask][['lat','lon']].drop_duplicates())
            per_100k = round(count / info['pop'] * 100_000, 1)
            results.append({
                'region':            region,
                'charger_count':     count,
                'population':        info['pop'],
                'chargers_per_100k': per_100k,
                'grade':             'A' if per_100k>=50 else 'B' if per_100k>=30 else 'C' if per_100k>=15 else 'D' if per_100k>=5 else 'F',
                'vs_national_avg':   'above' if per_100k > 20 else 'below',
            })

        results.sort(key=lambda x: -x['chargers_per_100k'])
        national_total = sum(r['charger_count'] for r in results)
        national_pop   = sum(r['population'] for r in results)

        return api_response({
            'country':              country,
            'national_total':       national_total,
            'national_per_100k':    round(national_total / national_pop * 100_000, 1),
            'best_region':          results[0]['region'],
            'worst_region':         results[-1]['region'],
            'regions':              results,
            'data_note':            'Based on 8.5M collected data points across UK, US and EU',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── GOV 3. PLANNING APPLICATION SUPPORT ─────────────────────
@app.route('/api/v1/gov/planning')
@require_api_key
def api_gov_planning():
    try:
        import pandas as pd
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        radius  = float(request.args.get('radius', 2))
        country = request.args.get('country', 'UK').upper()

        df  = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        deg = radius * 0.014
        nearby = df[
            (abs(df['lat'] - lat) < deg) &
            (abs(df['lon'] - lon) < deg)
        ]
        unique = nearby[['lat','lon','operator','capacity','location_type']].drop_duplicates(['lat','lon'])
        gap_score = max(0, 100 - len(unique) * 4)
        grade = 'A' if gap_score<=20 else 'B' if gap_score<=40 else 'C' if gap_score<=60 else 'D' if gap_score<=80 else 'F'

        return api_response({
            'planning_report': {
                'location':           {'lat': lat, 'lon': lon},
                'search_radius_miles': radius,
                'existing_chargers':   len(unique),
                'coverage_grade':      grade,
                'infrastructure_gap':  gap_score,
                'networks_present':    list(nearby['operator'].unique()[:10]) if len(nearby) > 0 else [],
                'avg_capacity':        round(float(nearby['capacity'].mean()), 1) if len(nearby) > 0 else 0,
                'location_types':      nearby['location_type'].value_counts().to_dict() if len(nearby) > 0 else {},
                'planning_recommendation': (
                    'Area is well served — new installation may face viability questions'
                    if gap_score <= 20 else
                    'Moderate provision — new chargers would be supported by demand data'
                    if gap_score <= 50 else
                    'Significant gap identified — strong case for new EV infrastructure'
                ),
                'policy_references':   [
                    'UK EV Infrastructure Strategy 2023',
                    'NPPF Paragraph 105 — Transport',
                    'PAS 2080 Carbon Management in Infrastructure',
                ],
                'generated':           datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'data_source':         'ChargeSmart — 8.5M data points across UK, US & EU',
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── GOV 4. NET ZERO PROGRESS ─────────────────────────────────
@app.route('/api/v1/gov/netzero')
@require_api_key
def api_gov_netzero():
    try:
        import pandas as pd
        country = request.args.get('country', 'UK').upper()
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        total = len(df[['lat','lon']].drop_duplicates())
        # Estimate growth trend from our collection period
        uk_targets = {
            '2025': 300_000, '2030': 1_000_000, '2035': 2_500_000
        }
        current_year = datetime.datetime.now().year
        target_2030  = uk_targets.get('2030', 1_000_000)
        pct_to_target= round(total / target_2030 * 100, 1)

        return api_response({
            'country':          country,
            'current_chargers': total,
            'reporting_period': datetime.datetime.now().strftime('%B %Y'),
            'progress': {
                'target_2030':          target_2030,
                'pct_of_2030_target':   pct_to_target,
                'chargers_still_needed':max(0, target_2030 - total),
                'on_track':             pct_to_target >= (current_year - 2020) / (2030 - 2020) * 100,
            },
            'co2_impact': {
                'estimated_annual_ev_journeys': total * 365 * 3,
                'estimated_co2_saved_tonnes':   round(total * 365 * 3 * 20 * 0.335 / 1000, 0),
                'equivalent_trees':             round(total * 365 * 3 * 20 * 0.335 / 21.77, 0),
            },
            'suitable_for': ['Annual reports','Net zero commitments','Press releases','Cabinet reporting'],
            'data_source':  'ChargeSmart · chargesmart.online',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── GOV 5. RURAL VS URBAN DISPARITY ─────────────────────────
@app.route('/api/v1/gov/disparity')
@require_api_key
def api_gov_disparity():
    try:
        import pandas as pd
        country = request.args.get('country', 'UK').upper()
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Define urban cores by high charger density areas
        df2 = df[['lat','lon']].drop_duplicates().copy()
        df2['grid_lat'] = (df2['lat'] / 0.1).round() * 0.1
        df2['grid_lon'] = (df2['lon'] / 0.1).round() * 0.1
        density = df2.groupby(['grid_lat','grid_lon']).size().reset_index(name='count')
        median_density = float(density['count'].median())

        urban_cells = density[density['count'] > median_density * 2]
        rural_cells  = density[density['count'] <= median_density]

        urban_total = int(urban_cells['count'].sum())
        rural_total = int(rural_cells['count'].sum())
        ratio = round(urban_total / max(rural_total, 1), 1)

        return api_response({
            'country': country,
            'urban': {
                'charger_count':    urban_total,
                'area_zones':       len(urban_cells),
                'avg_per_zone':     round(urban_total / max(len(urban_cells),1), 1),
                'coverage_grade':   'A',
            },
            'rural': {
                'charger_count':    rural_total,
                'area_zones':       len(rural_cells),
                'avg_per_zone':     round(rural_total / max(len(rural_cells),1), 1),
                'coverage_grade':   'D' if ratio > 5 else 'C',
            },
            'urban_to_rural_ratio': ratio,
            'disparity_level':  'Severe' if ratio > 10 else 'High' if ratio > 5 else 'Moderate' if ratio > 2 else 'Low',
            'policy_implication': f'Urban areas have {ratio}x more chargers per zone than rural areas. Rural EV drivers face significant range anxiety barriers.',
            'recommended_intervention': 'Rural rapid charging hubs at key A-road junctions',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── BIZ 6. SITE SUITABILITY SCORE ───────────────────────────

# ═══════════════════════════════════════════════════════════════
# NEW ENDPOINTS - GOV + BIZ
# ═══════════════════════════════════════════════════════════════

# ── GOV: EV READINESS SCORE ──────────────────────────────────
@app.route('/api/v1/gov/ev-readiness')
@require_api_key
def api_gov_ev_readiness():
    """Score a country/region on EV infrastructure readiness"""
    try:
        country = request.args.get('country', 'KE').upper()

        # Readiness data per country - renewable energy %, policy score, grid score, import tariff
        READINESS_DATA = {
            'KE': {'name':'Kenya',        'renewable_pct':90,  'policy_score':72, 'grid_score':58, 'import_tariff':25,  'ev_fleet_pct':0.3,  'highlight':'90% renewable grid — strongest EV foundation in Africa'},
            'ET': {'name':'Ethiopia',     'renewable_pct':98,  'policy_score':85, 'grid_score':45, 'import_tariff':0,   'ev_fleet_pct':0.8,  'highlight':'First country to ban petrol vehicle imports (2024)'},
            'RW': {'name':'Rwanda',       'renewable_pct':55,  'policy_score':80, 'grid_score':62, 'import_tariff':10,  'ev_fleet_pct':0.4,  'highlight':'Strong government EV policy, Kigali urban density ideal'},
            'ZA': {'name':'South Africa', 'renewable_pct':12,  'policy_score':45, 'grid_score':40, 'import_tariff':18,  'ev_fleet_pct':0.2,  'highlight':'Largest EV market in Africa but grid reliability a concern'},
            'NG': {'name':'Nigeria',      'renewable_pct':18,  'policy_score':35, 'grid_score':28, 'import_tariff':35,  'ev_fleet_pct':0.05, 'highlight':'Huge population opportunity but grid investment needed'},
            'MA': {'name':'Morocco',      'renewable_pct':42,  'policy_score':68, 'grid_score':72, 'import_tariff':20,  'ev_fleet_pct':0.3,  'highlight':'Best grid reliability in North Africa, EU proximity'},
            'AE': {'name':'UAE',          'renewable_pct':8,   'policy_score':88, 'grid_score':95, 'import_tariff':5,   'ev_fleet_pct':2.1,  'highlight':'World-class grid, strong government EV mandate'},
            'SA': {'name':'Saudi Arabia', 'renewable_pct':4,   'policy_score':70, 'grid_score':88, 'import_tariff':5,   'ev_fleet_pct':0.8,  'highlight':'Vision 2030 driving rapid EV adoption'},
            'IN': {'name':'India',        'renewable_pct':33,  'policy_score':75, 'grid_score':65, 'import_tariff':100, 'ev_fleet_pct':1.5,  'highlight':'Fastest growing EV market globally, FAME II policy'},
            'BR': {'name':'Brazil',       'renewable_pct':83,  'policy_score':60, 'grid_score':70, 'import_tariff':18,  'ev_fleet_pct':1.1,  'highlight':'83% renewable energy, strong ethanol-to-EV transition'},
            'UK': {'name':'UK',           'renewable_pct':40,  'policy_score':90, 'grid_score':95, 'import_tariff':0,   'ev_fleet_pct':16.0, 'highlight':'2035 petrol ban, £1.6B LEVI fund committed'},
            'US': {'name':'USA',          'renewable_pct':22,  'policy_score':78, 'grid_score':88, 'import_tariff':0,   'ev_fleet_pct':7.6,  'highlight':'$7.5B federal charging network investment'},
        }

        data = READINESS_DATA.get(country, {
            'name': country, 'renewable_pct': 20, 'policy_score': 40,
            'grid_score': 50, 'import_tariff': 20, 'ev_fleet_pct': 0.1,
            'highlight': 'Emerging EV market'
        })

        # Calculate overall readiness score (0-100)
        readiness_score = round(
            data['renewable_pct'] * 0.30 +
            data['policy_score']  * 0.30 +
            data['grid_score']    * 0.25 +
            max(0, 100 - data['import_tariff'] * 2) * 0.15
        , 1)

        # Tier classification
        if readiness_score >= 75:   tier = 'HIGH'
        elif readiness_score >= 50: tier = 'MEDIUM'
        else:                       tier = 'LOW'

        # Barriers and opportunities
        barriers = []
        opportunities = []
        if data['import_tariff'] > 20:  barriers.append(f"High EV import tariff ({data['import_tariff']}%)")
        if data['grid_score'] < 50:     barriers.append("Grid reliability needs improvement")
        if data['renewable_pct'] < 20:  barriers.append("High carbon grid reduces EV green credentials")
        if data['renewable_pct'] > 60:  opportunities.append(f"{data['renewable_pct']}% renewable energy = truly green EVs")
        if data['policy_score'] > 70:   opportunities.append("Strong government EV policy framework")
        if data['ev_fleet_pct'] > 1:    opportunities.append(f"EV fleet already at {data['ev_fleet_pct']}% — momentum building")

        return jsonify({
            'country':          country,
            'country_name':     data['name'],
            'readiness_score':  readiness_score,
            'tier':             tier,
            'components': {
                'renewable_energy_pct': data['renewable_pct'],
                'policy_score':         data['policy_score'],
                'grid_reliability':     data['grid_score'],
                'import_tariff_pct':    data['import_tariff'],
                'ev_fleet_pct':         data['ev_fleet_pct'],
            },
            'highlight':        data['highlight'],
            'barriers':         barriers,
            'opportunities':    opportunities,
            'recommendation':   f"{'Priority market for EV infrastructure investment' if tier == 'HIGH' else 'Targeted investment with policy support needed' if tier == 'MEDIUM' else 'Foundation work required before mass deployment'}",
            'data_source':      'ChargeSmart EV Readiness Index 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── GOV: CORRIDOR MAPPING ─────────────────────────────────────
@app.route('/api/v1/gov/corridor-mapping')
@require_api_key
def api_gov_corridor_mapping():
    """Map key highway corridors and identify charging gaps"""
    try:
        country  = request.args.get('country', 'KE').upper()
        corridor = request.args.get('corridor', '')

        CORRIDORS = {
            'KE': [
                {'name': 'Nairobi → Mombasa', 'distance_km': 480, 'est_drive_hrs': 6.5,
                 'chargers_existing': 4, 'chargers_needed': 12, 'gap_score': 94,
                 'key_stops': ['Nairobi','Machakos','Mtito Andei','Voi','Mombasa'],
                 'notes': 'Primary trade corridor. 4 existing chargers all in Nairobi. 476km unserviced.'},
                {'name': 'Nairobi → Nakuru', 'distance_km': 156, 'est_drive_hrs': 2.2,
                 'chargers_existing': 2, 'chargers_needed': 4, 'gap_score': 72,
                 'key_stops': ['Nairobi','Naivasha','Nakuru'],
                 'notes': 'High traffic tourist route to Rift Valley.'},
                {'name': 'Nairobi → Kisumu', 'distance_km': 350, 'est_drive_hrs': 5.0,
                 'chargers_existing': 1, 'chargers_needed': 9, 'gap_score': 91,
                 'key_stops': ['Nairobi','Nakuru','Kericho','Kisumu'],
                 'notes': 'Western Kenya economic corridor. Almost no charging infrastructure.'},
                {'name': 'Mombasa → Malindi', 'distance_km': 120, 'est_drive_hrs': 1.8,
                 'chargers_existing': 0, 'chargers_needed': 3, 'gap_score': 100,
                 'key_stops': ['Mombasa','Kilifi','Malindi'],
                 'notes': 'Coastal tourist corridor. Zero EV infrastructure.'},
            ],
            'ET': [
                {'name': 'Addis Ababa → Adama', 'distance_km': 100, 'est_drive_hrs': 1.5,
                 'chargers_existing': 3, 'chargers_needed': 4, 'gap_score': 35,
                 'key_stops': ['Addis Ababa','Dukem','Adama'],
                 'notes': 'Best served corridor in Ethiopia. Industrial zone route.'},
                {'name': 'Addis Ababa → Dire Dawa', 'distance_km': 515, 'est_drive_hrs': 7.0,
                 'chargers_existing': 1, 'chargers_needed': 13, 'gap_score': 96,
                 'key_stops': ['Addis Ababa','Adama','Awash','Harar','Dire Dawa'],
                 'notes': 'Critical trade route to Djibouti port. Almost no charging.'},
            ],
            'ZA': [
                {'name': 'Johannesburg → Cape Town', 'distance_km': 1400, 'est_drive_hrs': 14.0,
                 'chargers_existing': 18, 'chargers_needed': 35, 'gap_score': 58,
                 'key_stops': ['Johannesburg','Bloemfontein','Beaufort West','Cape Town'],
                 'notes': "Africa's most developed EV corridor but still significant gaps."},
            ],
            'AE': [
                {'name': 'Dubai → Abu Dhabi', 'distance_km': 140, 'est_drive_hrs': 1.5,
                 'chargers_existing': 28, 'chargers_needed': 30, 'gap_score': 8,
                 'key_stops': ['Dubai','Jebel Ali','Abu Dhabi'],
                 'notes': 'Well served. Near saturation point for current EV adoption.'},
            ],
            'UK': [
                {'name': 'London → Edinburgh (A1/M1)', 'distance_km': 660, 'est_drive_hrs': 7.0,
                 'chargers_existing': 45, 'chargers_needed': 48, 'gap_score': 22,
                 'key_stops': ['London','Peterborough','Leeds','Newcastle','Edinburgh'],
                 'notes': 'Mostly served but rural gaps between Leeds and Newcastle.'},
            ],
        }

        country_corridors = CORRIDORS.get(country, [])

        if corridor:
            country_corridors = [c for c in country_corridors
                                 if corridor.lower() in c['name'].lower()]

        if not country_corridors:
            return jsonify({'error': f'No corridor data for country {country}'}), 404

        # Summary stats
        total_chargers_existing = sum(c['chargers_existing'] for c in country_corridors)
        total_chargers_needed   = sum(c['chargers_needed']   for c in country_corridors)
        avg_gap_score           = round(sum(c['gap_score'] for c in country_corridors) / len(country_corridors), 1)

        return jsonify({
            'country':                  country,
            'corridors':                country_corridors,
            'summary': {
                'total_corridors':          len(country_corridors),
                'total_km_mapped':          sum(c['distance_km'] for c in country_corridors),
                'chargers_existing':        total_chargers_existing,
                'chargers_needed':          total_chargers_needed,
                'charger_deficit':          total_chargers_needed - total_chargers_existing,
                'avg_gap_score':            avg_gap_score,
                'most_critical_corridor':   max(country_corridors, key=lambda x: x['gap_score'])['name'],
            },
            'data_source': 'ChargeSmart Corridor Intelligence 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── GOV: FUNDING ROI ─────────────────────────────────────────
@app.route('/api/v1/gov/funding-roi')
@require_api_key
def api_gov_funding_roi():
    """Given investment amount, return projected chargers, coverage gain, carbon impact"""
    try:
        budget_usd   = float(request.args.get('budget_usd', 1000000))
        country      = request.args.get('country', 'KE').upper()
        charger_type = request.args.get('charger_type', 'ac22').lower()  # ac7, ac22, dc50, dc150

        # Cost per charger installed (USD) by type and country context
        CHARGER_COSTS = {
            'ac7':   {'hardware': 2000,  'install': 1500,  'kw': 7,   'label': 'AC 7kW  (slow)'},
            'ac22':  {'hardware': 5000,  'install': 3000,  'kw': 22,  'label': 'AC 22kW (fast)'},
            'dc50':  {'hardware': 25000, 'install': 15000, 'kw': 50,  'label': 'DC 50kW (rapid)'},
            'dc150': {'hardware': 60000, 'install': 25000, 'kw': 150, 'label': 'DC 150kW (ultra-rapid)'},
        }

        # Country cost multipliers
        COUNTRY_MULTIPLIERS = {
            'KE': 0.75, 'ET': 0.70, 'RW': 0.80, 'ZA': 0.85,
            'NG': 0.72, 'AE': 1.10, 'SA': 1.05, 'IN': 0.60,
            'UK': 1.20, 'US': 1.15, 'DE': 1.10, 'FR': 1.05,
        }

        costs      = CHARGER_COSTS.get(charger_type, CHARGER_COSTS['ac22'])
        multiplier = COUNTRY_MULTIPLIERS.get(country, 0.85)
        cost_per_charger = (costs['hardware'] + costs['install']) * multiplier

        chargers_deployable  = int(budget_usd / cost_per_charger)
        # Coverage: each charger serves ~15km radius in urban, ~30km rural
        coverage_km2         = chargers_deployable * 500   # avg
        vehicles_served_daily= chargers_deployable * 18    # avg sessions/day
        carbon_saved_tonnes_yr= chargers_deployable * costs['kw'] * 0.8 * 365 * 0.00021  # kg CO2/kWh grid saving

        # Revenue potential (if monetised)
        revenue_yr = chargers_deployable * costs['kw'] * 0.6 * 365 * 0.15  # 60% utilisation, $0.15/kWh

        # Payback period
        payback_years = round(budget_usd / revenue_yr, 1) if revenue_yr > 0 else None

        # Job creation estimate
        jobs_construction = chargers_deployable * 2
        jobs_ongoing      = max(1, chargers_deployable // 10)

        return jsonify({
            'country':              country,
            'budget_usd':           budget_usd,
            'charger_type':         costs['label'],
            'cost_per_charger_usd': round(cost_per_charger),
            'deployment': {
                'chargers_deployable':      chargers_deployable,
                'coverage_area_km2':        coverage_km2,
                'vehicles_served_daily':    vehicles_served_daily,
                'carbon_saved_tonnes_yr':   round(carbon_saved_tonnes_yr, 1),
            },
            'economic': {
                'revenue_potential_usd_yr': round(revenue_yr),
                'payback_period_years':     payback_years,
                'jobs_created_construction':jobs_construction,
                'jobs_created_ongoing':     jobs_ongoing,
            },
            'recommendation': (
                f"A ${budget_usd:,.0f} investment in {country} deploys {chargers_deployable} "
                f"{costs['label']} chargers, serving ~{vehicles_served_daily:,} vehicles/day "
                f"and saving {round(carbon_saved_tonnes_yr, 1)} tonnes CO2/year."
            ),
            'funding_sources': [
                'World Bank ESMAP Clean Energy Fund',
                'African Development Bank (AfDB) GET-FiT',
                'Green Climate Fund (GCF)',
                'USAID Power Africa',
            ] if country in ['KE','ET','RW','ZA','NG','GH','TZ'] else [
                'European Investment Bank',
                'UK LEVI Fund',
                'IFC Emerging Markets',
            ],
            'data_source': 'ChargeSmart Funding Intelligence 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── BIZ: ROI CALCULATOR ──────────────────────────────────────
@app.route('/api/v1/biz/roi-calculator')
@require_api_key
def api_biz_roi_calculator():
    """Returns payback period and projected revenue for a charging installation"""
    try:
        lat            = float(request.args.get('lat', 51.5))
        lon            = float(request.args.get('lon', -0.1))
        charger_type   = request.args.get('charger_type', 'ac22').lower()
        num_chargers   = int(request.args.get('num_chargers', 2))
        electricity_rate = float(request.args.get('electricity_rate', 0.28))  # £/kWh
        charge_rate    = float(request.args.get('charge_rate', 0.65))         # £/kWh charged to customer

        CHARGER_SPECS = {
            'ac7':   {'kw': 7,   'install_gbp': 3000,  'hardware_gbp': 1500,  'label': 'AC 7kW'},
            'ac22':  {'kw': 22,  'install_gbp': 6000,  'hardware_gbp': 4000,  'label': 'AC 22kW'},
            'dc50':  {'kw': 50,  'install_gbp': 18000, 'hardware_gbp': 22000, 'label': 'DC 50kW'},
            'dc150': {'kw': 150, 'install_gbp': 40000, 'hardware_gbp': 55000, 'label': 'DC 150kW'},
        }

        spec = CHARGER_SPECS.get(charger_type, CHARGER_SPECS['ac22'])

        # Use nearby data to estimate demand
        demand_factor = 0.45  # baseline 45% utilisation
        df = load_charger_data()
        if df is not None and len(df) > 0:
            import math
            nearby = df[
                (abs(df['lat'] - lat) < 0.1) &
                (abs(df['lon'] - lon) < 0.1)
            ]
            competitor_count = len(nearby['id'].unique()) if 'id' in df.columns else len(nearby)
            if competitor_count == 0:    demand_factor = 0.65  # desert = high demand
            elif competitor_count < 3:   demand_factor = 0.55
            elif competitor_count > 10:  demand_factor = 0.35  # saturated
        else:
            competitor_count = 0

        # Financial model
        total_capex       = (spec['install_gbp'] + spec['hardware_gbp']) * num_chargers
        hours_per_day     = 24 * demand_factor
        kwh_per_day       = spec['kw'] * hours_per_day * num_chargers
        revenue_per_day   = kwh_per_day * charge_rate
        cost_per_day      = kwh_per_day * electricity_rate
        profit_per_day    = revenue_per_day - cost_per_day
        revenue_per_year  = revenue_per_day * 365
        profit_per_year   = profit_per_day * 365
        payback_years     = round(total_capex / profit_per_year, 1) if profit_per_year > 0 else None

        # Grant eligibility (UK)
        ozev_grant        = min(350 * num_chargers, 2500)  # OZEV workplace grant
        net_capex         = total_capex - ozev_grant
        payback_with_grant= round(net_capex / profit_per_year, 1) if profit_per_year > 0 else None

        return jsonify({
            'location':         {'lat': lat, 'lon': lon},
            'charger_type':     spec['label'],
            'num_chargers':     num_chargers,
            'demand': {
                'utilisation_pct':      round(demand_factor * 100, 1),
                'nearby_competitors':   competitor_count,
                'sessions_per_day':     round(hours_per_day * num_chargers, 1),
                'kwh_delivered_per_day':round(kwh_per_day, 1),
            },
            'financials': {
                'total_capex_gbp':          round(total_capex),
                'ozev_grant_available_gbp': ozev_grant,
                'net_capex_after_grant_gbp':round(net_capex),
                'revenue_per_year_gbp':     round(revenue_per_year),
                'profit_per_year_gbp':      round(profit_per_year),
                'payback_years':            payback_years,
                'payback_years_with_grant': payback_with_grant,
                'roi_5yr_pct':              round(((profit_per_year * 5 - total_capex) / total_capex) * 100, 1),
            },
            'recommendation': (
                f"{num_chargers}x {spec['label']} chargers at this location should generate "
                f"£{round(profit_per_year):,}/year profit with a {payback_with_grant}-year payback "
                f"after OZEV grant."
            ),
            'data_source': 'ChargeSmart ROI Intelligence 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── FLEET: ROUTE COVERAGE ─────────────────────────────────────
@app.route('/api/v1/fleet/route-coverage')
@require_api_key
def api_fleet_route_coverage():
    """Given origin + destination, return charging availability along route"""
    try:
        orig_lat  = float(request.args.get('orig_lat', 51.5))
        orig_lon  = float(request.args.get('orig_lon', -0.1))
        dest_lat  = float(request.args.get('dest_lat', 52.5))
        dest_lon  = float(request.args.get('dest_lon', -1.9))
        range_km  = float(request.args.get('range_km', 300))
        vehicle   = request.args.get('vehicle', 'van')

        import math

        # Distance calculation
        def haversine(la1, lo1, la2, lo2):
            R = 6371
            dlat = math.radians(la2-la1)
            dlon = math.radians(lo2-lo1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
            return R * 2 * math.asin(math.sqrt(a))

        total_km = haversine(orig_lat, orig_lon, dest_lat, dest_lon)

        # Generate waypoints every 80km along route
        waypoints = []
        steps = max(1, int(total_km / 80))
        for i in range(steps + 1):
            t = i / steps
            wp_lat = orig_lat + (dest_lat - orig_lat) * t
            wp_lon = orig_lon + (dest_lon - orig_lon) * t
            waypoints.append({'lat': round(wp_lat,4), 'lon': round(wp_lon,4),
                              'km_from_start': round(total_km * t, 1)})

        # Check charger availability at each waypoint
        df = load_charger_data()
        legs = []
        for i, wp in enumerate(waypoints):
            if df is not None:
                nearby = df[
                    (abs(df['lat'] - wp['lat']) < 0.15) &
                    (abs(df['lon'] - wp['lon']) < 0.15)
                ]
                charger_count = len(nearby)
                has_rapid = (nearby['capacity'] >= 50).any() if len(nearby) > 0 and 'capacity' in nearby.columns else False
            else:
                charger_count = 0
                has_rapid = False

            status = 'GOOD' if charger_count >= 3 else 'LIMITED' if charger_count >= 1 else 'GAP'
            legs.append({
                'waypoint':      i + 1,
                'lat':           wp['lat'],
                'lon':           wp['lon'],
                'km_from_start': wp['km_from_start'],
                'chargers_nearby': charger_count,
                'has_rapid_charging': has_rapid,
                'status':        status,
            })

        gaps    = [l for l in legs if l['status'] == 'GAP']
        limited = [l for l in legs if l['status'] == 'LIMITED']
        good    = [l for l in legs if l['status'] == 'GOOD']

        # Can vehicle complete route?
        charges_needed = math.ceil(total_km / range_km)
        route_viable   = len(gaps) == 0 or (charges_needed <= len(good) + len(limited))

        return jsonify({
            'route': {
                'origin':      {'lat': orig_lat, 'lon': orig_lon},
                'destination': {'lat': dest_lat, 'lon': dest_lon},
                'total_km':    round(total_km, 1),
                'vehicle':     vehicle,
                'range_km':    range_km,
            },
            'legs':            legs,
            'summary': {
                'waypoints_checked':    len(legs),
                'good_coverage':        len(good),
                'limited_coverage':     len(limited),
                'charging_gaps':        len(gaps),
                'charges_needed':       charges_needed,
                'route_viable':         route_viable,
                'coverage_score':       round((len(good) / len(legs)) * 100, 1) if legs else 0,
            },
            'recommendation': (
                'Route is fully serviceable.' if route_viable
                else f'Route has {len(gaps)} charging gap(s). Consider carrying portable charger or planning overnight stop.'
            ),
            'data_source': 'ChargeSmart Fleet Intelligence 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── FLEET: ENERGY BUDGET ──────────────────────────────────────
@app.route('/api/v1/fleet/energy-budget')
@require_api_key
def api_fleet_energy_budget():
    """Calculate total energy cost for a fleet route"""
    try:
        distance_km      = float(request.args.get('distance_km', 200))
        num_vehicles     = int(request.args.get('num_vehicles', 10))
        vehicle_type     = request.args.get('vehicle_type', 'van').lower()
        trips_per_month  = int(request.args.get('trips_per_month', 20))
        electricity_rate = float(request.args.get('electricity_rate', 0.28))  # £/kWh

        # kWh per 100km by vehicle type
        CONSUMPTION = {
            'car':       15,   # e.g. Tesla Model 3
            'van':       28,   # e.g. Ford E-Transit
            'truck':     80,   # e.g. Volta Zero
            'bus':       120,  # e.g. BYD K9
            'motorbike': 8,
        }

        consumption = CONSUMPTION.get(vehicle_type, 28)
        kwh_per_trip_per_vehicle = (distance_km / 100) * consumption

        # Monthly costs
        kwh_per_month        = kwh_per_trip_per_vehicle * num_vehicles * trips_per_month
        cost_per_month       = kwh_per_month * electricity_rate
        cost_per_trip        = kwh_per_trip_per_vehicle * electricity_rate

        # Peak vs off-peak savings
        peak_rate            = electricity_rate * 1.4
        offpeak_rate         = electricity_rate * 0.6
        cost_peak_month      = kwh_per_month * peak_rate
        cost_offpeak_month   = kwh_per_month * offpeak_rate
        potential_saving     = cost_peak_month - cost_offpeak_month

        # vs diesel equivalent
        diesel_litres_month  = (distance_km / 100) * 8 * num_vehicles * trips_per_month  # 8L/100km
        diesel_cost_month    = diesel_litres_month * 1.55  # £/litre
        ev_saving_vs_diesel  = diesel_cost_month - cost_per_month

        return jsonify({
            'fleet': {
                'vehicle_type':      vehicle_type,
                'num_vehicles':      num_vehicles,
                'distance_km':       distance_km,
                'trips_per_month':   trips_per_month,
                'consumption_kwh_100km': consumption,
            },
            'energy': {
                'kwh_per_trip_per_vehicle': round(kwh_per_trip_per_vehicle, 1),
                'total_kwh_per_month':      round(kwh_per_month, 1),
            },
            'costs': {
                'electricity_rate_gbp_kwh':  electricity_rate,
                'cost_per_trip_gbp':         round(cost_per_trip, 2),
                'cost_per_month_gbp':         round(cost_per_month, 2),
                'cost_per_year_gbp':          round(cost_per_month * 12, 2),
                'if_charged_peak_gbp_month':  round(cost_peak_month, 2),
                'if_charged_offpeak_gbp_month':round(cost_offpeak_month, 2),
                'saving_by_offpeak_gbp_month': round(potential_saving, 2),
            },
            'vs_diesel': {
                'diesel_cost_per_month_gbp':  round(diesel_cost_month, 2),
                'ev_saving_per_month_gbp':    round(ev_saving_vs_diesel, 2),
                'ev_saving_per_year_gbp':     round(ev_saving_vs_diesel * 12, 2),
                'saving_pct':                 round((ev_saving_vs_diesel / diesel_cost_month) * 100, 1) if diesel_cost_month > 0 else 0,
            },
            'recommendation': (
                f"Fleet of {num_vehicles} {vehicle_type}s costs £{round(cost_per_month):,}/month to charge. "
                f"Switching to off-peak charging saves £{round(potential_saving):,}/month. "
                f"Saves £{round(ev_saving_vs_diesel):,}/month vs diesel."
            ),
            'data_source': 'ChargeSmart Fleet Intelligence 2026',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/biz/site-score')
@require_api_key
def api_biz_site_score():
    try:
        import pandas as pd
        lat        = float(request.args.get('lat', 51.5))
        lon        = float(request.args.get('lon', -0.1))
        site_type  = request.args.get('site_type', 'retail')
        country    = request.args.get('country', 'UK').upper()

        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        deg1 = 0.014  # ~1 mile
        deg3 = 0.042  # ~3 miles
        nearby1 = df[(abs(df['lat']-lat)<deg1)&(abs(df['lon']-lon)<deg1)]
        nearby3 = df[(abs(df['lat']-lat)<deg3)&(abs(df['lon']-lon)<deg3)]
        c1 = len(nearby1[['lat','lon']].drop_duplicates())
        c3 = len(nearby3[['lat','lon']].drop_duplicates())

        # Score components
        demand_score     = min(c3 * 2, 30)         # nearby EV activity = demand signal
        gap_score        = max(0, 30 - c1 * 3)     # fewer nearby = bigger opportunity
        site_type_scores = {'motorway':35,'retail':30,'supermarket':28,'hotel':25,'office':20,'other':15}
        site_score_val   = site_type_scores.get(site_type, 15)
        total_score      = min(demand_score + gap_score + site_score_val, 100)

        roi_monthly = round(total_score * 4.5, 0)  # rough £ estimate

        return api_response({
            'lat': lat, 'lon': lon,
            'site_type':           site_type,
            'suitability_score':   total_score,
            'grade':               'A' if total_score>=80 else 'B' if total_score>=60 else 'C' if total_score>=40 else 'D',
            'verdict':             'Excellent site' if total_score>=80 else 'Good site' if total_score>=60 else 'Viable site' if total_score>=40 else 'Marginal site',
            'nearby_chargers_1mi': c1,
            'nearby_chargers_3mi': c3,
            'competition_level':   'Low' if c1<3 else 'Medium' if c1<8 else 'High',
            'estimated_monthly_revenue': f'£{int(roi_monthly):,}',
            'recommended_capacity':4 if total_score>=70 else 2,
            'recommended_type':    'DC Fast (50kW+)' if site_type in ['motorway','retail'] else 'AC (7-22kW)',
            'payback_years':       round(35000 / (roi_monthly * 12), 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── BIZ 7. COMPETITOR CHARGER AUDIT ─────────────────────────
@app.route('/api/v1/biz/competitor-audit')
@require_api_key
def api_biz_competitor_audit():
    try:
        import pandas as pd, pickle
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        radius  = float(request.args.get('radius', 5))
        country = request.args.get('country', 'UK').upper()

        df  = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        deg = radius * 0.014
        nearby = df[
            (abs(df['lat']-lat)<deg)&(abs(df['lon']-lon)<deg)
        ][['lat','lon','operator','capacity','location_type']].drop_duplicates(['lat','lon'])

        model = pickle.load(open(data_path('charger_model.pkl'),'rb')) if os.path.exists(data_path('charger_model.pkl')) else None
        country_map = {'UK':0,'US':1,'EU':2}
        loc_map     = {'motorway':3,'supermarket':2,'council':1,'tesla':2,'other':1}
        hour = datetime.datetime.now().hour
        day  = datetime.datetime.now().weekday()

        competitors = []
        for _, row in nearby.iterrows():
            prob_free = 50.0
            if model:
                fv = [[hour, day, 1 if day>=5 else 0,
                       min(int(row['capacity']),20),
                       row['lat'], row['lon'],
                       country_map.get(country,0),
                       loc_map.get(str(row['location_type']),1)]]
                prob_free = round(float(model.predict_proba(fv)[0][0])*100, 1)
            competitors.append({
                'lat':              round(float(row['lat']),5),
                'lon':              round(float(row['lon']),5),
                'operator':         str(row['operator']),
                'capacity':         int(row['capacity']),
                'location_type':    str(row['location_type']),
                'current_availability': prob_free,
                'threat_level':     'High' if prob_free<40 else 'Medium' if prob_free<65 else 'Low',
            })

        competitors.sort(key=lambda x: x['current_availability'])
        op_summary = {}
        for c in competitors:
            op = c['operator']
            if op not in op_summary:
                op_summary[op] = {'count':0,'total_capacity':0}
            op_summary[op]['count'] += 1
            op_summary[op]['total_capacity'] += c['capacity']

        return api_response({
            'lat': lat, 'lon': lon,
            'radius_miles':        radius,
            'total_competitors':   len(competitors),
            'high_threat':         sum(1 for c in competitors if c['threat_level']=='High'),
            'operator_summary':    op_summary,
            'busiest_competitor':  competitors[0] if competitors else None,
            'quietest_competitor': competitors[-1] if competitors else None,
            'market_opportunity':  'Strong' if len(competitors)<5 else 'Moderate' if len(competitors)<15 else 'Competitive',
            'competitors':         competitors[:30],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── BIZ 8. FLEET DEPOT OPTIMISATION ─────────────────────────
@app.route('/api/v1/biz/depot-optimise', methods=['POST'])
@require_api_key
def api_biz_depot_optimise():
    try:
        import pandas as pd
        data    = request.get_json() or {}
        depots  = data.get('depots', [])
        country = data.get('country', 'UK').upper()

        if not depots:
            return jsonify({'error': 'Provide depots array in request body'}), 400

        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        results = []
        for depot in depots:
            lat  = float(depot.get('lat', 51.5))
            lon  = float(depot.get('lon', -0.1))
            name = depot.get('name', f'{lat},{lon}')
            vehicles = int(depot.get('vehicles', 5))

            deg = 0.014
            if df is not None:
                nearby = df[(abs(df['lat']-lat)<deg)&(abs(df['lon']-lon)<deg)]
                existing = len(nearby[['lat','lon']].drop_duplicates())
            else:
                existing = 0

            needed = max(0, vehicles - existing)
            gap    = max(0, 100 - existing * 8)

            results.append({
                'depot_name':          name,
                'lat':                 lat,
                'lon':                 lon,
                'vehicles':            vehicles,
                'existing_chargers_nearby': existing,
                'chargers_needed':     needed,
                'gap_score':           gap,
                'priority':            'Critical' if gap>=80 else 'High' if gap>=60 else 'Medium' if gap>=40 else 'Low',
                'recommended_install': needed,
                'estimated_install_cost': f'£{needed * 8500:,}',
                'recommended_type':    'DC Fast 50kW' if vehicles>10 else 'AC 22kW',
            })

        results.sort(key=lambda x: -x['gap_score'])
        total_cost = sum(int(r['estimated_install_cost'].replace('£','').replace(',','')) for r in results)

        return api_response({
            'total_depots':         len(results),
            'depots_needing_action':sum(1 for r in results if r['priority'] in ['Critical','High']),
            'total_chargers_needed':sum(r['chargers_needed'] for r in results),
            'total_estimated_cost': f'£{total_cost:,}',
            'depots':               results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── BIZ 9. PROPERTY EV SCORE ─────────────────────────────────
@app.route('/api/v1/biz/property-score')
@require_api_key
def api_biz_property_score():
    try:
        import pandas as pd
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        country = request.args.get('country', 'UK').upper()

        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        def count_miles(m):
            d = m * 0.014
            return len(df[(abs(df['lat']-lat)<d)&(abs(df['lon']-lon)<d)][['lat','lon']].drop_duplicates())

        c0_5 = count_miles(0.5)
        c1   = count_miles(1)
        c3   = count_miles(3)

        score = min(c0_5*20 + c1*10 + c3*3, 100)
        grade = 'A' if score>=80 else 'B' if score>=60 else 'C' if score>=40 else 'D' if score>=20 else 'F'
        stars = 5 if score>=80 else 4 if score>=60 else 3 if score>=40 else 2 if score>=20 else 1

        return api_response({
            'lat': lat, 'lon': lon,
            'ev_score':              score,
            'grade':                 grade,
            'stars':                 stars,
            'label':                 f'EV Ready — Grade {grade}',
            'chargers_within_half_mile': c0_5,
            'chargers_within_1_mile':    c1,
            'chargers_within_3_miles':   c3,
            'verdict':               'Excellent EV charging access' if score>=80
                                else 'Good EV charging access' if score>=60
                                else 'Adequate EV charging nearby' if score>=40
                                else 'Limited EV charging — may affect EV owners',
            'selling_point':         f'{c1} public charger{"s" if c1!=1 else ""} within 1 mile',
            'embed_badge':           f'<div style="font-family:sans-serif;padding:8px 12px;background:#f0fff0;border:1px solid #00C851;border-radius:6px;font-size:13px;">⚡ EV Score: {grade} ({score}/100) · chargesmart.online</div>',
            'powered_by':            'chargesmart.online',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── BIZ 10. RETAIL DWELL TIME OPPORTUNITY ───────────────────
@app.route('/api/v1/biz/retail-opportunity')
@require_api_key
def api_biz_retail_opportunity():
    try:
        import pandas as pd
        lat       = float(request.args.get('lat', 51.5))
        lon       = float(request.args.get('lon', -0.1))
        site_type = request.args.get('site_type', 'retail')
        country   = request.args.get('country', 'UK').upper()

        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        deg3 = 0.042
        nearby = df[(abs(df['lat']-lat)<deg3)&(abs(df['lon']-lon)<deg3)]
        ev_activity = len(nearby)

        # Dwell time by site type (minutes)
        dwell_times = {'supermarket':45,'retail':60,'motorway':25,'hotel':480,'office':480,'other':40}
        dwell       = dwell_times.get(site_type, 40)

        # Revenue model
        charge_per_session = {'motorway':12,'supermarket':6,'retail':8,'hotel':15,'office':10,'other':7}
        revenue_per_session = charge_per_session.get(site_type, 8)
        sessions_per_day    = max(2, min(ev_activity // 50, 20))
        monthly_revenue     = sessions_per_day * 30 * revenue_per_session
        annual_revenue      = monthly_revenue * 12

        return api_response({
            'lat': lat, 'lon': lon,
            'site_type':              site_type,
            'ev_activity_score':      min(ev_activity // 10, 100),
            'estimated_daily_sessions': sessions_per_day,
            'avg_dwell_minutes':      dwell,
            'revenue_per_session':    f'£{revenue_per_session}',
            'estimated_monthly_revenue': f'£{monthly_revenue:,}',
            'estimated_annual_revenue':  f'£{annual_revenue:,}',
            'payback_period_years':   round(35000 / annual_revenue, 1),
            'opportunity_rating':     'Excellent' if annual_revenue>50000 else 'Good' if annual_revenue>20000 else 'Moderate' if annual_revenue>10000 else 'Low',
            'additional_benefits':    [
                f'{dwell} min dwell time increases in-store spend',
                'EV drivers have 40% higher average income than petrol drivers',
                'Charging amenity attracts repeat visits',
                'Sustainability credentials for ESG reporting',
            ],
            'recommended_capacity':   max(2, sessions_per_day // 3),
            'powered_by':             'chargesmart.online',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ═══════════════════════════════════════════════════════════════
# AUTHENTICATION — MAGIC LINK SYSTEM
# ═══════════════════════════════════════════════════════════════

from auth import (create_magic_token, verify_magic_token, create_session,
                  get_session_user, delete_session, get_or_create_user,
                  update_user, send_magic_link)
FROM_EMAIL       = os.environ.get('FROM_EMAIL', 'hello@chargesmart.online')

# send_magic_link is imported from auth.py (uses Brevo)

def get_current_user():
    """Get logged-in user from session cookie."""
    token = request.cookies.get('cs_session')
    if not token:
        return None
    return get_session_user(token)

# ── REQUEST MAGIC LINK ───────────────────────────────────────

# ── LOGIN PAGE ───────────────────────────────────────────────
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/auth/request', methods=['POST'])
def auth_request():
    try:
        data  = request.get_json() or {}
        email = data.get('email', '').lower().strip()
        name  = data.get('name', '')
        if not email or '@' not in email:
            return jsonify({'error': 'Valid email required'}), 400
        # Create user if first time
        get_or_create_user(email, name)
        token    = create_magic_token(email)
        base_url = request.host_url
        sent     = send_magic_link(email, token, base_url)
        return jsonify({
            'success': True,
            'message': f'Login link sent to {email}. Check your inbox — it expires in 15 minutes.',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── VERIFY MAGIC LINK ────────────────────────────────────────
@app.route('/auth/verify')
def auth_verify():
    try:
        token = request.args.get('token', '')
        valid, email, error = verify_magic_token(token)
        if not valid:
            return redirect(f'/login?error={error}')
        # Get or create user, then update last login
        user = get_or_create_user(email)
        uid  = user.get('uid', email)
        update_user(uid, {'last_login': datetime.datetime.now().isoformat()})
        session_token = create_session(uid)
        response = redirect('/')
        response.set_cookie(
            'cs_session', session_token,
            max_age=30*24*60*60,  # 30 days
            httponly=True,
            samesite='Lax',
            secure=request.is_secure
        )
        return response
    except Exception as e:
        return render_template('auth_error.html', error=str(e))

# ── LOGOUT ───────────────────────────────────────────────────
@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    token    = request.cookies.get('cs_session')
    response = jsonify({'success': True})
    if token:
        delete_session(token)
    response.delete_cookie('cs_session')
    return response

# ── ACCOUNT PAGE ─────────────────────────────────────────────
@app.route('/account')
def account_page():
    user = get_current_user()
    if not user:
        return redirect('/?login=1')
    return render_template('account.html', user=user)

# ── GET CURRENT USER (API) ───────────────────────────────────

# ── ACCOUNT API ──────────────────────────────────────────────
@app.route('/api/account')
def api_account():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'logged_in': False}), 401
        # Attach API key usage stats
        usage = {}
        if user.get('api_key'):
            try:
                from api_system import get_key_stats
                usage = get_key_stats(user['api_key']) or {}
            except Exception:
                usage = {}
        from api_system import TIER_LIMITS, TIER_PRICES
        tier = user.get('plan', 'free')
        return jsonify({
            'logged_in':    True,
            'uid':          user.get('uid'),
            'email':        user.get('email'),
            'name':         user.get('name', ''),
            'plan':         tier,
            'plan_price':   TIER_PRICES.get(tier, 0),
            'daily_limit':  TIER_LIMITS.get(tier, 100),
            'api_key':      user.get('api_key'),
            'favourites':   user.get('favourites', []),
            'created':      user.get('created', ''),
            'last_login':   user.get('last_login', ''),
            'login_count':  user.get('login_count', 0),
            'usage':        usage,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account/generate-key', methods=['POST'])
def api_generate_key():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Not logged in'}), 401
        from api_system import create_api_key, load_keys, save_keys
        result = create_api_key(user['email'], user.get('plan', 'free'))
        update_user(user['uid'], {'api_key': result['key']})
        return jsonify({'success': True, 'api_key': result['key'], 'tier': result['tier']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account/favourites', methods=['POST'])
def api_save_favourites():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Not logged in'}), 401
        data = request.get_json() or {}
        favourites = data.get('favourites', [])
        update_user(user['uid'], {'favourites': favourites})
        return jsonify({'success': True, 'count': len(favourites)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account/update', methods=['POST'])
def api_update_profile():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Not logged in'}), 401
        data = request.get_json() or {}
        updates = {}
        if 'name' in data:
            updates['name'] = data['name'][:60]
        update_user(user['uid'], updates)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

PLAN_LABELS = {
    'free':         'Free',
    'pro':          'Pro',
    'fleet':        'Fleet',
    'api_developer':'Developer',
    'api_business': 'Business',
    'enterprise':   'Enterprise',
}


@app.route('/auth/me')
def auth_me():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'logged_in': False})
        # Attach API key usage if they have one
        usage = None
        if user.get('api_key'):
            from api_system import get_key_stats
            usage = get_key_stats(user['api_key'])
        return jsonify({
            'logged_in':    True,
            'uid':          user.get('uid', ''),
            'email':        user['email'],
            'name':         user.get('name', ''),
            'plan':         user.get('plan', 'free'),
            'plan_label':   PLAN_LABELS.get(user.get('plan','free'), 'Free'),
            'api_key':      user.get('api_key'),
            'favourites':   user.get('favourites', []),
            'created':      user.get('created'),
            'last_login':   user.get('last_login'),
            'api_usage':    usage,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── SYNC FAVOURITES ──────────────────────────────────────────
@app.route('/auth/favourites', methods=['POST'])
def auth_sync_favourites():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Not logged in'}), 401
        data = request.get_json() or {}
        favs = data.get('favourites', [])
        update_user(user['uid'], {'favourites': favs})
        return jsonify({'success': True, 'favourites': favs})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── LINK API KEY TO ACCOUNT ──────────────────────────────────
@app.route('/auth/link-api-key', methods=['POST'])
def auth_link_api_key():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Not logged in'}), 401
        data    = request.get_json() or {}
        api_key = data.get('api_key', '').strip()
        from api_system import validate_key
        valid, tier, remaining, error = validate_key(api_key)
        if not valid:
            return jsonify({'error': 'Invalid API key'}), 400
        update_user(user['uid'], {'api_key': api_key, 'plan': tier})
        return jsonify({'success': True, 'tier': tier})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── GENERATE API KEY FOR ACCOUNT ─────────────────────────────
@app.route('/auth/generate-api-key', methods=['POST'])
def auth_generate_api_key():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Not logged in'}), 401
        from api_system import create_api_key, load_keys, save_keys
        result  = create_api_key(user['email'], 'free')
        api_key = result['key']
        update_user(user['uid'], {'api_key': api_key})
        return jsonify({'success': True, 'api_key': api_key, 'tier': result['tier']})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── UPDATE NAME ──────────────────────────────────────────────
@app.route('/auth/update-profile', methods=['POST'])
def auth_update_profile():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Not logged in'}), 401
        data = request.get_json() or {}
        name = data.get('name', '').strip()
        if name:
            update_user(user['uid'], {'name': name})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



# ══════════════════════════════════════════════════════════════
# BIZ — EV OPPORTUNITY FINDER  /api/v1/biz/opportunity-finder
# ══════════════════════════════════════════════════════════════
#
# Proactive site discovery for charging network operators.
# Given a country + optional city/region, returns ranked
# candidate zones with profitability scores, competition
# analysis, demand signals and ROI estimates.
#
# Factors scored:
#   - Population density proxy (urban centre distance)
#   - Existing charger gap (low competition = opportunity)
#   - EV demand signal (nearby charger usage density)
#   - Road network type (highway > urban > rural)
#   - Grid readiness (from global_data readiness score)
#   - Income / GDP proxy (willingness to pay)
#   - Renewable energy % (ESG investor appeal)
#   - Tourism index (transient demand)
#
# Tiers: Business+

URBAN_CENTRES = {
    # country_code: [(name, lat, lon, population_m, tourism_score, income_index)]
    'KE': [
        ('Nairobi CBD',        -1.286, 36.820, 4.5, 72, 65),
        ('Westlands',          -1.267, 36.811, 0.8, 60, 85),
        ('Karen/Langata',      -1.330, 36.745, 0.4, 55, 90),
        ('Thika Road',         -1.220, 36.870, 1.2, 40, 60),
        ('Mombasa City',       -4.050, 39.670, 1.2, 85, 60),
        ('Mombasa North Coast',-3.970, 39.720, 0.3, 90, 70),
        ('Kisumu City',        -0.090, 34.760, 0.6, 50, 50),
        ('Nakuru Town',        -0.304, 36.073, 0.5, 45, 55),
        ('Nairobi Airport',    -1.319, 36.927, 0.1, 80, 75),
        ('Gigiri/UN Area',     -1.237, 36.803, 0.2, 50, 95),
    ],
    'ET': [
        ('Addis Ababa Centre',  9.025, 38.747, 4.5, 65, 45),
        ('Bole Airport Area',   8.977, 38.799, 0.8, 75, 60),
        ('Adama/Nazret',        8.540, 39.270, 0.4, 40, 45),
        ('Hawassa City',        7.060, 38.476, 0.3, 50, 42),
        ('Bahir Dar',          11.594, 37.388, 0.2, 60, 40),
        ('Dire Dawa',           9.593, 41.866, 0.4, 35, 42),
    ],
    'RW': [
        ('Kigali City Centre',  -1.944, 30.061, 1.4, 72, 55),
        ('Kigali Airport',      -1.969, 30.135, 0.2, 65, 60),
        ('Musanze',             -1.499, 29.634, 0.1, 80, 45),
        ('Huye/Butare',         -2.597, 29.739, 0.1, 45, 42),
    ],
    'NG': [
        ('Lagos Island',        6.455,  3.394, 3.0, 65, 55),
        ('Victoria Island',     6.428,  3.421, 0.8, 70, 85),
        ('Lekki',               6.440,  3.530, 1.2, 60, 80),
        ('Abuja Central',       9.057,  7.491, 2.0, 60, 70),
        ('Port Harcourt',       4.815,  7.049, 1.0, 45, 65),
        ('Kano City',          12.000,  8.520, 1.5, 35, 45),
        ('Ikeja/Airport',       6.579,  3.321, 1.0, 55, 65),
    ],
    'ZA': [
        ('Sandton',            -26.107, 28.057, 0.8, 65, 92),
        ('Cape Town CBD',      -33.926, 18.424, 0.9, 90, 88),
        ('Waterfront CT',      -33.902, 18.421, 0.3, 92, 90),
        ('Johannesburg CBD',   -26.205, 28.042, 1.5, 55, 75),
        ('Durban Centre',      -29.858, 31.021, 0.8, 78, 72),
        ('Pretoria CBD',       -25.746, 28.188, 0.7, 55, 75),
        ('Midrand',            -25.997, 28.129, 0.5, 50, 82),
        ('OR Tambo Airport',   -26.133, 28.242, 0.2, 70, 80),
    ],
    'AE': [
        ('Dubai Downtown',      25.197, 55.274, 0.5, 95, 98),
        ('Dubai Mall Area',     25.198, 55.279, 0.3, 96, 98),
        ('Dubai Marina',        25.080, 55.140, 0.4, 88, 95),
        ('Abu Dhabi Corniche',  24.467, 54.357, 0.4, 82, 96),
        ('Sharjah Centre',      25.357, 55.391, 0.5, 65, 80),
        ('Dubai Airport',       25.253, 55.366, 0.1, 85, 92),
        ('Al Ain City',         24.207, 55.744, 0.3, 60, 85),
        ('JBR Beach',           25.077, 55.132, 0.2, 90, 92),
    ],
    'MA': [
        ('Casablanca Centre',   33.589, -7.604, 3.5, 70, 68),
        ('Rabat City',          33.994, -6.854, 0.8, 72, 72),
        ('Marrakech Medina',    31.629, -7.987, 0.9, 92, 65),
        ('Tangier Port',        35.779, -5.804, 0.5, 78, 62),
        ('Agadir Beach',        30.420, -9.598, 0.3, 88, 62),
        ('Fes Old Town',        34.054, -4.998, 0.4, 82, 58),
    ],
    'IN': [
        ('Mumbai BKC',          19.069, 72.870, 2.0, 72, 75),
        ('Bangalore Whitefield', 12.978, 77.748, 1.5, 60, 82),
        ('Delhi Connaught Pl',  28.632, 77.220, 2.0, 70, 78),
        ('Pune Koregaon',       18.537, 73.894, 0.8, 58, 72),
        ('Hyderabad HITEC',     17.450, 78.382, 0.9, 60, 75),
        ('Chennai Anna Nagar',  13.086, 80.210, 0.8, 58, 65),
        ('Gurgaon Cyber City',  28.494, 77.089, 0.7, 55, 85),
    ],
    'BR': [
        ('São Paulo Paulista',  -23.562, -46.655, 3.0, 72, 78),
        ('Rio Ipanema',         -22.985, -43.198, 1.0, 92, 82),
        ('Rio Barra',           -23.000, -43.365, 0.8, 85, 85),
        ('Curitiba Centre',     -25.430, -49.271, 0.8, 65, 72),
        ('Brasilia Asa Sul',    -15.793, -47.882, 0.6, 60, 80),
        ('Belo Horizonte',      -19.917, -43.934, 0.9, 60, 68),
    ],
    'GB': [
        ('London City',          51.514, -0.072, 2.0, 90, 95),
        ('London Canary Wharf',  51.505,  0.020, 0.5, 80, 95),
        ('Birmingham Centre',    52.480, -1.898, 1.1, 70, 78),
        ('Manchester Spinningfields', 53.480, -2.245, 0.8, 75, 82),
        ('Edinburgh Royal Mile', 55.950, -3.190, 0.4, 88, 82),
        ('Bristol Harbourside',  51.450, -2.597, 0.4, 78, 80),
        ('Leeds Centre',         53.800, -1.549, 0.5, 68, 75),
        ('Glasgow Centre',       55.861, -4.251, 0.5, 72, 72),
    ],
    'US': [
        ('NYC Midtown',          40.754, -73.984, 2.0, 92, 95),
        ('LA Westside',          34.024,-118.496, 1.5, 88, 92),
        ('Chicago Loop',         41.882, -87.629, 1.2, 82, 88),
        ('Houston Galleria',     29.739, -95.462, 0.8, 68, 85),
        ('Miami Brickell',       25.765, -80.195, 0.7, 88, 88),
        ('SF Financial Dist',    37.794,-122.399, 0.8, 85, 95),
        ('Seattle South Lake Union', 47.625,-122.336, 0.6, 80, 92),
        ('Austin Domain',        30.400, -97.722, 0.5, 78, 88),
    ],
}

# Default fallback centres for countries not in URBAN_CENTRES
DEFAULT_CENTRES = {
    'TH': [('Bangkok Sukhumvit', 13.741, 100.560, 2.0, 88, 72),
            ('Chiang Mai Old City', 18.787, 98.993, 0.3, 85, 58),
            ('Pattaya Beach', 12.927, 100.877, 0.2, 82, 62)],
    'VN': [('Ho Chi Minh D1', 10.776, 106.701, 2.5, 82, 55),
            ('Hanoi Hoan Kiem', 21.028, 105.852, 1.5, 78, 52),
            ('Da Nang Beach', 16.067, 108.221, 0.4, 85, 50)],
    'ID': [('Jakarta Sudirman', -6.208, 106.845, 3.0, 70, 58),
            ('Bali Seminyak', -8.693, 115.165, 0.3, 92, 62),
            ('Surabaya Centre', -7.250, 112.750, 1.0, 55, 55)],
    'CL': [('Santiago Las Condes', -33.408, -70.580, 1.2, 72, 78),
            ('Valparaíso Port', -33.047, -71.612, 0.3, 75, 65),
            ('Viña del Mar', -33.024, -71.552, 0.2, 80, 68)],
    'CO': [('Bogotá Chapinero', 4.651, -74.058, 1.5, 72, 62),
            ('Medellín El Poblado', 6.209, -75.569, 0.8, 80, 65),
            ('Cartagena Old City', 10.425, -75.550, 0.2, 88, 60)],
    'SA': [('Riyadh KAFD', 24.764, 46.625, 1.5, 65, 90),
            ('Jeddah Corniche', 21.540, 39.175, 0.8, 72, 85),
            ('NEOM', 28.000, 35.200, 0.1, 60, 95)],
    'EG': [('Cairo Maadi', 29.959, 31.250, 1.5, 72, 58),
            ('Cairo Zamalek', 30.061, 31.220, 0.4, 68, 65),
            ('Alexandria Corniche', 31.200, 29.920, 0.6, 65, 55),
            ('Hurghada Resort', 27.258, 33.812, 0.2, 88, 55)],
    'RU': [('Moscow City', 55.749, 37.621, 3.0, 72, 72),
            ('St Petersburg Centre', 59.938, 30.316, 0.8, 82, 68)],
    'TR': [('Istanbul Levent', 41.081, 29.011, 1.5, 85, 68),
            ('Ankara Çankaya', 39.920, 32.854, 0.8, 58, 65),
            ('Antalya Resort', 36.890, 30.708, 0.3, 90, 62)],
    'AU': [('Sydney CBD', -33.870, 151.209, 1.2, 88, 90),
            ('Melbourne Docklands', -37.816, 144.955, 1.0, 85, 88),
            ('Brisbane South Bank', -27.472, 153.021, 0.6, 78, 82)],
    'JP': [('Tokyo Shinjuku', 35.690, 139.699, 3.5, 90, 88),
            ('Osaka Namba', 34.665, 135.501, 1.5, 85, 82),
            ('Kyoto Gion', 35.003, 135.776, 0.4, 92, 80)],
}


@app.route('/api/v1/biz/opportunity-finder')
@require_api_key
def api_opportunity_finder():
    """
    Proactive EV charging site discovery.

    Returns ranked candidate locations for new charging stations
    based on demand signals, competition gaps, infrastructure
    quality, income levels and tourism potential.

    Parameters:
      country   (required) — ISO country code e.g. KE, NG, ZA
      city      (optional) — filter to specific city name
      limit     (optional) — number of results (default 10, max 25)
      min_score (optional) — minimum opportunity score 0-100 (default 50)
      focus     (optional) — 'gap' (low competition), 'demand' (high traffic),
                             'highway' (corridor), 'tourism', 'premium' (high income)

    Example:
      /api/v1/biz/opportunity-finder?country=KE&limit=10
      /api/v1/biz/opportunity-finder?country=NG&focus=premium&city=Lagos
      /api/v1/biz/opportunity-finder?country=ZA&focus=highway
    """
    try:
        country   = request.args.get('country', '').upper().strip()
        city      = request.args.get('city', '').strip().lower()
        limit     = min(int(request.args.get('limit', 10)), 25)
        min_score = int(request.args.get('min_score', 40))
        focus     = request.args.get('focus', 'balanced').lower()

        if not country:
            return jsonify({'error': 'country parameter required. e.g. ?country=KE'}), 400

        # Get country intelligence
        country_intel = None
        if GLOBAL_DATA_LOADED:
            country_intel = get_country(country)

        # Get urban centres for this country
        centres = URBAN_CENTRES.get(country, DEFAULT_CENTRES.get(country, []))
        if not centres:
            return jsonify({
                'error': f'No urban centre data for {country} yet.',
                'available_countries': list(URBAN_CENTRES.keys()) + list(DEFAULT_CENTRES.keys()),
                'tip': 'Contact hello@chargesmart.online to request country data.'
            }), 404

        # Load charger data for competition analysis
        df = load_charger_data(country)

        # ── SCORE EACH CENTRE ──────────────────────────────────
        results = []
        deg2 = 0.018  # ~2km radius

        for centre in centres:
            name, lat, lon, pop_m, tourism, income = centre

            # Filter by city if specified
            if city and city not in name.lower():
                continue

            # ── COMPETITION SCORE (0-25) ──────────────────────
            # Fewer nearby chargers = bigger gap = higher score
            if df is not None:
                nearby = df[
                    (abs(df['lat'] - lat) < deg2) &
                    (abs(df['lon'] - lon) < deg2)
                ]
                existing_nearby = len(nearby[['lat','lon']].drop_duplicates())
            else:
                existing_nearby = 0

            gap_score = max(0, 25 - existing_nearby * 3)

            # ── DEMAND SCORE (0-25) ───────────────────────────
            # Population density + wider area charger activity
            if df is not None:
                wider = df[
                    (abs(df['lat'] - lat) < 0.05) &
                    (abs(df['lon'] - lon) < 0.05)
                ]
                activity = len(wider)
            else:
                activity = 0

            demand_score = min(25, int(pop_m * 4) + min(activity, 10))

            # ── INCOME / WILLINGNESS TO PAY SCORE (0-20) ─────
            income_score = int(income * 0.20)

            # ── TOURISM SCORE (0-15) ──────────────────────────
            tourism_score = int(tourism * 0.15)

            # ── GRID / INFRASTRUCTURE SCORE (0-15) ───────────
            if country_intel:
                grid_score = int(country_intel['grid'] * 0.15)
            else:
                grid_score = 8  # neutral default

            # ── TOTAL ─────────────────────────────────────────
            total = gap_score + demand_score + income_score + tourism_score + grid_score
            total = min(total, 100)

            # ── APPLY FOCUS WEIGHT ────────────────────────────
            if focus == 'gap':
                total = min(100, total + (gap_score * 0.4))
            elif focus == 'demand':
                total = min(100, total + (demand_score * 0.4))
            elif focus == 'tourism':
                total = min(100, total + (tourism_score * 0.8))
            elif focus == 'premium':
                total = min(100, total + (income_score * 0.6))

            total = round(total)

            # ── ROI ESTIMATE ──────────────────────────────────
            cost_mult   = 0.70 if (country_intel and country_intel['gdp_per_cap'] < 3000) else \
                          0.80 if (country_intel and country_intel['gdp_per_cap'] < 8000) else \
                          0.95 if (country_intel and country_intel['gdp_per_cap'] < 30000) else 1.10
            charger_cost = int(8000 * cost_mult)
            sessions_day = round(total / 10 * 1.5, 1)       # demand proxy
            revenue_month = int(sessions_day * 30 * 8.5 * cost_mult)  # avg session value
            payback_yrs  = round((charger_cost * 2) / (revenue_month * 12), 1) if revenue_month else None

            # ── RECOMMENDATION ────────────────────────────────
            grade = 'A+' if total >= 88 else 'A' if total >= 78 else \
                    'B+' if total >= 68 else 'B' if total >= 58 else \
                    'C'  if total >= 45 else 'D'

            charger_type = 'DC Fast (50kW+)' if income >= 75 or tourism >= 80 \
                           else 'AC Fast (22kW)'
            recommended_units = 4 if total >= 80 else 3 if total >= 65 else 2

            verdict = (
                'Premium opportunity — high income, low competition'   if total >= 88 else
                'Strong opportunity — deploy immediately'               if total >= 78 else
                'Good opportunity — viable with standard ROI'          if total >= 68 else
                'Moderate — worth exploring with site survey'          if total >= 58 else
                'Lower priority — consider after primary sites'        if total >= 45 else
                'Not recommended at this stage'
            )

            results.append({
                'rank':              0,  # set after sort
                'location':          name,
                'coordinates':       {'lat': round(lat, 4), 'lon': round(lon, 4)},
                'opportunity_score': total,
                'grade':             grade,
                'verdict':           verdict,
                'score_breakdown': {
                    'competition_gap':   gap_score,
                    'ev_demand':         demand_score,
                    'income_level':      income_score,
                    'tourism_potential': tourism_score,
                    'grid_quality':      grid_score,
                },
                'competition': {
                    'chargers_within_2km': existing_nearby,
                    'level': 'None'   if existing_nearby == 0 else
                             'Low'    if existing_nearby <= 2 else
                             'Medium' if existing_nearby <= 5 else 'High',
                },
                'recommendation': {
                    'charger_type':    charger_type,
                    'units_suggested': recommended_units,
                    'estimated_cost':  f'${charger_cost * recommended_units:,}',
                    'sessions_per_day':sessions_day,
                    'revenue_month':   f'${revenue_month:,}',
                    'payback_years':   payback_yrs,
                },
            })

        # Sort and filter
        results.sort(key=lambda x: x['opportunity_score'], reverse=True)
        results = [r for r in results if r['opportunity_score'] >= min_score]
        results = results[:limit]

        # Add rank
        for i, r in enumerate(results):
            r['rank'] = i + 1

        if not results:
            return jsonify({
                'message': f'No locations found above score {min_score} for {country}.',
                'tip':     f'Try lowering min_score or removing city filter.',
            }), 200

        # Summary
        top = results[0] if results else None
        country_name = country_intel['name'] if country_intel else country

        return jsonify({
            'country':      country_name,
            'country_code': country,
            'focus':        focus,
            'total_found':  len(results),
            'summary': {
                'top_location':     top['location'] if top else None,
                'top_score':        top['opportunity_score'] if top else None,
                'avg_score':        round(sum(r['opportunity_score'] for r in results) / len(results), 1),
                'grade_A_count':    sum(1 for r in results if r['grade'] in ['A+', 'A']),
                'total_investment_estimate': f"${sum(int(r['recommendation']['estimated_cost'].replace('$','').replace(',','')) for r in results):,}",
            },
            'opportunities': results,
            'grid_quality':  country_intel['grid'] if country_intel else 'N/A',
            'renewable_pct': country_intel['renewable'] if country_intel else 'N/A',
            'source':        'ChargeSmart Opportunity Intelligence · chargesmart.online',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

