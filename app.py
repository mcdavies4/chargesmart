from flask import Flask, request, jsonify, render_template, redirect, make_response, session, url_for
import pandas as pd
import numpy as np
import pickle
import os
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
    if os.path.exists('enriched_data.csv'):
        df = pd.read_csv('enriched_data.csv')
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
    with open('charger_model.pkl', 'wb') as f:
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

if os.path.exists('charger_model.pkl'):
    try:
        with open('charger_model.pkl', 'rb') as f:
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
        if os.path.exists('enriched_data.csv'):
            import pandas as pd
            import numpy as np
            df = pd.read_csv('enriched_data.csv', usecols=['lat','lon','country'])
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
            'charger_count': len(charger_coords),
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

        # Ensure api_keys.json exists
        import json as _json
        if not os.path.exists('api_keys.json'):
            with open('api_keys.json', 'w') as f:
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
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl', 'rb'))
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
            if os.path.exists('charger_model.pkl'):
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
        'version': '2.0.0',
        'deployed': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'endpoints': 25,
        'status': 'live'
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
    if not os.path.exists('enriched_data.csv'):
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
        df_full.to_csv('enriched_data.csv', index=False)
        print(f"Generated {len(df_full)} synthetic rows -> enriched_data.csv")
    cols = ['lat','lon','hour','day_of_week','capacity','location_type','operator','country']
    # Only load cols that exist
    available = pd.read_csv('enriched_data.csv', nrows=0).columns.tolist()
    load_cols = [c for c in cols if c in available]
    df = pd.read_csv('enriched_data.csv', usecols=load_cols)
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
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl','rb'))

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
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl','rb'))

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
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl','rb'))

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

        model = pickle.load(open('charger_model.pkl','rb')) if os.path.exists('charger_model.pkl') else None
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
        update_user(email, {'last_login': datetime.datetime.now().isoformat()})
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
        update_user(user['email'], {'api_key': result['key']})
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
        update_user(user['email'], {'favourites': favourites})
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
        update_user(user['email'], updates)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        sync_favourites(user['email'], favs)
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
        update_user(user['email'], {'api_key': api_key, 'plan': tier})
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
        api_key = create_api_key(user['email'], 'free')
        update_user(user['email'], {'api_key': api_key})
        return jsonify({'success': True, 'api_key': api_key})
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
            update_user(user['email'], {'name': name})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


PLAN_LABELS = {
    'free':         'Free',
    'pro':          'Pro',
    'fleet':        'Fleet',
    'api_developer':'Developer',
    'api_business': 'Business',
    'enterprise':   'Enterprise',
}

