
TIER_LIMITS = {
    'free':           100,
    'pro':            500,
    'fleet':          2000,
    'api_developer':  10000,
    'api_business':   100000,
    'api_enterprise': 9999999,
}

TIER_PRICES = {
    'free':           0,
    'pro':            9.99,
    'fleet':          29.99,
    'api_developer':  49,
    'api_business':   199,
    'api_enterprise': 999,
}

TIER_PRIORITY = {
    'free': 0, 'pro': 1, 'fleet': 2,
    'api_developer': 3, 'api_business': 4, 'api_enterprise': 5
}

"""
ChargeSmart API Key System
Handles key generation, validation, rate limiting and usage tracking
"""
import json
import os
import secrets
import string
from datetime import datetime, date

# Use /tmp on Railway (read-only filesystem), local otherwise
import os as _os
# Use DATA_DIR env var (Railway Volume = /data, local = .)
_data_dir = _os.environ.get('DATA_DIR', '.')
KEYS_FILE = _os.path.join(_data_dir, 'api_keys.json')

TIER_LIMITS = {
    'free':       100,
    'developer':  10000,
    'business':   100000,
    'enterprise': 999999999,
}

TIER_PRICES = {
    'free':       0,
    'developer':  49,
    'business':   199,
    'enterprise': 999,
}


# Which endpoints each tier can access
TIER_ENDPOINTS = {
    'free': [
        '/api/v1/predict',
        '/api/v1/deserts',
        '/api/v1/journey-cost',
        '/api/v1/carbon',
        '/api/v1/reviews',
    ],
    'developer': [
        '/api/v1/predict', '/api/v1/deserts', '/api/v1/journey-cost',
        '/api/v1/carbon', '/api/v1/reviews',
        '/api/v1/peak-hours', '/api/v1/operators', '/api/v1/coverage',
        '/api/v1/nearest', '/api/v1/heatmap', '/api/v1/compare',
        '/api/v1/operators/compare', '/api/v1/forecast',
        '/api/v1/gap-score', '/api/v1/predict/batch',
    ],
    'business': 'all',
    'enterprise': 'all',
}

GOV_BIZ_ENDPOINTS = [
    '/api/v1/gov/investment-priority', '/api/v1/gov/charger-density',
    '/api/v1/gov/planning', '/api/v1/gov/netzero', '/api/v1/gov/disparity',
    '/api/v1/biz/site-score', '/api/v1/biz/competitor-audit',
    '/api/v1/biz/depot-optimise', '/api/v1/biz/property-score',
    '/api/v1/biz/retail-opportunity',
]

def check_endpoint_access(tier, endpoint_path):
    """Returns (allowed, error_message)"""
    allowed = TIER_ENDPOINTS.get(tier, TIER_ENDPOINTS['free'])
    if allowed == 'all':
        return True, None
    # Strip query string
    path = endpoint_path.split('?')[0]
    if path in allowed:
        return True, None
    # Determine which tier unlocks it
    if path in GOV_BIZ_ENDPOINTS:
        needed = 'Business'
        price  = '£199/month'
    else:
        needed = 'Developer'
        price  = '£49/month'
    return False, f'This endpoint requires the {needed} tier ({price}). Upgrade at chargesmart.online/developers'

def load_keys():
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            return json.load(f)
    return {}

def save_keys(keys):
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)

def generate_key():
    chars = string.ascii_letters + string.digits
    token = ''.join(secrets.choice(chars) for _ in range(32))
    return f"cs_live_{token}"

def create_api_key(email, tier='free'):
    keys = load_keys()
    # Check if email already has a key
    for key, data in keys.items():
        if data.get('email') == email:
            return {'key': key, 'tier': data['tier'], 'existing': True}
    key = generate_key()
    keys[key] = {
        'email':      email,
        'tier':       tier,
        'created':    datetime.now().strftime('%Y-%m-%d'),
        'usage':      {},
        'total_calls': 0,
    }
    save_keys(keys)
    return {'key': key, 'tier': tier, 'existing': False}

def validate_key(api_key):
    """Returns (valid, tier, remaining_calls, error_message)"""
    if not api_key:
        return False, None, 0, 'Missing API key. Add Authorization: Bearer YOUR_KEY header'
    keys = load_keys()
    if api_key not in keys:
        return False, None, 0, 'Invalid API key. Get yours free at chargesmart.online/developers'
    data = keys[api_key]
    tier  = data.get('tier', 'free')
    limit = TIER_LIMITS.get(tier, 100)
    today = str(date.today())
    usage_today = data.get('usage', {}).get(today, 0)
    if usage_today >= limit:
        return False, tier, 0, f'Daily limit reached ({limit} calls/day on {tier} tier). Upgrade at chargesmart.online/developers'
    return True, tier, limit - usage_today, None

def record_usage(api_key):
    keys = load_keys()
    if api_key not in keys:
        return
    today = str(date.today())
    if 'usage' not in keys[api_key]:
        keys[api_key]['usage'] = {}
    keys[api_key]['usage'][today] = keys[api_key]['usage'].get(today, 0) + 1
    keys[api_key]['total_calls'] = keys[api_key].get('total_calls', 0) + 1
    save_keys(keys)

def get_key_stats(api_key):
    keys = load_keys()
    if api_key not in keys:
        return None
    data = keys[api_key]
    tier  = data.get('tier', 'free')
    limit = TIER_LIMITS[tier]
    today = str(date.today())
    today_usage = data.get('usage', {}).get(today, 0)
    # Last 7 days usage
    from datetime import timedelta
    weekly = []
    for i in range(7):
        d = str(date.today() - timedelta(days=i))
        weekly.append({'date': d, 'calls': data.get('usage', {}).get(d, 0)})
    return {
        'email':         data.get('email'),
        'tier':          tier,
        'limit_per_day': limit,
        'used_today':    today_usage,
        'remaining':     max(0, limit - today_usage),
        'total_calls':   data.get('total_calls', 0),
        'created':       data.get('created'),
        'weekly_usage':  list(reversed(weekly)),
    }