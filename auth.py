"""
ChargeSmart Authentication System
Magic link passwordless login + session management
"""
import json, os, secrets, hashlib
from datetime import datetime, timedelta

USERS_FILE   = 'users.json'
TOKENS_FILE  = 'magic_tokens.json'
SESSIONS_FILE= 'sessions.json'

# ── FILE HELPERS ─────────────────────────────────────────────
def load_users():
    return json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}

def save_users(u):
    json.dump(u, open(USERS_FILE,'w'), indent=2)

def load_tokens():
    return json.load(open(TOKENS_FILE)) if os.path.exists(TOKENS_FILE) else {}

def save_tokens(t):
    json.dump(t, open(TOKENS_FILE,'w'), indent=2)

def load_sessions():
    return json.load(open(SESSIONS_FILE)) if os.path.exists(SESSIONS_FILE) else {}

def save_sessions(s):
    json.dump(s, open(SESSIONS_FILE,'w'), indent=2)

# ── USER MANAGEMENT ──────────────────────────────────────────
def get_or_create_user(email, name=''):
    users = load_users()
    email = email.lower().strip()
    uid   = hashlib.md5(email.encode()).hexdigest()[:16]
    if uid not in users:
        users[uid] = {
            'uid':        uid,
            'email':      email,
            'name':       name or email.split('@')[0].title(),
            'created':    datetime.now().strftime('%Y-%m-%d'),
            'plan':       'free',
            'api_key':    None,
            'favourites': [],
            'login_count':0,
            'last_login': None,
        }
        save_users(users)
    return users[uid]

def get_user_by_uid(uid):
    return load_users().get(uid)

def get_user_by_email(email):
    email = email.lower().strip()
    uid   = hashlib.md5(email.encode()).hexdigest()[:16]
    return load_users().get(uid)

def update_user(uid, updates):
    users = load_users()
    if uid in users:
        users[uid].update(updates)
        save_users(users)
    return users.get(uid)

# ── MAGIC LINKS ──────────────────────────────────────────────
def create_magic_token(email):
    """Generate a one-time login token valid for 15 minutes"""
    tokens = load_tokens()
    # Clean expired tokens
    now = datetime.now()
    tokens = {k:v for k,v in tokens.items()
              if datetime.fromisoformat(v['expires']) > now}
    token  = secrets.token_urlsafe(32)
    tokens[token] = {
        'email':   email.lower().strip(),
        'expires': (now + timedelta(minutes=15)).isoformat(),
        'used':    False,
    }
    save_tokens(tokens)
    return token

def verify_magic_token(token):
    """Returns email if valid, None if expired/used/invalid"""
    tokens = load_tokens()
    if token not in tokens:
        return None
    t = tokens[token]
    if t['used']:
        return None
    if datetime.fromisoformat(t['expires']) < datetime.now():
        return None
    # Mark as used
    tokens[token]['used'] = True
    save_tokens(tokens)
    return t['email']

# ── SESSIONS ─────────────────────────────────────────────────
def create_session(uid):
    """Create a 30-day session for a user"""
    sessions = load_sessions()
    # Clean old sessions
    now = datetime.now()
    sessions = {k:v for k,v in sessions.items()
                if datetime.fromisoformat(v['expires']) > now}
    session_id = secrets.token_urlsafe(48)
    sessions[session_id] = {
        'uid':     uid,
        'created': now.isoformat(),
        'expires': (now + timedelta(days=30)).isoformat(),
    }
    save_sessions(sessions)
    return session_id

def get_session_user(session_id):
    """Returns user dict if session valid, None otherwise"""
    if not session_id:
        return None
    sessions = load_sessions()
    s = sessions.get(session_id)
    if not s:
        return None
    if datetime.fromisoformat(s['expires']) < datetime.now():
        return None
    return get_user_by_uid(s['uid'])

def delete_session(session_id):
    sessions = load_sessions()
    sessions.pop(session_id, None)
    save_sessions(sessions)

# ── EMAIL ─────────────────────────────────────────────────────
def send_magic_link(email, token, base_url):
    """Send magic link via email — uses SendGrid if configured, logs to console otherwise"""
    link = f"{base_url}auth/verify?token={token}"

    sendgrid_key = os.environ.get('SENDGRID_API_KEY', '')
    from_email   = os.environ.get('FROM_EMAIL', 'hello@chargesmart.online')

    if sendgrid_key:
        try:
            import urllib.request, json as _json
            payload = _json.dumps({
                'personalizations': [{'to': [{'email': email}]}],
                'from': {'email': from_email, 'name': 'ChargeSmart'},
                'subject': '⚡ Your ChargeSmart login link',
                'content': [{
                    'type': 'text/html',
                    'value': f'''
                    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:32px;">
                        <div style="font-size:28px;margin-bottom:8px;">⚡</div>
                        <h2 style="margin-bottom:8px;">Your login link</h2>
                        <p style="color:#666;margin-bottom:24px;">Click the button below to sign in to ChargeSmart. This link expires in 15 minutes and can only be used once.</p>
                        <a href="{link}" style="display:inline-block;background:#00ff87;color:#0a0a0f;padding:14px 28px;border-radius:8px;text-decoration:none;font-weight:700;font-family:monospace;letter-spacing:1px;">SIGN IN TO CHARGESMART →</a>
                        <p style="color:#999;font-size:12px;margin-top:24px;">Or copy this link: {link}</p>
                        <p style="color:#999;font-size:11px;">If you didn't request this, ignore this email.</p>
                    </div>'''
                }]
            }).encode()
            req = urllib.request.Request(
                'https://api.sendgrid.com/v3/mail/send',
                data=payload,
                headers={'Authorization': f'Bearer {sendgrid_key}',
                         'Content-Type': 'application/json'},
                method='POST'
            )
            urllib.request.urlopen(req, timeout=10)
            return True, 'Email sent'
        except Exception as e:
            print(f"SendGrid error: {e}")
            print(f"MAGIC LINK (fallback): {link}")
            return False, str(e)
    else:
        # No email service — log link so it works in development
        print(f"\n{'='*60}")
        print(f"MAGIC LINK for {email}:")
        print(f"{link}")
        print(f"{'='*60}\n")
        return True, 'link_logged'
