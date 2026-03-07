"""
ChargeSmart — Railway Pipeline Worker
======================================
Runs as a separate Railway service (worker).
Collects data every 6 hours, enriches, retrains,
then saves enriched_data.csv + charger_model.pkl
to /tmp so the web service can use them.

Railway env vars needed:
  PIPELINE_MODE   = emerging | africa | all | existing  (default: emerging)
  PIPELINE_HOURS  = 6  (how often to run, default: 6)
"""

import os, sys, time, json, pickle, subprocess

# Match the same DATA_DIR as the web service
DATA_DIR = os.environ.get('DATA_DIR', '.')
os.chdir(DATA_DIR)  # write all files directly to DATA_DIR

from datetime import datetime

MODE          = os.environ.get('PIPELINE_MODE', 'emerging')
INTERVAL_HRS  = int(os.environ.get('PIPELINE_HOURS', '6'))
INTERVAL_SECS = INTERVAL_HRS * 3600

def ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def banner(msg):
    print(f"\n{'='*55}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'='*55}", flush=True)

def run_pipeline():
    banner(f"PIPELINE START — mode={MODE}")

    # ── COLLECT ───────────────────────────────────────────────
    log("Step 1: Collecting data...")
    try:
        import collector
        modes = {
            'all':        collector.ALL_REGIONS,
            'emerging':   collector.EMERGING_MARKETS,
            'africa':     collector.AFRICA_REGIONS,
            'existing':   collector.EXISTING_REGIONS,
            'middle_east':collector.MIDDLE_EAST_REGIONS,
            'asia':       collector.ASIA_REGIONS,
            'latam':      collector.LATAM_REGIONS,
        }
        regions = modes.get(MODE, collector.EMERGING_MARKETS)
        log(f"Collecting {len(regions)} regions: {', '.join(regions)}")
        count = collector.collect_all(regions)
        log(f"✅ Collected {count:,} chargers")
    except Exception as e:
        log(f"❌ Collection failed: {e}")
        return False

    # ── ENRICH ────────────────────────────────────────────────
    log("Step 2: Enriching data...")
    try:
        import enricher
        enricher.enrich()

        if os.path.exists('enriched_data.csv'):
            import pandas as pd
            df = pd.read_csv('enriched_data.csv')
            log(f"✅ enriched_data.csv: {len(df):,} rows")
            if 'country' in df.columns:
                counts = df['country'].value_counts()
                for country, n in counts.items():
                    log(f"   {country}: {n:,}")
        else:
            log("❌ enriched_data.csv not created")
            return False
    except Exception as e:
        log(f"❌ Enrichment failed: {e}")
        return False

    # ── RETRAIN ───────────────────────────────────────────────
    log("Step 3: Retraining model...")
    try:
        result = subprocess.run(
            [sys.executable, 'model.py'],
            capture_output=False,
            timeout=1800  # 30 min max
        )
        if result.returncode == 0 and os.path.exists('charger_model.pkl'):
            size = os.path.getsize('charger_model.pkl') / 1024 / 1024
            log(f"✅ Model saved ({size:.1f}MB)")
        else:
            log("❌ Model training failed")
            return False
    except subprocess.TimeoutExpired:
        log("❌ Model training timed out (>30 mins)")
        return False
    except Exception as e:
        log(f"❌ Training error: {e}")
        return False

    banner(f"PIPELINE COMPLETE ✅")
    log(f"Next run in {INTERVAL_HRS} hours")
    return True

# ── MAIN LOOP ─────────────────────────────────────────────────
if __name__ == '__main__':
    log(f"ChargeSmart Pipeline Worker starting")
    log(f"Mode: {MODE} | Interval: every {INTERVAL_HRS} hours")
    log(f"First run starting now...\n")

    while True:
        try:
            run_pipeline()
        except Exception as e:
            log(f"❌ Pipeline crashed: {e}")
            log("Will retry next interval")

        log(f"Sleeping {INTERVAL_HRS} hours...")
        time.sleep(INTERVAL_SECS)
