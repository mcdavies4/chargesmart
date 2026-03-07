"""
ChargeSmart — Collect, Enrich & Retrain
========================================
Collects fresh data from all 30 regions, enriches it, retrains the model,
then tells you to push to Railway.

Usage:
    python collect_and_train.py            # collect all 30 regions once
    python collect_and_train.py loop       # collect all 30 regions every 10 mins forever
    python collect_and_train.py emerging   # emerging markets only
    python collect_and_train.py retrain    # skip collection, just enrich + retrain

Press Ctrl+C at any time to stop looping and trigger an immediate enrich + retrain.
"""

import sys, os, json, glob, time, pickle, subprocess
from datetime import datetime
from collections import Counter

# ── SETTINGS ─────────────────────────────────────────────────
COLLECT_INTERVAL_MINS = 10   # how long between collection rounds in loop mode
COLLECT_MODE          = sys.argv[1] if len(sys.argv) > 1 else "all"

# ─────────────────────────────────────────────────────────────
def banner(text):
    print(f"\n{'='*55}")
    print(f"  {text}")
    print(f"{'='*55}")

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ── STEP 1: COLLECT ───────────────────────────────────────────
def run_collection(mode):
    banner(f"[{ts()}] STEP 1 — COLLECTING ({mode.upper()})")
    import collector

    modes = {
        "all":        collector.ALL_REGIONS,
        "existing":   collector.EXISTING_REGIONS,
        "emerging":   collector.EMERGING_MARKETS,
        "africa":     collector.AFRICA_REGIONS,
        "middle_east":collector.MIDDLE_EAST_REGIONS,
        "asia":       collector.ASIA_REGIONS,
        "latam":      collector.LATAM_REGIONS,
    }
    regions = modes.get(mode, collector.ALL_REGIONS)
    print(f"  Regions ({len(regions)}): {', '.join(regions)}\n")
    count = collector.collect_all(regions)
    print(f"\n  ✅ Collected {count:,} chargers")
    return count

# ── STEP 2: ENRICH ────────────────────────────────────────────
def run_enrichment():
    banner(f"[{ts()}] STEP 2 — ENRICHING DATA")
    import enricher
    enricher.enrich()

    if os.path.exists("enriched_data.csv"):
        import pandas as pd
        df = pd.read_csv("enriched_data.csv")
        size_mb = os.path.getsize("enriched_data.csv") / 1024 / 1024
        print(f"\n  ✅ enriched_data.csv: {len(df):,} rows | {size_mb:.1f}MB")
        if "country" in df.columns:
            print(f"\n  Breakdown by country:")
            for country, count in df["country"].value_counts().items():
                bar = "█" * int(count / max(df["country"].value_counts()) * 30)
                print(f"    {country:<8} {count:>10,}  {bar}")
        return len(df)
    else:
        print("  ❌ enriched_data.csv not created — check enricher.py")
        return 0

# ── STEP 3: RETRAIN ───────────────────────────────────────────
def run_training():
    banner(f"[{ts()}] STEP 3 — RETRAINING MODEL")
    print("  Using GradientBoosting with 21 features...")
    print("  This takes 5-15 minutes depending on data size.\n")

    result = subprocess.run(
        [sys.executable, "model.py"],
        capture_output=False  # show output live
    )

    if result.returncode != 0:
        print("  ❌ model.py failed")
        return False

    if os.path.exists("charger_model.pkl"):
        size_mb = os.path.getsize("charger_model.pkl") / 1024 / 1024
        print(f"\n  ✅ charger_model.pkl saved ({size_mb:.1f}MB)")
        if os.path.exists("model_features.txt"):
            feats = open("model_features.txt").read().split(",")
            print(f"  ✅ Features: {feats}")
        return True
    return False

# ── STEP 4: SUMMARY + PUSH REMINDER ─────────────────────────
def print_summary(rows, trained):
    banner("PIPELINE COMPLETE")

    data_files = glob.glob("data_*.json")
    total_raw  = 0
    for f in data_files:
        try:
            with open(f) as fp:
                for line in fp:
                    snap = json.loads(line)
                    total_raw += len(snap.get("chargers", []))
        except:
            pass

    print(f"""
  Raw charger snapshots:  {total_raw:,}
  Enriched ML rows:       {rows:,}
  Model retrained:        {'✅ Yes' if trained else '❌ No'}
  Finished at:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

  ── DEPLOY TO RAILWAY ─────────────────────────────────────
  git add enriched_data.csv charger_model.pkl model_features.txt
  git commit -m "Retrain: {rows:,} rows, {datetime.now().strftime('%Y-%m-%d')}"
  git push
  ──────────────────────────────────────────────────────────
""")

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":

    print(f"""
╔══════════════════════════════════════════════════════╗
║       ChargeSmart — Collect & Retrain Pipeline       ║
║       Mode: {COLLECT_MODE:<41}║
║       Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                    ║
╚══════════════════════════════════════════════════════╝
""")

    if COLLECT_MODE == "retrain":
        # Skip collection, just enrich + retrain
        rows    = run_enrichment()
        trained = run_training()
        print_summary(rows, trained)
        sys.exit(0)

    if COLLECT_MODE == "loop":
        # Continuous collection loop
        actual_mode = sys.argv[2] if len(sys.argv) > 2 else "all"
        print(f"  Loop mode — collecting every {COLLECT_INTERVAL_MINS} mins. Ctrl+C to stop + retrain.\n")
        rounds = 0
        try:
            while True:
                rounds += 1
                print(f"\n  ── Round {rounds} ──────────────────────────────────")
                run_collection(actual_mode)
                print(f"\n  Waiting {COLLECT_INTERVAL_MINS} minutes before next round...")
                time.sleep(COLLECT_INTERVAL_MINS * 60)
        except KeyboardInterrupt:
            print(f"\n\n  Stopped after {rounds} rounds. Running enrich + retrain now...")

        rows    = run_enrichment()
        trained = run_training()
        print_summary(rows, trained)

    else:
        # Single pass — collect once then enrich + retrain
        try:
            run_collection(COLLECT_MODE)
        except KeyboardInterrupt:
            print("\n  Collection interrupted — proceeding to enrich + retrain...")

        rows    = run_enrichment()
        trained = run_training()
        print_summary(rows, trained)
