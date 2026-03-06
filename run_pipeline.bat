@echo off
echo ============================================
echo  ChargeSmart Data Pipeline
echo ============================================

echo.
echo Step 1: Collecting charger data (UK + EU + US)...
python collector.py existing
echo.

echo Step 2: Enriching raw data...
python enricher.py
echo.

echo Step 3: Training model...
python model.py
echo.

echo Step 4: Pushing to Railway...
git add enriched_data.csv charger_model.pkl
git commit -m "Update dataset and retrain model"
git push
echo.
echo Done! Site will redeploy with fresh data.
