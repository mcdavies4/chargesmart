@echo off
echo.
echo ============================================================
echo   ChargeSmart -- Africa + Emerging Markets Full Pipeline
echo ============================================================
echo.

:: Step 1 - Collect Africa (fresh pass on top of what collector already gathered)
echo [STEP 1] Collecting Africa regions...
python collector.py africa
echo.

:: Step 2 - Collect all other emerging markets (Middle East, Asia, LatAm)
echo [STEP 2] Collecting Middle East + Asia + LatAm...
python -c "import collector; collector.collect_all(collector.MIDDLE_EAST_REGIONS + collector.ASIA_REGIONS + collector.LATAM_REGIONS)"
echo.

:: Step 3 - Enrich all data into enriched_data.csv
echo [STEP 3] Enriching all data...
python enricher.py
echo.

:: Step 4 - Retrain model
echo [STEP 4] Retraining model...
python model.py
echo.

:: Step 5 - Check data
echo [STEP 5] Data audit...
python check_data.py
echo.

:: Step 6 - Git push
echo [STEP 6] Pushing to Railway...
git add enriched_data.csv charger_model.pkl model_features.txt
git commit -m "Retrain: Africa + Emerging markets refresh %date%"
git push
echo.

echo ============================================================
echo   DONE. Site will redeploy on Railway automatically.
echo ============================================================
pause
