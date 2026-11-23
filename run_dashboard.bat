@echo off
title FraudShield AI - Autoencoder Server
echo ========================================================
echo Starting FraudShield AI Backend Server...
echo Model: 14-Bottleneck Autoencoder (29 -> 20 -> 14 -> 20 -> 29)
echo Dataset: creditcard.csv (284,807 transactions)
echo ========================================================
echo.
echo Opening Dashboard in browser: http://127.0.0.1:8000
echo.
start "" "http://127.0.0.1:8000"
python -m uvicorn backend_server:app --host 127.0.0.1 --port 8000
pause
