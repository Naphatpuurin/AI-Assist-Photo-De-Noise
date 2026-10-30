@echo off
title OpenRAW Denoise - Canon EOS 6D
echo ===================================================
echo   OpenRAW Denoise (Canon EOS 6D AI Denoise App)
echo   Hardware Acceleration: NVIDIA GeForce RTX 5060
echo ===================================================
echo.
python app\main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
