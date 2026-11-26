@echo off
setlocal EnableExtensions EnableDelayedExpansion
title PadelIQ Live Launcher
cd /d "%~dp0"

rem ---- find Python -------------------------------------------------------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY (
    echo [ERROR] Python was not found on PATH. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

set "ESP32_FILE=esp32_ip.txt"
set "RTSP_FILE=rtsp_url.txt"
set "YT_FILE=youtube_url.txt"
set "EMPTY_COUNT=0"

:menu
set "RC="
cls
echo ==========================================================
echo               PadelIQ  Live Camera Engine
echo ==========================================================
echo   [1] ENTERPRISE WEB COMMAND CENTER  (http://localhost:8000)
echo ----------------------------------------------------------
echo   Desktop (OpenCV window) modes:
echo   [2] Live Webcam (BRIO preferred)   (--source webcam, MJPG 720p)
echo   [M] Mobile QR Phone Camera         (--source mobile, scan QR, HTTPS :8443)
echo   [3] ESP32-CAM Stream               (http://IP:81/stream)
echo   [4] RTSP IP Camera
echo   [5] YouTube video / live URL       (via yt-dlp -g)
echo   [6] Simulated Live Stream          (clean match footage loop)
echo ----------------------------------------------------------
echo   [7] Auto-detect source             (ESP32 / webcam / video)
echo   [8] Run headless smoke test        (--test-run 100)
echo   [9] Auto-calibrate the demo court  (from court lines)
echo   [Q] Quit
echo ----------------------------------------------------------
echo   Desktop keys: [C] calibrate [L] lock [R] rotate 90 [E] shutter [F] fullscreen
echo                 [F12] failover [M] QR portal [S] record [X] reset [Q] quit
echo   Score:        [1] +1 Team A (P1-P2)  [2] +1 Team B (P3-P4)  [0] reset score
echo ==========================================================
set "CHOICE="
set /p "CHOICE=Select an option: "
if not defined CHOICE (
    set /a EMPTY_COUNT+=1
    if !EMPTY_COUNT! GEQ 5 goto end
    goto menu
)
set "EMPTY_COUNT=0"
if /i "!CHOICE!"=="1" goto enterprise
if /i "!CHOICE!"=="2" goto webcam
if /i "!CHOICE!"=="m" goto mobile
if /i "!CHOICE!"=="3" goto esp32
if /i "!CHOICE!"=="4" goto rtsp
if /i "!CHOICE!"=="5" goto youtube
if /i "!CHOICE!"=="6" goto sim
if /i "!CHOICE!"=="7" goto auto
if /i "!CHOICE!"=="8" goto smoketest
if /i "!CHOICE!"=="9" goto autocal
if /i "!CHOICE!"=="q" goto end
goto menu

:enterprise
echo Starting PadelIQ Enterprise Command Center on http://localhost:8000 ...
echo (the browser opens automatically; press Ctrl+C here to stop the server)
%PY% padeliq_enterprise_server.py --open-browser
set "RC=!errorlevel!"
goto done

:youtube
set "SAVED_YT="
if exist "%YT_FILE%" set /p SAVED_YT=<"%YT_FILE%"
if defined SAVED_YT echo Saved YouTube URL found (press Enter to reuse it).
echo Example: https://www.youtube.com/watch?v=C1XNW75T0bs
set "YT_URL="
set /p "YT_URL=YouTube URL: "
if not defined YT_URL set "YT_URL=!SAVED_YT!"
if not defined YT_URL (
    echo [ERROR] No YouTube URL given.
    goto done
)
> "%YT_FILE%" echo(!YT_URL!
%PY% -c "import yt_dlp" 2>nul || (
    echo Installing yt-dlp ...
    %PY% -m pip install -q yt-dlp
)
%PY% padeliq_live_stream.py --source "!YT_URL!"
set "RC=!errorlevel!"
goto done

:autocal
%PY% padeliq_live_stream.py --source sim --auto-calibrate
set "RC=!errorlevel!"
if "!RC!"=="0" echo Preview saved as auto_calibration_preview.jpg
goto done

:webcam
set "CAMIDX=webcam"
set /p "CAMIDX=Webcam index [Enter = best camera, Logitech BRIO preferred]: "
%PY% padeliq_live_stream.py --source !CAMIDX!
set "RC=!errorlevel!"
goto done

:mobile
echo Phone: join the same Wi-Fi, scan the QR code in the window, accept the certificate once.
echo (if Windows Firewall asks, allow Python on private networks so the phone can reach port 8443)
%PY% -c "import qrcode, cryptography" 2>nul || (
    echo Installing qrcode + cryptography ...
    %PY% -m pip install -q "qrcode[pil]" cryptography
)
%PY% padeliq_live_stream.py --source mobile
set "RC=!errorlevel!"
goto done

:esp32
set "SAVED_IP="
if exist "%ESP32_FILE%" set /p SAVED_IP=<"%ESP32_FILE%"
if not defined SAVED_IP set "SAVED_IP=192.168.1.50"
echo Saved ESP32-CAM address: !SAVED_IP!
echo (enter an IP like 192.168.1.50, or a full URL like http://192.168.1.50:81/stream)
set "ESP_IP="
set /p "ESP_IP=ESP32-CAM IP or URL [Enter = !SAVED_IP!]: "
if not defined ESP_IP set "ESP_IP=!SAVED_IP!"
> "%ESP32_FILE%" echo(!ESP_IP!
%PY% padeliq_live_stream.py --source "!ESP_IP!"
set "RC=!errorlevel!"
goto done

:rtsp
set "SAVED_RTSP="
if exist "%RTSP_FILE%" set /p SAVED_RTSP=<"%RTSP_FILE%"
if defined SAVED_RTSP echo Saved RTSP URL found (press Enter to reuse it).
echo Example: rtsp://user:password@192.168.1.64:554/Streaming/Channels/101
set "RTSP_URL="
set /p "RTSP_URL=RTSP URL: "
if not defined RTSP_URL set "RTSP_URL=!SAVED_RTSP!"
if not defined RTSP_URL (
    echo [ERROR] No RTSP URL given.
    goto done
)
> "%RTSP_FILE%" echo(!RTSP_URL!
%PY% padeliq_live_stream.py --source "!RTSP_URL!"
set "RC=!errorlevel!"
goto done

:sim
%PY% padeliq_live_stream.py --source sim
set "RC=!errorlevel!"
goto done

:auto
set "AUTO_ARGS=--source auto"
if exist "%ESP32_FILE%" (
    set /p SAVED_IP=<"%ESP32_FILE%"
    if defined SAVED_IP set "AUTO_ARGS=--source auto --esp32-ip "!SAVED_IP!""
)
%PY% padeliq_live_stream.py !AUTO_ARGS!
set "RC=!errorlevel!"
goto done

:smoketest
%PY% padeliq_live_stream.py --source sim --test-run 100
set "RC=!errorlevel!"
if not "!RC!"=="0" (
    echo.
    echo [FAILED] Smoke test failed - see messages above.
) else (
    echo.
    echo [PASSED] Smoke test OK - screenshot saved as live_hud_verified.jpg
)
goto done

:done
if not defined RC set "RC=0"
echo.
echo Engine exited with code !RC!.
pause
goto menu

:end
endlocal
exit /b 0
