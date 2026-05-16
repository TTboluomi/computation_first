@echo off
title Deepfake Platform Local Tunnel
echo Connecting to 192.168.218.154 and forwarding local port 5000...
echo.
echo After login, keep this window open.
echo Then open:
echo   http://127.0.0.1:5000/
echo   http://127.0.0.1:5000/admin
echo.
ssh -N -L 5000:127.0.0.1:5000 root@192.168.218.154
pause
