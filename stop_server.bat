@echo off
echo ===================================================
echo   Stopping Career PathFinder & Freeing Port 8000...
echo ===================================================
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
echo [+] Done. Port 8000 is now free.
