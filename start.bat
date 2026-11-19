@echo off
setlocal EnableExtensions
title SemCache AI launcher

rem ============================================================
rem  SemCache AI one-click launcher
rem  The port numbers below must match backend/config.py
rem  (BACKEND_PORT, FRONTEND_PORT) and the API base URL in
rem  frontend/app.js. Keep all three in sync.
rem ============================================================
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5500"
set "OLLAMA_URL=http://localhost:11434"
set "CHAT_MODEL=qwen2.5:3b"
set "EMBED_MODEL=nomic-embed-text"

rem --- Step 1: work from the folder that holds this file ---
cd /d "%~dp0"

echo.
echo === SemCache AI launcher ===
echo.

rem --- Step 2: find Python 3.11 or newer ---
set "PY_CMD="
set "PY_VER="
call :try_python "py -3.11"
call :try_python "py -3"
call :try_python "python"
if defined PY_CMD goto :python_ok

echo [ERROR] No Python 3.11 or newer found.
echo Tried: py -3.11, then py -3, then python.
echo Install Python from https://www.python.org/downloads/ and
echo tick "Add python.exe to PATH" during setup. Then run again.
pause
exit /b 1

:python_ok
echo [OK] Using %PY_CMD% (%PY_VER%)

rem --- Step 3: virtual environment (create once, reuse later) ---
if exist "venv\Scripts\python.exe" (
    echo [OK] Reusing the existing venv in .\venv
    goto :venv_ok
)
echo [..] Creating the venv, first run only...
%PY_CMD% -m venv venv
if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    echo Delete the half-made .\venv folder and run this again.
    pause
    exit /b 1
)
echo [OK] Created a new venv in .\venv
:venv_ok

rem --- Step 4: dependencies ---
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt is missing from this folder.
    echo Restore it from the project repository, then run again.
    pause
    exit /b 1
)

rem Skip the install when requirements.txt has not changed since
rem the last successful run (hash kept in venv\.req_hash) AND the
rem packages still import. The import check matters: without it,
rem deleting a package from the venv would go unnoticed.
set "REQ_HASH="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%~dp0requirements.txt" SHA256') do if not defined REQ_HASH set "REQ_HASH=%%H"
set "OLD_HASH="
if exist "venv\.req_hash" set /p OLD_HASH=<"venv\.req_hash"
if "%REQ_HASH%"=="%OLD_HASH%" (
    "venv\Scripts\python.exe" -c "import fastapi, uvicorn, pydantic, numpy, httpx" >nul 2>&1
    if not errorlevel 1 (
        echo [OK] requirements.txt unchanged and all packages import - skipping pip install
        goto :deps_ok
    )
    echo [..] A package is missing from the venv - reinstalling...
)

echo [..] Installing dependencies from requirements.txt...
call "venv\Scripts\activate.bat"
python -m pip install --quiet --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    echo Likely cause: no internet, or a package could not build.
    echo Read the pip message above, fix it, and run this again.
    pause
    exit /b 1
)
> "venv\.req_hash" echo %REQ_HASH%
echo [OK] Dependencies installed
goto :ollama_check

:deps_ok
call "venv\Scripts\activate.bat"

rem --- Step 5: Ollama server ---
:ollama_check
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed, or not on PATH.
    echo Download it from https://ollama.com/download and run again.
    pause
    exit /b 1
)
call :ollama_up
if not errorlevel 1 goto :ollama_ok
echo [..] Ollama is not answering - starting it in a minimized window...
start "Ollama Serve" /min cmd /c ollama serve
set /a TRIES=0
:ollama_wait
ping -n 2 127.0.0.1 >nul
call :ollama_up
if not errorlevel 1 goto :ollama_ok
set /a TRIES+=1
if %TRIES% lss 20 goto :ollama_wait
echo [ERROR] Ollama did not answer within 20 seconds.
echo Open a terminal, run "ollama serve", and read the error.
pause
exit /b 1
:ollama_ok
echo [OK] Ollama server is running

rem --- Step 6: models (pull only what is missing) ---
ollama list 2>nul | findstr /i /c:"%CHAT_MODEL%" >nul
if errorlevel 1 (
    echo [..] Pulling %CHAT_MODEL% - about 1.9 GB, this can take a while...
    ollama pull %CHAT_MODEL%
    if errorlevel 1 (
        echo [ERROR] Pulling %CHAT_MODEL% failed. Check your internet and retry.
        pause
        exit /b 1
    )
)
ollama list 2>nul | findstr /i /c:"%EMBED_MODEL%" >nul
if errorlevel 1 (
    echo [..] Pulling %EMBED_MODEL% - about 274 MB...
    ollama pull %EMBED_MODEL%
    if errorlevel 1 (
        echo [ERROR] Pulling %EMBED_MODEL% failed. Check your internet and retry.
        pause
        exit /b 1
    )
)
echo [OK] Models ready: %CHAT_MODEL% and %EMBED_MODEL%

rem --- Step 7: the two ports must be free ---
call :port_used %BACKEND_PORT%
if not errorlevel 1 (
    echo [ERROR] Port %BACKEND_PORT% is already in use by another program.
    echo Close that program, or run stop.bat if SemCache is already
    echo running, then start this again.
    pause
    exit /b 1
)
call :port_used %FRONTEND_PORT%
if not errorlevel 1 (
    echo [ERROR] Port %FRONTEND_PORT% is already in use by another program.
    echo Close that program, or run stop.bat if SemCache is already
    echo running, then start this again.
    pause
    exit /b 1
)
echo [OK] Ports %BACKEND_PORT% and %FRONTEND_PORT% are free

rem --- Step 8: backend and frontend, each in its own window ---
echo [..] Starting the backend window...
start "SemCache Backend" cmd /k "cd /d "%~dp0" && venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port %BACKEND_PORT%"
echo [..] Starting the frontend window...
start "SemCache Frontend" cmd /k "cd /d "%~dp0" && venv\Scripts\python.exe -m http.server %FRONTEND_PORT% --bind 127.0.0.1 --directory frontend"

rem --- Step 9: wait until both answer, then open the browser ---
echo [..] Waiting for backend and frontend to answer...
set /a TRIES=0
:wait_services
ping -n 2 127.0.0.1 >nul
set /a TRIES+=1
curl -s -o nul -m 2 "http://127.0.0.1:%BACKEND_PORT%/health"
set "BACKEND_UP=%errorlevel%"
curl -s -o nul -m 2 "http://127.0.0.1:%FRONTEND_PORT%/"
set "FRONTEND_UP=%errorlevel%"
if "%BACKEND_UP%"=="0" if "%FRONTEND_UP%"=="0" goto :services_up
if %TRIES% lss 30 goto :wait_services
echo [ERROR] The services did not answer within 30 seconds.
echo Look at the two SemCache windows for the actual error message.
pause
exit /b 1

:services_up
echo [OK] Backend and frontend are up
rem Prefer Microsoft Edge; fall back to the default browser.
set "EDGE_PATH="
for /f "tokens=2*" %%A in ('reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe" /ve 2^>nul') do set "EDGE_PATH=%%B"
if not defined EDGE_PATH for /f "tokens=2*" %%A in ('reg query "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe" /ve 2^>nul') do set "EDGE_PATH=%%B"
if defined EDGE_PATH (
    start "" "%EDGE_PATH%" "http://127.0.0.1:%FRONTEND_PORT%"
) else (
    start "" "http://127.0.0.1:%FRONTEND_PORT%"
)

rem --- Open DB Browser for SQLite with the cache database ---
rem So you can watch the stats/entries tables update in real time
rem while chatting. Look in the usual install locations first.
set "DBBROWSE_PATH="
if exist "%ProgramFiles(x86)%\DB Browser for SQLite\DB Browser for SQLite.exe" set "DBBROWSE_PATH=%ProgramFiles(x86)%\DB Browser for SQLite\DB Browser for SQLite.exe"
if not defined DBBROWSE_PATH if exist "%ProgramFiles%\DB Browser for SQLite\DB Browser for SQLite.exe" set "DBBROWSE_PATH=%ProgramFiles%\DB Browser for SQLite\DB Browser for SQLite.exe"
if not defined DBBROWSE_PATH if exist "%LOCALAPPDATA%\Programs\DB Browser for SQLite\DB Browser for SQLite.exe" set "DBBROWSE_PATH=%LOCALAPPDATA%\Programs\DB Browser for SQLite\DB Browser for SQLite.exe"
if defined DBBROWSE_PATH (
    echo [..] Opening DB Browser for SQLite with semcache.db...
    start "" "%DBBROWSE_PATH%" "%~dp0data\semcache.db"
    echo [OK] DB Browser for SQLite opened with data\semcache.db
) else (
    echo [WARN] DB Browser for SQLite not found in the usual locations.
    echo        Install it from https://sqlitebrowser.org/ and re-run, or
    echo        open data\semcache.db manually to watch the live history.
)

rem --- Step 10: summary ---
echo.
echo === SemCache AI is running ===
echo Backend : http://127.0.0.1:%BACKEND_PORT%
echo Frontend: http://127.0.0.1:%FRONTEND_PORT%
echo DB      : data\semcache.db (opened in DB Browser for SQLite)
echo To stop : close the "SemCache Backend" and "SemCache Frontend"
echo           windows, or just run stop.bat
echo.
pause
exit /b 0

rem ---------------- helper routines ----------------

rem Sets PY_CMD/PY_VER if the given command is Python 3.11+.
:try_python
if defined PY_CMD exit /b 0
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
set "PY_CMD=%~1"
for /f "delims=" %%V in ('%~1 --version 2^>nul') do set "PY_VER=%%V"
exit /b 0

rem errorlevel 0 = Ollama answers, 1 = it does not.
:ollama_up
curl -s -o nul -m 2 "%OLLAMA_URL%/api/tags"
exit /b %errorlevel%

rem errorlevel 0 = something is listening on the port, 1 = port is free.
:port_used
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1', %1); exit 0 } catch { exit 1 } finally { $c.Close() }" >nul 2>&1
exit /b %errorlevel%
