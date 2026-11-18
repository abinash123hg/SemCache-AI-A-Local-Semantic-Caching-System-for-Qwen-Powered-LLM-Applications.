@echo off
setlocal
rem Closes the two SemCache windows and their python child processes.
rem The filter is the window title, so no other program is touched.
rem The Ollama server is left running on purpose: other apps may use it.
taskkill /f /t /fi "WINDOWTITLE eq SemCache Backend*" >nul 2>&1
taskkill /f /t /fi "WINDOWTITLE eq SemCache Frontend*" >nul 2>&1
echo SemCache windows closed (if they were running).
ping -n 4 127.0.0.1 >nul
exit /b 0
