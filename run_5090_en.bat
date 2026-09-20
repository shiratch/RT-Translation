@echo off
setlocal
rem RTX 5090 balanced profile: English-to-Japanese subtitles.
set RT_TRANSLATOR_PROFILE=rtx5090
set RT_TRANSLATOR_MODE=en
call "%~dp0run.bat"
exit /b %errorlevel%
