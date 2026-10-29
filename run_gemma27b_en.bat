@echo off
setlocal
rem TranslateGemma 27B Q6_K: English-to-Japanese subtitles on RTX 5090.
set RT_TRANSLATOR_PROFILE=translategemma27b
set RT_TRANSLATOR_MODE=en
call "%~dp0run.bat"
exit /b %errorlevel%
