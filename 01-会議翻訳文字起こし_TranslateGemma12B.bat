@echo off
setlocal
rem TranslateGemma 12B Q8_0: English-to-Japanese subtitles on RTX 5090.
set RT_TRANSLATOR_PROFILE=translategemma12b
set RT_TRANSLATOR_MODE=en
call "%~dp0run.bat"
exit /b %errorlevel%
