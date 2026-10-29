@echo off
setlocal
rem Hy-MT2 30B-A3B Q4_K_M: English-to-Japanese subtitles on RTX 5090.
set RT_TRANSLATOR_PROFILE=hymt2
set RT_TRANSLATOR_MODE=en
call "%~dp0run.bat"
exit /b %errorlevel%
