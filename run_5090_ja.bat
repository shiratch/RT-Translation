@echo off
setlocal
rem RTX 5090 balanced profile: Japanese transcription.
set RT_TRANSLATOR_PROFILE=rtx5090
set RT_TRANSLATOR_MODE=ja
call "%~dp0run.bat"
exit /b %errorlevel%
