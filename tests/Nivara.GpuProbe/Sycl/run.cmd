@echo off
rem Launches sycl_runner.exe inside the Intel oneAPI environment so the SYCL
rem runtime DLLs (sycl8.dll, ur_loader.dll) resolve. The .NET probe spawns
rem this wrapper instead of the exe directly.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
"%~dp0sycl_runner.exe" %*
exit /b %ERRORLEVEL%