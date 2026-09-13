@echo off
rem Builds the SYCL probe runner with the Intel oneAPI DPC++ compiler.
rem Requires: Intel oneAPI Base Toolkit + VS Build Tools (C++ workload).
rem build.cmd sources whichever toolchain is missing from PATH: oneAPI setvars,
rem and VsDevCmd for the MSVC host tools that icpx links against.
rem
rem   build.cmd            -> normal build (sycl_runner.exe next to this script)
rem   build.cmd clean      -> remove build artifacts
rem
rem After building, the .NET probe runs it via the "sycl" mode:
rem   dotnet run -c Release --project tests/Nivara.GpuProbe -- sycl

setlocal EnableDelayedExpansion
set SCRIPT_DIR=%~dp0
set OUT=%SCRIPT_DIR%sycl_runner.exe

if /i "%~1"=="clean" (
    if exist "%OUT%" del "%OUT%"
    exit /b 0
)

rem --- MSVC host toolchain (cl.exe) -------------------------------------------
where cl >nul 2>nul
if errorlevel 1 (
    set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    if exist "!VSDEVCMD!" (
        call "!VSDEVCMD!" -arch=x64 -host_arch=x64 >nul
    ) else (
        echo [sycl] cl.exe not on PATH and VsDevCmd.bat not found.
        echo [sycl] Install VS 2022 Build Tools with the C++ workload.
        exit /b 1
    )
)

rem --- Intel oneAPI DPC++ (icpx) ----------------------------------------------
where icpx >nul 2>nul
if errorlevel 1 (
    set "SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    if exist "!SETVARS!" (
        call "!SETVARS!" >nul
    ) else (
        echo [sycl] icpx not found and Intel oneAPI setvars.bat not found.
        exit /b 1
    )
)

where icpx >nul 2>nul
if errorlevel 1 (
    echo [sycl] icpx still not on PATH after setvars. Check the oneAPI install.
    exit /b 1
)

echo [sycl] icpx: %~dp0
icpx --version | findstr /c:"DPC++" >nul
if errorlevel 1 icpx --version
echo [sycl] building %OUT% ...
icpx -fsycl -O2 "%SCRIPT_DIR%sycl_runner.cpp" -o "%OUT%"
if errorlevel 1 (
    echo [sycl] icpx FAILED
    exit /b 1
)
echo [sycl] OK - %OUT%
endlocal