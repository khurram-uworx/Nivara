@echo off
REM Nivara Fine-Tuning Timing Benchmark
REM Usage: benchmark_timing.cmd [examples=100] [batch_size=2] [epochs=1]
REM
REM Compares PyTorch vs Nivara fine-tuning timing on a subset of SST-2.
REM
REM Results are written to benchmark_results.txt

set "EXAMPLES=%~1"
set "BATCH_SIZE=%~2"
set "EPOCHS=%~3"
if "%EXAMPLES%"=="" set "EXAMPLES=100"
if "%BATCH_SIZE%"=="" set "BATCH_SIZE=2"
if "%EPOCHS%"=="" set "EPOCHS=1"

echo Nivara Fine-Tuning Benchmark > benchmark_results.txt
echo %DATE% %TIME% >> benchmark_results.txt
echo examples=%EXAMPLES% batch_size=%BATCH_SIZE% epochs=%EPOCHS% >> benchmark_results.txt
echo. >> benchmark_results.txt

REM --- PyTorch baseline ---
echo [PyTorch] Starting...
echo. >> benchmark_results.txt
echo === PyTorch === >> benchmark_results.txt
python Python\benchmark_timing.py --epochs %EPOCHS% --batch-size %BATCH_SIZE% --max-examples %EXAMPLES% 2>&1 | tee -append benchmark_results.txt
echo. >> benchmark_results.txt

echo ======================== >> benchmark_results.txt
echo. >> benchmark_results.txt

REM --- Nivara baseline ---
echo [Nivara] Starting...
echo === Nivara === >> benchmark_results.txt
dotnet run --project ..\..\samples\NivaraFineTuning -c Release -- --mode train --epochs %EPOCHS% --batch-size %BATCH_SIZE% --max-examples %EXAMPLES% 2>&1 | tee -append benchmark_results.txt

echo. >> benchmark_results.txt
echo Done. Results written to benchmark_results.txt
