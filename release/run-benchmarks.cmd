@echo off
REM Canonical, committed benchmark runner for Nivara releases.
REM Usage: release\run-benchmarks.cmd <version>   (e.g. 1.5.1)
REM Writes all captures to release\<version>\. Safe to re-run; never overwrites a prior
REM release folder's artefacts unless you delete it first (see RELEASING.md).

setlocal
if "%~1"=="" (
  echo Usage: %~nx0 ^<version^>  ^(e.g. 1.5.1^) 1>&2
  exit /b 2
)
set "VERSION=%~1"
set "ROOT=%~dp0.."
cd /d "%ROOT%"
set "OUT=release\v%VERSION%"
if not exist "%OUT%" mkdir "%OUT%"

echo === Core perf harness ===
dotnet run --project tests\Nivara.PerformanceTests -c Release -- --json "%OUT%\nivara-perf-v%VERSION%.json" --runs 3 > "%OUT%\core.txt" 2>&1

echo === NivaraIncident ===
dotnet run --project samples\NivaraIncident -c Release -- analyze --quick --dataset samples\data\incident-lab --scenario A > "%OUT%\nivaraincident-v%VERSION%.txt" 2>&1

echo === NivaraInference (CPU) ===
dotnet run --project samples\NivaraInference -c Release -- mobilenet_v2 benchmark > "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- resnet18 benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- minilm benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- distilbert benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- distilbert_sst benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- smollm benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- smollm --precision bf16 benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- qwen benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- modernbert benchmark >> "%OUT%\nivarainference-cpu-v%VERSION%.txt" 2>&1

echo === NivaraInference (Silk) ===
dotnet run --project samples\NivaraInference -c Release -- minilm --silk benchmark > "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- distilbert --silk benchmark >> "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- distilbert_sst --silk benchmark >> "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- laya --silk benchmark >> "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- modernbert --silk benchmark >> "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- qwen --silk benchmark >> "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- qwen --silk --plain benchmark >> "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- smollm --silk benchmark >> "%OUT%\nivarainference-silk-v%VERSION%.txt" 2>&1

echo === NivaraInference (BLAS) ===
dotnet run --project samples\NivaraInference -c Release -- minilm benchmark --blas --threads 2 > "%OUT%\nivarainference-blas-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- distilbert benchmark --blas --threads 2 >> "%OUT%\nivarainference-blas-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- distilbert_sst benchmark --blas --threads 2 >> "%OUT%\nivarainference-blas-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- modernbert benchmark --blas --threads 4 >> "%OUT%\nivarainference-blas-v%VERSION%.txt" 2>&1
dotnet run --project samples\NivaraInference -c Release -- laya benchmark --blas --threads 2 >> "%OUT%\nivarainference-blas-v%VERSION%.txt" 2>&1

echo === PyTorch inference references ===
pushd samples\NivaraInference\Python
python minilm_benchmark.py > "%ROOT%\%OUT%\pytorch-minilm-v%VERSION%.txt" 2>&1
python distilbert_benchmark.py > "%ROOT%\%OUT%\pytorch-distilbert-v%VERSION%.txt" 2>&1
python modernbert_benchmark.py > "%ROOT%\%OUT%\pytorch-modernbert-v%VERSION%.txt" 2>&1
python smollm_generate_reference.py --dtype float32 > "%ROOT%\%OUT%\pytorch-smollm-f32-v%VERSION%.txt" 2>&1
python smollm_generate_reference.py --dtype bfloat16 > "%ROOT%\%OUT%\pytorch-smollm-bf16-v%VERSION%.txt" 2>&1
popd

echo === NivaraFineTuning ===
pushd samples\NivaraFineTuning
call benchmark_timing.cmd 100 2 1
popd
if exist samples\NivaraFineTuning\benchmark_results.txt copy /Y samples\NivaraFineTuning\benchmark_results.txt "%OUT%\nivarafinetuning-v%VERSION%.txt" >nul

echo DONE > "%OUT%\status.txt"
endlocal
