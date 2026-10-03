$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit) first.' }
}
$runtimePython = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $runtimePython -m pip install --upgrade pip 'setuptools<81' wheel
if ($LASTEXITCODE -ne 0) { throw 'Environment bootstrap failed.' }
# Install the same CUDA build used to validate this project.
& $runtimePython -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $runtimePython -m pip install --no-build-isolation -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$env:PYTHONPATH = 'src;third_party/mia_net_official'
& $runtimePython -B check_runtime.py
if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed.' }
