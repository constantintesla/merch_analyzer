Param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Не найдено виртуальное окружение: $PythonExe. Сначала выполните: python -m venv .venv и pip install -r requirements.txt"
}

& $PythonExe -m uvicorn app.main:app --reload --host $BindHost --port $Port
