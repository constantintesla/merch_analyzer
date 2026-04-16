Param(
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Import-EnvFile {
    param([string]$FilePath)

    if (-not (Test-Path $FilePath)) {
        return
    }

    Get-Content $FilePath | ForEach-Object {
        $line = $_.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) { return }
        if ($line.StartsWith("#")) { return }
        if ($line -notmatch "=") { return }

        $key, $value = $line -split "=", 2
        $key = $key.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        if ([string]::IsNullOrWhiteSpace($key)) { return }

        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

$EnvFileCandidates = @(
    (Join-Path $ProjectRoot "env"),
    (Join-Path $ProjectRoot ".env")
)

foreach ($envFile in $EnvFileCandidates) {
    Import-EnvFile -FilePath $envFile
}

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Не найдено виртуальное окружение: $PythonExe. Сначала выполните: python -m venv .venv и pip install -r requirements.txt"
}

& $PythonExe -m uvicorn app.main:app --reload --host $BindHost --port $Port
