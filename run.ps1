Param(
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0",
    [switch]$NoReload,
    [switch]$EnsureVenv,
    [switch]$InstallRequirements,
    [switch]$EnsureSkuDockerImage,
    [switch]$SkipSkuChecks,
    [switch]$PreflightOnly
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
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }
        if ($line -notmatch "=") { return }

        $key, $value = $line -split "=", 2
        $key = $key.Trim()
        $value = $value.Trim()
        if ($value.Contains("#")) {
            $value = ($value -split "\s+#", 2)[0].Trim()
        }
        $value = $value.Trim('"').Trim("'")
        if ([string]::IsNullOrWhiteSpace($key)) { return }

        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

function Get-EnvOrDefault {
    param(
        [string]$Name,
        [string]$DefaultValue
    )

    $value = [System.Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $DefaultValue
    }
    return $value
}

function Assert-CommandExists {
    param(
        [string]$CommandName,
        [string]$Hint
    )
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Команда '$CommandName' недоступна. $Hint"
    }
}

function Test-DockerImageExists {
    param([string]$ImageName)
    docker image inspect $ImageName 2>$null *> $null
    return $LASTEXITCODE -eq 0
}

$EnvFileCandidates = @(
    (Join-Path $ProjectRoot "env"),
    (Join-Path $ProjectRoot ".env")
)

foreach ($envFile in $EnvFileCandidates) {
    Import-EnvFile -FilePath $envFile
}

Assert-CommandExists -CommandName "python" -Hint "Install Python 3.10+ and add it to PATH."

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    if (-not $EnsureVenv) {
        throw "Virtual environment not found: $PythonExe. Run .\run.ps1 -EnsureVenv -InstallRequirements or create .venv manually."
    }

    Write-Host "Creating .venv..." -ForegroundColor Cyan
    try {
        py -3 -m venv (Join-Path $ProjectRoot ".venv")
    }
    catch {
        python -m venv (Join-Path $ProjectRoot ".venv")
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Failed to create virtual environment: $PythonExe"
}

if ($InstallRequirements) {
    $RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $RequirementsPath)) {
        throw "requirements.txt not found: $RequirementsPath"
    }
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r $RequirementsPath
}

$AppEntry = Join-Path $ProjectRoot "app\main.py"
if (-not (Test-Path $AppEntry)) {
    throw "Application entry file not found: $AppEntry"
}

[System.Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")

$RequiredModules = @("fastapi", "uvicorn", "PIL", "numpy", "pandas")
foreach ($moduleName in $RequiredModules) {
    & $PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$moduleName') else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Python module '$moduleName' is missing in .venv. Run: .\run.ps1 -InstallRequirements"
    }
}

if (-not $SkipSkuChecks) {
    $SkuRepoPathRaw = Get-EnvOrDefault -Name "SKU110K_REPO_PATH" -DefaultValue "third_party/SKU110K_CVPR19"
    $SkuWeightsPathRaw = Get-EnvOrDefault -Name "SKU110K_WEIGHTS_PATH" -DefaultValue "models/sku110k_pretrained.h5"
    $SkuRunMode = (Get-EnvOrDefault -Name "SKU110K_RUN_MODE" -DefaultValue "docker").ToLower()
    $SkuDockerImage = Get-EnvOrDefault -Name "SKU110K_DOCKER_IMAGE" -DefaultValue "merch-analyzer-sku110k:tf1.15"
    $SkuPythonBin = Get-EnvOrDefault -Name "SKU110K_PYTHON_BIN" -DefaultValue ".venv_sku/Scripts/python.exe"

    $SkuRepoPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $SkuRepoPathRaw))
    $SkuWeightsPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $SkuWeightsPathRaw))

    if (-not (Test-Path $SkuRepoPath)) {
        throw "SKU110K repo not found: $SkuRepoPath. See README and clone to third_party/SKU110K_CVPR19."
    }
    if (-not (Test-Path $SkuWeightsPath)) {
        throw "SKU110K weights file not found: $SkuWeightsPath."
    }

    switch ($SkuRunMode) {
        "docker" {
            Assert-CommandExists -CommandName "docker" -Hint "Install/start Docker Desktop."
            docker info *> $null
            if ($LASTEXITCODE -ne 0) {
                throw "Docker is installed but daemon is unavailable. Start Docker Desktop."
            }

            if (-not (Test-DockerImageExists -ImageName $SkuDockerImage)) {
                if (-not $EnsureSkuDockerImage) {
                    throw "SKU110K Docker image '$SkuDockerImage' not found. Run with -EnsureSkuDockerImage or execute scripts/build_sku110k_docker.ps1."
                }

                $DockerBuildScript = Join-Path $ProjectRoot "scripts\build_sku110k_docker.ps1"
                if (-not (Test-Path $DockerBuildScript)) {
                    throw "Docker build script not found: $DockerBuildScript"
                }
                Write-Host "Building SKU110K Docker image: $SkuDockerImage..." -ForegroundColor Cyan
                & $DockerBuildScript -ImageName $SkuDockerImage
                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to build SKU110K Docker image: $SkuDockerImage"
                }
            }
        }
        "wsl" {
            Assert-CommandExists -CommandName "wsl" -Hint "Enable WSL or switch SKU110K_RUN_MODE to docker/native."
        }
        "native" {
            $SkuPythonPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $SkuPythonBin))
            if (-not (Test-Path $SkuPythonPath)) {
                throw "SKU110K_PYTHON_BIN not found: $SkuPythonPath (native mode)."
            }
        }
        "auto" {
            # auto is resolved in Python code; verify minimum prerequisites here.
            $hasDocker = [bool](Get-Command docker -ErrorAction SilentlyContinue)
            $hasWsl = [bool](Get-Command wsl -ErrorAction SilentlyContinue)
            $hasNativeBin = Test-Path ([System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $SkuPythonBin)))
            if (-not ($hasDocker -or $hasWsl -or $hasNativeBin)) {
                throw "SKU110K_RUN_MODE=auto, but no docker, no wsl, and no SKU110K_PYTHON_BIN=$SkuPythonBin found."
            }
        }
        default {
            throw "Invalid SKU110K_RUN_MODE='$SkuRunMode'. Allowed: auto|docker|wsl|native."
        }
    }
}

$UvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $BindHost, "--port", $Port)
if (-not $NoReload) {
    $UvicornArgs += @("--reload", "--reload-dir", $ProjectRoot)
}

Write-Host "Preflight OK: environment and dependencies are ready." -ForegroundColor Green
if ($PreflightOnly) {
    Write-Host "PreflightOnly mode: server not started." -ForegroundColor Yellow
    exit 0
}

Write-Host "Starting FastAPI: http://$BindHost`:$Port" -ForegroundColor Green
& $PythonExe @UvicornArgs
