Param(
    [string]$ImageName = "merch-analyzer-sku110k:tf1.15"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

docker build -t $ImageName -f .\docker\sku110k\Dockerfile .
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed with exit code $LASTEXITCODE. Is Docker Desktop running?"
}
Write-Host "Built image: $ImageName"
