# Copy CPR ML sources from a full monorepo checkout into cpr-assist/cpr_ml/.
#
#   .\scripts\sync_cpr_ml_from_monorepo.ps1 -MonorepoRoot C:\path\to\cpr

param(
    [Parameter(Mandatory = $true)]
    [string] $MonorepoRoot
)

$ErrorActionPreference = "Stop"
$mono = (Resolve-Path $MonorepoRoot).Path
if (-not (Test-Path (Join-Path $mono "src\config.py"))) {
    Write-Error "MonorepoRoot must contain src\config.py"
}

$dest = Join-Path $PSScriptRoot "..\cpr_ml"
if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path $dest | Out-Null }
$dest = (Resolve-Path $dest).Path
$robocopy = @(
    @("$mono\src", "$dest\src", "/MIR", "/XD", "__pycache__", ".pytest_cache", ".mypy_cache", "/XF", "*.pyc"),
    @("$mono\configs", "$dest\configs", "/MIR", "/XD", "__pycache__"),
    @("$mono\api\cpr_api", "$dest\api\cpr_api", "/MIR", "/XD", "__pycache__"),
    @("$mono\experiments\cpr_s0_image_classifier\runs\s0_image_model", "$dest\experiments\cpr_s0_image_classifier\runs\s0_image_model", "/MIR"),
    @("$mono\experiments\ct_depth_tabular\frozen", "$dest\experiments\ct_depth_tabular\frozen", "/MIR")
)
foreach ($args in $robocopy) {
    $null = New-Item -ItemType Directory -Force -Path $args[1]
    & robocopy @args | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with $LASTEXITCODE" }
}
Write-Host "Synced into $dest"
