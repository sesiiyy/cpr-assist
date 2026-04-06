# Run from repository root (cpr/)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
$img = Join-Path $root "runs\ch1_updated_mppose\intermediate\detector_crops\crop_00.png"
$out = Join-Path $root "experiments\pose_backend_compare\results\regen"
$script = Join-Path $PSScriptRoot "compare_pose_backends_on_image.py"

& $py $script --image $img --out-dir $out --torchvision
