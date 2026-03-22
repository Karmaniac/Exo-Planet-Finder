# run_pipeline.ps1

param(
    [Parameter(Position=0)]
    [string]$Subcommand = "pipeline",

    [Parameter(Position=1)]
    [string]$TicId = "",

    [int]$MaxTargets = 300,
    [switch]$ToiOnly,
    [switch]$SkipFetch,
    [switch]$TestOnly,
    [int]$Sector = 0,
    [switch]$NoPlot
)

$ErrorActionPreference = "Stop"

function Write-Cyan   ($msg) { Write-Host $msg -ForegroundColor Cyan  }
function Write-Green  ($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Yellow ($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Red    ($msg) { Write-Host $msg -ForegroundColor Red   }

# -- Locate venv ---------------------------------------------------------------
$VenvName  = "exoplanet"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath  = $null

foreach ($candidate in @(
    "$ScriptDir\$VenvName",
    "$HOME\$VenvName",
    ".\$VenvName"
)) {
    if (Test-Path "$candidate\Scripts\activate.ps1") {
        $VenvPath = $candidate
        break
    }
}

if (-not $VenvPath) {
    Write-Red "ERROR: Could not find venv '$VenvName'."
    Write-Host "Expected it at one of:"
    Write-Host "  $ScriptDir\$VenvName"
    Write-Host "  $HOME\$VenvName"
    Write-Host "  .\$VenvName"
    Write-Host ""
    Write-Host "Create it with:  python -m venv $VenvName"
    exit 1
}

Write-Cyan "Activating venv: $VenvPath"
& "$VenvPath\Scripts\Activate.ps1"

$CsvFile = "labeled_tess_dataset.csv"

# ==============================================================================
# PREDICT subcommand
# ==============================================================================
if ($Subcommand -eq "predict") {

    if (-not $TicId) {
        Write-Red "ERROR: TIC ID required."
        Write-Host "Usage: .\run_pipeline.ps1 predict <TIC_ID> [-Sector N] [-NoPlot]"
        Write-Host "Example: .\run_pipeline.ps1 predict 307210830"
        exit 1
    }

    Write-Cyan "============================================="
    Write-Cyan "  ExoPlanet Finder - Predict"
    Write-Cyan "============================================="
    Write-Host "  TIC ID  : $TicId"
    Write-Host "  Sector  : $(if ($Sector -gt 0) { $Sector } else { 'auto' })"
    Write-Host "  Plot    : $(if ($NoPlot) { 'no' } else { 'yes' })"
    Write-Cyan "============================================="

    $PredictArgs = @($TicId)
    if ($Sector -gt 0) { $PredictArgs += "--sector", $Sector }
    if ($NoPlot)        { $PredictArgs += "--no-plot" }

    python predict.py @PredictArgs
    exit 0
}

# ==============================================================================
# PIPELINE subcommand (default)
# ==============================================================================

Write-Cyan "============================================="
Write-Cyan "  ExoPlanet Finder - Pipeline"
Write-Cyan "============================================="
Write-Host "  Max targets : $MaxTargets"
Write-Host "  TOI only    : $ToiOnly"
Write-Host "  Skip fetch  : $SkipFetch"
Write-Host "  Test only   : $TestOnly"
Write-Cyan "============================================="
Write-Host ""

# -- Step 1: Fetch -------------------------------------------------------------
if (-not $TestOnly) {
    if ($SkipFetch -and (Test-Path $CsvFile)) {
        Write-Yellow "[Step 1] Skipping fetch - $CsvFile already exists."
    } else {
        Write-Green "[Step 1] Fetching labeled TESS dataset..."
        python fetch_tess_labeled_dataset.py --max-targets $MaxTargets --output $CsvFile
        Write-Green "[Step 1] Done. Saved to $CsvFile"
    }
    Write-Host ""

    # -- Step 2: Train ---------------------------------------------------------
    Write-Green "[Step 2] Training classifier..."
    $TrainArgs = @("--csv", $CsvFile)
    if ($ToiOnly) {
        $TrainArgs += "--toi-only"
        Write-Yellow "  (TOI-only mode - BLS skipped)"
    }
    python train_classifier.py @TrainArgs
    Write-Green "[Step 2] Done. Model saved to exoplanet_rf_classifier.joblib"
    Write-Host ""
}

# -- Step 3: Test --------------------------------------------------------------
Write-Green "[Step 3] Running test..."
python test.py
Write-Green "[Step 3] Done."
