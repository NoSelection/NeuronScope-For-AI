$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $false
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resultsRoot = Join-Path $repoRoot "results"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$archiveRoot = Join-Path $resultsRoot "_pre_erratum_archives"
$logRoot = Join-Path $resultsRoot "_rerun_logs"
$runLogDir = Join-Path $logRoot $timestamp

New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null
New-Item -ItemType Directory -Force -Path $runLogDir | Out-Null

$statusPath = Join-Path $runLogDir "status.txt"
$manifestPath = Join-Path $runLogDir "manifest.txt"
$v2LogPath = Join-Path $runLogDir "v2.log"
$v2ErrPath = Join-Path $runLogDir "v2.stderr.log"
$v3LogPath = Join-Path $runLogDir "v3.log"
$v3ErrPath = Join-Path $runLogDir "v3.stderr.log"

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $statusPath -Append
}

function Move-IfExists {
    param(
        [string]$SourcePath,
        [string]$TargetPath
    )
    if (Test-Path -LiteralPath $SourcePath) {
        Move-Item -LiteralPath $SourcePath -Destination $TargetPath
        Write-Status ("Archived {0} -> {1}" -f $SourcePath, $TargetPath)
    }
}

function Invoke-PythonLogged {
    param(
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath,
        [string]$Label
    )

    $proc = Start-Process `
        -FilePath python `
        -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -Wait `
        -PassThru `
        -NoNewWindow

    if ($proc.ExitCode -ne 0) {
        throw "{0} failed with exit code {1}. See {2} and {3}" -f $Label, $proc.ExitCode, $StdoutPath, $StderrPath
    }
}

Push-Location $repoRoot
try {
    Write-Status "Preparing erratum rerun."

    $v2Source = Join-Path $resultsRoot "self_model_circuits_v2"
    $v3Source = Join-Path $resultsRoot "self_model_circuits_v3"
    $v2Archive = Join-Path $archiveRoot ("self_model_circuits_v2_" + $timestamp)
    $v3Archive = Join-Path $archiveRoot ("self_model_circuits_v3_" + $timestamp)

    Move-IfExists -SourcePath $v2Source -TargetPath $v2Archive
    Move-IfExists -SourcePath $v3Source -TargetPath $v3Archive

    @(
        "timestamp=$timestamp",
        "repo_root=$repoRoot",
        ("git_head=" + (git rev-parse HEAD)),
        "git_status_start:",
        (git status --short),
        "",
        ("python_version=" + (python --version 2>&1)),
        ("torch_transformers=" + (python -c "import torch, transformers; print(f'torch={torch.__version__}; transformers={transformers.__version__}')")),
        ("llm_config_sha256=" + ((Get-FileHash -Algorithm SHA256 (Join-Path $repoRoot 'LLM\config.json')).Hash))
    ) | Set-Content -LiteralPath $manifestPath

    Write-Status ("Manifest written to {0}" -f $manifestPath)
    Write-Status "Starting v2 rerun."

    Invoke-PythonLogged `
        -Arguments @("-m", "experiments.run_v2_self_model") `
        -StdoutPath $v2LogPath `
        -StderrPath $v2ErrPath `
        -Label "v2 rerun"

    Write-Status "v2 rerun completed successfully."
    Write-Status "Starting v3 rerun."

    Invoke-PythonLogged `
        -Arguments @("-m", "experiments.run_v3_head_sweep") `
        -StdoutPath $v3LogPath `
        -StderrPath $v3ErrPath `
        -Label "v3 rerun"

    Write-Status "v3 rerun completed successfully."
    Write-Status "Erratum rerun finished."
}
catch {
    Write-Status ("Rerun failed: {0}" -f $_.Exception.Message)
    throw
}
finally {
    Pop-Location
}
