# A Windows Performance Recorder trace of the whole machine while bbhost runs:
# the CPU profile (a sample a millisecond on every core, with stacks, and every
# context switch with its wait reason), for tools/etw/profile.py. The method is
# Kyo's KyoPS4x one (its PERF_NOTES, "the first ETW CPU profile" and "trace 3").
#
#   powershell -File tools\etw\record.ps1 -Seconds 30 -Out build\perf\trace.etl
#   -Delay 60       start recording 60 s from now (into a run already started)
#   -Profiles CPU,GPU   add WPR's GPU profile (DxgKrnl: what GPUView reads)
#
# WPR starts kernel tracing, which takes an administrator: not elevated, this
# asks once (UAC) and runs itself elevated, waiting for that to finish.
param(
    [int]$Seconds = 30,
    [string]$Out = "build\perf\trace.etl",
    [int]$Delay = 0,
    [string]$Profiles = "CPU",
    [switch]$Elevated
)
$ErrorActionPreference = "Stop"
$wpr = "C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\wpr.exe"
if (-not (Test-Path $wpr)) { Write-Error "wpr.exe not found: install the Windows Performance Toolkit (Windows ADK)"; exit 1 }
$Out = [System.IO.Path]::GetFullPath($Out)

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    if ($Elevated) { Write-Error "still not elevated"; exit 1 }
    if ($Delay -gt 0) { Start-Sleep -Seconds $Delay }
    $argline = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Seconds $Seconds -Out `"$Out`" -Profiles $Profiles -Elevated"
    $p = Start-Process powershell -Verb RunAs -ArgumentList $argline -Wait -PassThru -WindowStyle Minimized
    if (-not (Test-Path $Out)) {
        $why = if (Test-Path "$Out.record.log") { (Get-Content "$Out.record.log") -join "; " } else { "no log: the UAC prompt was declined?" }
        Write-Host "no trace written (exit $($p.ExitCode)): $why"
        exit 1
    }
    Write-Host "trace: $Out ($([math]::Round((Get-Item $Out).Length / 1MB)) MB)"
    exit 0
}
if (-not $Elevated -and $Delay -gt 0) { Start-Sleep -Seconds $Delay }

# From here native tools run: under "Stop", Windows PowerShell 5.1 turns any
# line one writes to stderr into a terminating error (wpr -cancel with nothing
# recording did, and the elevated run died before starting). What this half
# says goes to <Out>.record.log too: run elevated, its window is not seen.
$ErrorActionPreference = "Continue"
$log = "$Out.record.log"
function Say($m) { Write-Host $m; Add-Content -Path $log -Value $m }
Set-Content -Path $log -Value "record.ps1: $Profiles for $Seconds s into $Out"
# A trace a previous run left going would refuse the start.
$null = & $wpr -cancel 2>&1
$start = @()
foreach ($p in $Profiles.Split(",")) { $start += @("-start", $p.Trim()) }
$o = & $wpr @start -filemode 2>&1
if ($LASTEXITCODE -ne 0) { Say "wpr -start failed ($LASTEXITCODE): $o"; exit 1 }
Start-Sleep -Seconds $Seconds
$o = & $wpr -stop $Out 2>&1
if ($LASTEXITCODE -ne 0) { Say "wpr -stop failed ($LASTEXITCODE): $o"; exit 1 }
Say "trace written"
