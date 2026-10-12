<#
.SYNOPSIS
    Installs and verifies FSR 4 v07 INT8/DOT4 shader assets for bbhost on Windows.
.DESCRIPTION
    Downloads the MIT-licensed FSR 4 v07 INT8/DOT4 asset set built by Q2RTX
    from AMD's FidelityFX SDK FSR 4 source. Verifies all files and SHA-256
    digests against the published manifests, redownloading any corrupt files.
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$Destination = "",

    [Parameter()]
    [string[]]$Tiers = @("1080", "2160")
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Destination)) {
    # Default to fsr4_shaders in the repository / package root (two levels above tools/win)
    $RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $Destination = Join-Path $RepoRoot "fsr4_shaders"
} else {
    if (-not [System.IO.Path]::IsPathRooted($Destination)) {
        $Destination = [System.IO.Path]::GetFullPath($Destination)
    }
}

$Commit = "ae8d628fae208813172446d1e49ed94150b04658"
$BaseUrl = "https://raw.githubusercontent.com/FireBurn/Q2RTX/$Commit/baseq2/fsr4_shaders"

if (-not (Test-Path -Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}
$DestResolved = (Resolve-Path $Destination).Path

$Files = [System.Collections.Generic.List[string]]::new()
$Files.Add("LICENSE-FSR4-v07.txt")
$Files.Add("rcas.spv")
$Files.Add("spd_auto_exposure.spv")

$Models = @("native", "quality", "balanced", "performance", "ultraperf", "drs")
foreach ($m in $Models) {
    $Files.Add("fsr4_model_v07_i8_${m}_initializers.bin")
    $Files.Add("fsr4_model_v07_i8_${m}_pre_weights.bin")
    $Files.Add("fsr4_model_v07_i8_${m}_shader_manifest.json")
    foreach ($t in $Tiers) {
        $Files.Add("fsr4_model_v07_i8_${m}_${t}_pre.spv")
        $Files.Add("fsr4_model_v07_i8_${m}_${t}_post.spv")
        for ($p = 1; $p -le 12; $p++) {
            $Files.Add("fsr4_model_v07_i8_${m}_${t}_pass${p}.spv")
        }
    }
}

function Download-AssetFile([string]$fileName, [string]$targetPath) {
    $url = "$BaseUrl/$fileName"
    $partPath = "$targetPath.part"
    Write-Host "Downloading $fileName..."
    Invoke-WebRequest -Uri $url -OutFile $partPath -UseBasicParsing
    Move-Item -Path $partPath -Destination $targetPath -Force
}

Write-Host "Installing FSR 4 v07 assets into $DestResolved ($($Files.Count) files requested)..."

$Downloaded = 0
foreach ($f in $Files) {
    # Path-bound check
    if ($f -match '[\\/]' -or $f.Contains('..')) {
        throw "Invalid filename in requested asset list: $f"
    }
    $targetPath = Join-Path $DestResolved $f
    if (-not ((Test-Path -Path $targetPath) -and ((Get-Item $targetPath).Length -gt 0))) {
        Download-AssetFile -fileName $f -targetPath $targetPath
        $Downloaded++
    }
}

Write-Host "Initial fetch complete: $Downloaded newly downloaded, $(($Files.Count - $Downloaded)) already present."

Write-Host "Verifying downloaded assets against manifests..."
$VerifiedCount = 0

foreach ($m in $Models) {
    $manifestName = "fsr4_model_v07_i8_${m}_shader_manifest.json"
    $manifestFile = Join-Path $DestResolved $manifestName
    if (-not (Test-Path $manifestFile)) {
        throw "Manifest $manifestName is missing after download."
    }
    $json = Get-Content $manifestFile -Raw | ConvertFrom-Json
    $artifacts = $json.artifacts
    if (-not $artifacts) {
        throw "Manifest $manifestName does not contain artifacts object."
    }

    foreach ($prop in $artifacts.PSObject.Properties) {
        $artName = $prop.Name
        # Security sanity check: artifact name must be simple relative filename
        if ($artName -match '[\\/]' -or $artName.Contains('..')) {
            throw "Manifest $manifestName contains dangerous path traversal filename: $artName"
        }
        $expectedSha = $prop.Value.sha256
        $expectedSize = $prop.Value.size

        # Only verify files that match the requested tiers/models
        $isRelevant = $false
        if ($artName -eq "rcas.spv" -or $artName -eq "spd_auto_exposure.spv" -or
            $artName.EndsWith("_initializers.bin") -or $artName.EndsWith("_pre_weights.bin")) {
            $isRelevant = $true
        } else {
            foreach ($t in $Tiers) {
                if ($artName.Contains("_${t}_")) {
                    $isRelevant = $true
                    break
                }
            }
        }
        if (-not $isRelevant) {
            continue
        }

        $artPath = Join-Path $DestResolved $artName
        $attempts = 0
        $isValid = $false

        while (-not $isValid -and $attempts -lt 2) {
            $attempts++
            if (-not (Test-Path $artPath)) {
                Write-Warning "$artName is missing, attempting redownload (attempt $attempts)..."
                Download-AssetFile -fileName $artName -targetPath $artPath
            }

            $item = Get-Item $artPath
            $sizeOk = (-not $expectedSize) -or ($item.Length -eq $expectedSize)
            $shaOk = $false
            if ($sizeOk -and $expectedSha) {
                $actualSha = (Get-FileHash -Path $artPath -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actualSha -eq $expectedSha.ToLowerInvariant()) {
                    $shaOk = $true
                }
            }

            if ($sizeOk -and $shaOk) {
                $isValid = $true
            } else {
                if ($attempts -lt 2) {
                    Write-Warning "$artName is corrupt (size or hash mismatch), redownloading..."
                    Remove-Item -Path $artPath -Force
                    Download-AssetFile -fileName $artName -targetPath $artPath
                } else {
                    throw "Verification failure: $artName remains corrupt after redownload."
                }
            }
        }
        $VerifiedCount++
    }
}

Write-Host "Verified $VerifiedCount files against manifest checksums successfully."
Write-Host "FSR 4 v07 asset installation finished successfully."