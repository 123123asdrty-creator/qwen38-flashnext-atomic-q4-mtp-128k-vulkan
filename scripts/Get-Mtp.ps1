[CmdletBinding()]
param([string]$DestinationDirectory = (Join-Path $PSScriptRoot '..\models\mtp'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($PSVersionTable.PSVersion -lt [Version]'7.0') { throw 'PowerShell 7 or newer is required.' }
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\config\mtp-download.json') -Raw | ConvertFrom-Json
$name = 'mtp-Qwen3.8-Flash-Next-Q4_0-qwen4exp-fast.gguf'
if ($manifest.file -ne $name -or $manifest.parts.Count -ne 2) { throw 'Unexpected MTP download manifest.' }
New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
$directory = (Resolve-Path -LiteralPath $DestinationDirectory).Path
$target = Join-Path $directory $name
function Test-Artifact([string]$File, [long]$Bytes, [string]$Hash) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $File).Length -ne $Bytes) { return $false }
    return (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Hash
}
$lock = [IO.File]::Open((Join-Path $directory ($name + '.lock')), 'OpenOrCreate', 'ReadWrite', 'None')
$temporary = Join-Path $directory ($name + '.' + [Guid]::NewGuid().ToString('N') + '.assembling')
try {
    if (Test-Path -LiteralPath $target) {
        if (-not (Test-Artifact $target $manifest.bytes $manifest.sha256)) { throw 'Existing MTP helper does not match the measured version. Move it aside before downloading.' }
        Write-Host 'MTP helper checksum verified.'
        return $target
    }
    $parts = @()
    for ($index = 0; $index -lt $manifest.parts.Count; $index++) {
        $part = $manifest.parts[$index]
        if ($part.file -ne ($name + '.part' + ($index + 1))) { throw 'Unexpected MTP part filename.' }
        if (-not $part.url.StartsWith('https://github.com/123123asdrty-creator/Qwen3.8-Flash-Next-AtomicChat-128K-Vulkan/releases/download/mtp-2026.09.04/')) { throw 'Unexpected MTP download source.' }
        $partPath = Join-Path $directory $part.file
        if (-not (Test-Artifact $partPath $part.bytes $part.sha256)) {
            if ((Test-Path -LiteralPath $partPath) -and (Get-Item -LiteralPath $partPath).Length -ge $part.bytes) {
                Remove-Item -LiteralPath $partPath
            }
            Write-Host ('Downloading MTP part {0} of 2...' -f ($index + 1))
            Invoke-WebRequest -Uri $part.url -OutFile $partPath -Resume -MaximumRetryCount 2 -RetryIntervalSec 2
            if (-not (Test-Artifact $partPath $part.bytes $part.sha256)) { throw ('MTP part checksum mismatch: ' + $part.file) }
        }
        $parts += $partPath
    }
    Write-Host 'Joining and verifying the MTP helper...'
    $output = [IO.File]::Open($temporary, 'CreateNew', 'Write', 'None')
    try {
        foreach ($partPath in $parts) {
            $inputStream = [IO.File]::OpenRead($partPath)
            try { $inputStream.CopyTo($output, 8MB) } finally { $inputStream.Dispose() }
        }
    } finally { $output.Dispose() }
    if (-not (Test-Artifact $temporary $manifest.bytes $manifest.sha256)) { throw 'Assembled MTP checksum mismatch.' }
    [IO.File]::Move($temporary, $target)
    foreach ($partPath in $parts) { Remove-Item -LiteralPath $partPath }
    Write-Host 'MTP helper ready.'
    return $target
} finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    $lock.Dispose()
}
