[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string] $ModelPath,

    [switch] $FullHash
)

$ErrorActionPreference = 'Stop'
$resolved = (Resolve-Path -LiteralPath $ModelPath).Path
$leaf = Split-Path -Leaf $resolved
$directory = Split-Path -Parent $resolved
$expectedFirst = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf'
$expectedBytes = [int64]94525394976

if ($leaf -ne $expectedFirst) {
    throw "Select the first AtomicChat Q4_K_M-M64 shard: $expectedFirst"
}

$files = Get-ChildItem -LiteralPath $directory -File |
    Where-Object Name -Match '^Qwen3\.8-Flash-Next-AD-4\.27bpw-Q4_K_M-M64-\d{5}-of-00033\.gguf$' |
    Sort-Object Name

if ($files.Count -ne 33) {
    throw "Expected 33 model shards beside the selected file; found $($files.Count)."
}

for ($index = 1; $index -le 33; $index++) {
    $expectedName = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-{0:D5}-of-00033.gguf' -f $index
    if ($files[$index - 1].Name -ne $expectedName) {
        throw "Missing or misnamed shard: $expectedName"
    }
}

$totalBytes = ($files | Measure-Object Length -Sum).Sum
if ($totalBytes -ne $expectedBytes) {
    throw "Shard byte total mismatch. Expected $expectedBytes; found $totalBytes."
}

Write-Host "Model structure OK: 33 shards, $totalBytes bytes (88.03 GiB)."

if ($FullHash) {
    $manifestPath = Join-Path $PSScriptRoot '..\config\atomic-q4km-m64.sha256'
    $expected = @{}
    foreach ($line in Get-Content -LiteralPath $manifestPath) {
        if ($line -match '^([0-9a-f]{64}) \*(.+)$') {
            $expected[$matches[2]] = $matches[1]
        }
    }
    if ($expected.Count -ne 33) { throw 'Checksum manifest is incomplete.' }

    foreach ($file in $files) {
        Write-Host "Hashing $($file.Name)..."
        $actual = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected[$file.Name]) {
            throw "SHA256 mismatch: $($file.Name)"
        }
    }
    Write-Host 'All 33 SHA256 checksums match the AtomicChat manifest.'
}

