[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $ModelPath,
    [string] $DraftModelPath,
    [switch] $FullHash
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$resolved = (Resolve-Path -LiteralPath $ModelPath).Path
$leaf = Split-Path -Leaf $resolved
$directory = Split-Path -Parent $resolved
$expectedFirst = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf'
$expectedBytes = [int64]94525394976
if ($leaf -ne $expectedFirst) { throw "Select the exact first Atomic Q4_K_M-M64 shard: $expectedFirst" }
$files = @(Get-ChildItem -LiteralPath $directory -File | Where-Object Name -Match '^Qwen3\.8-Flash-Next-AD-4\.27bpw-Q4_K_M-M64-\d{5}-of-00033\.gguf$' | Sort-Object Name)
if ($files.Count -ne 33) { throw "Expected 33 model shards beside shard 1; found $($files.Count)." }
for ($index = 1; $index -le 33; $index++) {
    $expectedName = 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-{0:D5}-of-00033.gguf' -f $index
    if ($files[$index - 1].Name -ne $expectedName) { throw "Missing or misnamed shard: $expectedName" }
}
$totalBytes = [int64](($files | Measure-Object Length -Sum).Sum)
if ($totalBytes -ne $expectedBytes) { throw "Shard byte total mismatch. Expected $expectedBytes; found $totalBytes." }
Write-Host "Target model OK: 33 shards, $totalBytes bytes."
if ($FullHash) {
    $manifestPath = Join-Path $PSScriptRoot '..\config\atomic-q4km-m64.sha256'
    $expected = @{}
    foreach ($line in Get-Content -LiteralPath $manifestPath) {
        if ($line -match '^([0-9a-f]{64}) \*(.+)$') { $expected[$matches[2]] = $matches[1] }
    }
    if ($expected.Count -ne 33) { throw 'The target-model checksum manifest is incomplete.' }
    foreach ($file in $files) {
        Write-Host "Hashing $($file.Name)..."
        $actual = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected[$file.Name]) { throw "SHA-256 mismatch: $($file.Name)" }
    }
    Write-Host 'All 33 target-model checksums match.'
}
if (-not [string]::IsNullOrWhiteSpace($DraftModelPath)) {
    $draft = Get-Item -LiteralPath (Resolve-Path -LiteralPath $DraftModelPath).Path
    $expectedDraftName = 'mtp-Qwen3.8-Flash-Next-Q4_0-qwen4exp-fast.gguf'
    $expectedDraftBytes = [int64]2362007744
    $expectedDraftHash = '41ef1d94ee9249d4140de494d1ad6de4441860e1b50e31cf4cceb0971f8ddf12'
    if ($draft.Name -ne $expectedDraftName) { throw "Select the exact converted MTP sidecar: $expectedDraftName" }
    if ($draft.Length -ne $expectedDraftBytes) { throw "MTP sidecar size mismatch. Expected $expectedDraftBytes; found $($draft.Length)." }
    Write-Host 'Hashing the MTP sidecar...'
    $draftHash = (Get-FileHash -LiteralPath $draft.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($draftHash -ne $expectedDraftHash) { throw 'MTP sidecar SHA-256 mismatch. Sidecars with another graph layout are incompatible.' }
    Write-Host 'MTP sidecar checksum matches.'
}
