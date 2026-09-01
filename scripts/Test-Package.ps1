[CmdletBinding()]
param([string] $Server = (Join-Path $PSScriptRoot '..\runtime\llama-server.exe'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$manifest = Join-Path $PSScriptRoot '..\config\runtime.sha256'
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw "Missing runtime manifest: $manifest" }
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rows = @(Get-Content -LiteralPath $manifest | Where-Object { $_ -match '^([0-9a-f]{64}) \*(.+)$' })
if ($rows.Count -lt 1) { throw 'Runtime manifest is empty.' }
foreach ($row in $rows) {
    [void]($row -match '^([0-9a-f]{64}) \*(.+)$')
    $expected = $matches[1]
    $path = Join-Path $root $matches[2]
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing runtime file: $path" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Runtime SHA-256 mismatch: $path" }
}
$version = & (Resolve-Path -LiteralPath $Server).Path --version 2>&1
$versionText = $version -join [Environment]::NewLine
if ($versionText -notmatch 'build 10682, commit 843d57505') { throw "Unexpected runtime build identity: $versionText" }
Write-Host 'Runtime package checksums and build identity match.'
