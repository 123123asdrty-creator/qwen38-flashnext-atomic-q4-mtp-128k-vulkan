[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $SourceDirectory,
    [string] $BuildDirectory = 'build-atomic-mtp',
    [ValidateRange(1, 128)]
    [int] $Jobs = 12
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceDirectory).Path
$packageRoot = Split-Path -Parent $PSScriptRoot
$head = (git -C $source rev-parse HEAD).Trim()
if ($head -ne '843d5750579a15ed4a42d73eb862855c271021ac') {
    throw "Expected base commit 843d5750579a15ed4a42d73eb862855c271021ac; found $head"
}
$patch = Join-Path $packageRoot 'patches\qwen4exp-atomic-mtp-runtime.patch'
git -C $source apply --check $patch
if ($LASTEXITCODE -ne 0) { throw 'Source patch preflight failed.' }
git -C $source apply $patch
if ($LASTEXITCODE -ne 0) { throw 'Source patch failed.' }
Copy-Item -LiteralPath (Join-Path $packageRoot 'patches\source-additions\src\llama-hotmoe.cpp') -Destination (Join-Path $source 'src\llama-hotmoe.cpp')
Copy-Item -LiteralPath (Join-Path $packageRoot 'patches\source-additions\src\llama-hotmoe.h') -Destination (Join-Path $source 'src\llama-hotmoe.h')
$builder = Join-Path $packageRoot 'scripts\Build-Windows-Vulkan.ps1'
& $builder -SourceDirectory $source -BuildDirectory $BuildDirectory -Target 'llama-server;llama-perplexity' -Jobs $Jobs
exit $LASTEXITCODE
