[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $SourceDirectory,
    [string] $BuildDirectory = 'build-atomic-mtp',
    [string] $Target = 'llama-server;llama-perplexity',
    [ValidateRange(1, 128)]
    [int] $Jobs = 12
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceDirectory).Path
$build = Join-Path $source $BuildDirectory
$vsDev = @(
    'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat',
    'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat',
    'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $vsDev) { throw 'Visual Studio C++ developer environment was not found.' }
$environment = & cmd.exe /d /s /c ('"' + $vsDev + '" -arch=x64 >nul && set')
foreach ($entry in $environment) {
    $separator = $entry.IndexOf('=')
    if ($separator -gt 0) {
        [Environment]::SetEnvironmentVariable($entry.Substring(0, $separator), $entry.Substring($separator + 1), 'Process')
    }
}
& cmake -S $source -B $build -G 'NMake Makefiles' -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
foreach ($item in $Target.Split(';', [StringSplitOptions]::RemoveEmptyEntries)) {
    & cmake --build $build --config Release --target $item --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
