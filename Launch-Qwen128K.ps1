[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string] $ModelPath,

    [ValidateRange(1024, 65535)]
    [int] $Port = 8080,

    [switch] $CheckOnly
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$server = Join-Path $packageRoot 'runtime\llama-server.exe'
$profile = Join-Path $packageRoot 'config\hotmoe-coding.rank'
$verifyScript = Join-Path $packageRoot 'scripts\Test-ModelFiles.ps1'

foreach ($required in @($server, $profile, $verifyScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Runtime package is incomplete; missing: $required"
    }
}

$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
& $verifyScript -ModelPath $resolvedModel

$settings = [ordered]@{
    Build              = 'b200-c589f0e'
    Context            = 131072
    CacheK             = 'f16'
    CacheV             = 'f16'
    CpuMoELayers       = 38
    HotSlotsPerLayer   = 24
    DynamicSwaps       = 2
    Batch              = 4096
    Ubatch             = 2048
    Threads            = 22
    FlashAttention     = 'on'
    LoadMode           = 'mmap'
    Speculation        = 'none'
    HostMoE            = 'off'
}

Write-Host 'Qwen3.8-Flash-Next 128K Vulkan/HotMoE profile'
$settings.GetEnumerator() | Format-Table -AutoSize
Write-Host "Model: $resolvedModel"
Write-Host "Listening: http://127.0.0.1:$Port"

if ($CheckOnly) {
    Write-Host 'Preflight passed; server was not started.'
    exit 0
}

# Exact environment used for the accepted A/B profile.
$env:LLAMA_HOTMOE = '24'
$env:LLAMA_HOTMOE_PROFILE = $profile
$env:LLAMA_HOTMOE_MAX_TOK = '1'
$env:LLAMA_HOTMOE_DYNAMIC = '2'
$env:LLAMA_HOTMOE_TRACE = '4'
$env:LLAMA_HOTMOE_PHASED = '0'
$env:LLAMA_HOTMOE_ASYNC = '0'
$env:LLAMA_HOSTMOE = '0'
$env:GGML_HOTMOE_STATS = '4320'
$env:LLAMA_ATTN_ROT_DISABLE = '1'
$env:LLAMA_MEMORY_PRIORITY = '4'
$env:LLAMA_PLE_PREFETCH = '1'

$serverArgs = @(
    '-m', $resolvedModel,
    '--host', '127.0.0.1', '--port', [string] $Port,
    '-ngl', '999', '-ncmoe', '38', '--fit', 'off',
    '--flash-attn', 'on', '-c', '131072', '-np', '1',
    '-b', '4096', '-ub', '2048',
    '-t', '22', '-tb', '22',
    '-ctk', 'f16', '-ctv', 'f16',
    '--prio', '0', '--prio-batch', '0',
    '--load-mode', 'mmap', '--tensor-read-lazy', 'on',
    '--no-context-shift', '--cache-prompt', '--spec-type', 'none',
    '-ot', 'per_layer_token_embd\.weight=CPU',
    '--jinja', '--metrics', '-lv', '4'
)

# Foreground execution preserves Ctrl+C behavior and passes paths containing
# spaces without Start-Process command-line re-tokenization.
& $server @serverArgs
exit $LASTEXITCODE
