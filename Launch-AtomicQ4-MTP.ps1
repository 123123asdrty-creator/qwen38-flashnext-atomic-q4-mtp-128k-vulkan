[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [Alias('ModelPath')]
    [string] $Model,

    [Alias('DraftModelPath')]
    [string] $DraftModel,

    [string] $Server = '',

    [ValidateRange(8192, 262144)]
    [int] $Context = 131072,

    [ValidateRange(32, 8192)]
    [int] $Batch = 4096,

    [ValidateRange(32, 8192)]
    [int] $UBatch = 1536,

    [ValidateRange(32, 4096)]
    [int] $DraftUBatch = 256,

    [ValidateRange(0, 8)]
    [int] $ContextCheckpoints = 2,

    [ValidateRange(0, 8)]
    [int] $DraftNMax = 3,

    [ValidateRange(0.0, 1.0)]
    [double] $DraftPMin = 0.7,

    [ValidateRange(1024, 65535)]
    [int] $Port = 8080,

    [ValidateRange(1, 256)]
    [int] $Threads = 22,

    [ValidateRange(0, 64)]
    [int] $CpuMoeLayers = 40,

    [ValidateSet('bf16', 'f16', 'q8_0')]
    [string] $CacheK = 'bf16',

    [ValidateSet('bf16', 'f16', 'q8_0')]
    [string] $CacheV = 'bf16',

    [ValidateSet('q8_0', 'bf16', 'f16')]
    [string] $DraftCache = 'q8_0',

    [ValidateRange(0, 16)]
    [int] $GpuDeviceIndex = 1,

    [switch] $DisableMtp,
    [switch] $DisablePromptCache,
    [switch] $DisableQsaKeyOnly,
    [switch] $DisableHostMoe,
    [switch] $DisableSparsePrefill,
    [switch] $SkipHardwareCheck,
    [switch] $CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion -lt [Version]'7.0') {
    throw 'PowerShell 7 or newer is required.'
}

$packageRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Server)) {
    $Server = Join-Path $packageRoot 'runtime\llama-server.exe'
}
$resolvedServer = (Resolve-Path -LiteralPath $Server).Path
$resolvedModel = (Resolve-Path -LiteralPath $Model).Path
$resolvedDraft = $null
if (-not $DisableMtp) {
    if ([string]::IsNullOrWhiteSpace($DraftModel)) {
        $DraftModel = & (Join-Path $packageRoot 'scripts\Get-Mtp.ps1')
    }
    $resolvedDraft = (Resolve-Path -LiteralPath $DraftModel).Path
}

& (Join-Path $packageRoot 'scripts\Test-Package.ps1') -Server $resolvedServer
& (Join-Path $packageRoot 'scripts\Test-ModelFiles.ps1') -ModelPath $resolvedModel -DraftModelPath $resolvedDraft

if (-not $SkipHardwareCheck) {
    $visibleExisted = Test-Path Env:GGML_VK_VISIBLE_DEVICES
    $visibleBefore = $env:GGML_VK_VISIBLE_DEVICES
    try {
        $env:GGML_VK_VISIBLE_DEVICES = [string]$GpuDeviceIndex
        $deviceList = (& $resolvedServer --list-devices 2>&1) -join [Environment]::NewLine
    }
    finally {
        if ($visibleExisted) { $env:GGML_VK_VISIBLE_DEVICES = $visibleBefore }
        else { Remove-Item Env:GGML_VK_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
    }
    if ($deviceList -notmatch 'Vulkan0:\s*AMD Radeon RX 7900 XTX') {
        throw "GPU fail-closed preflight failed for device index $GpuDeviceIndex. Runtime reported: $deviceList"
    }
}

$conflicts = @(Get-Process -Name 'llama-server','llama-cli','llama-perplexity','llama-bench' -ErrorAction SilentlyContinue)
if (-not $CheckOnly -and $conflicts.Count -gt 0) {
    $text = ($conflicts | Sort-Object Id -Unique | ForEach-Object { '{0}(PID {1})' -f $_.ProcessName, $_.Id }) -join ', '
    throw "Another inference process is running: $text"
}

$profile = [ordered]@{
    Model = $resolvedModel
    DraftModel = if ($DisableMtp) { 'disabled' } else { $resolvedDraft }
    Server = $resolvedServer
    Context = $Context
    Batch = $Batch
    UBatch = $UBatch
    DraftUBatch = [Math]::Min($UBatch, $DraftUBatch)
    ContextCheckpoints = $ContextCheckpoints
    HostMoe = -not $DisableHostMoe
    SparsePrefill = -not $DisableSparsePrefill
    Threads = $Threads
    CpuMoeLayers = $CpuMoeLayers
    TargetKV = "$CacheK/$CacheV"
    QsaKeyOnly = -not $DisableQsaKeyOnly
    Mtp = -not $DisableMtp
    DraftNMax = if ($DisableMtp) { 0 } else { $DraftNMax }
    DraftPMin = if ($DisableMtp) { 0 } else { $DraftPMin }
    Port = $Port
}
$profile.GetEnumerator() | Format-Table -AutoSize

if ($CacheK -ne 'bf16' -or $CacheV -ne 'bf16') {
    Write-Warning 'The selected target KV cache is outside the quality-accepted BF16 profile.'
}
if ($CheckOnly) {
    Write-Host 'Preflight passed; the server was not started.'
    return
}

$logDirectory = Join-Path $packageRoot 'local-results\server-logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory ('server-{0}.log' -f [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss'))

$arguments = @(
    '-m', $resolvedModel,
    '--host', '127.0.0.1', '--port', [string]$Port,
    '-np', '1', '-c', [string]$Context,
    '-b', [string]$Batch, '-ub', [string]$UBatch,
    '-t', [string]$Threads, '-tb', [string]$Threads,
    '--cache-type-k', $CacheK, '--cache-type-v', $CacheV,
    '--flash-attn', 'on', '--load-mode', 'mmap',
    '-ot', 'per_layer_token_embd\.weight=CPU',
    '-ncmoe', [string]$CpuMoeLayers, '--fit', 'off', '-ngl', '999',
    '--no-context-shift', '--cache-reuse', '0', '-ctxcp', [string]$ContextCheckpoints, '-cram', '0',
    '--metrics', '--props', '--offline', '--jinja',
    '--log-file', $logPath, '--log-colors', 'off', '--verbosity', '4'
)
$arguments += if ($DisablePromptCache) { '--no-cache-prompt' } else { '--cache-prompt' }
if ($DisableMtp) {
    $arguments += @('--spec-type', 'none')
} else {
    $arguments += @(
        '--spec-type', 'draft-mtp',
        '--spec-draft-model', $resolvedDraft,
        '--spec-draft-n-max', [string]$DraftNMax,
        '--spec-draft-ubatch', [string][Math]::Min($UBatch, $DraftUBatch),
        '--spec-draft-p-min', ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $DraftPMin)),
        '--spec-draft-type-k', $DraftCache,
        '--spec-draft-type-v', $DraftCache,
        '-ngld', '99',
        '--spec-draft-prio', '0',
        '--spec-draft-prio-batch', '0'
    )
}

$psi = [Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $resolvedServer
$psi.WorkingDirectory = Split-Path -Parent $resolvedServer
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
foreach ($argument in $arguments) {
    [void]$psi.ArgumentList.Add([string]$argument)
}
foreach ($name in @($psi.Environment.Keys)) {
    if ($name -match '^(?i:LLAMA_|GGML_)') {
        [void]$psi.Environment.Remove($name)
    }
}
$environment = [ordered]@{
    GGML_VK_VISIBLE_DEVICES = [string]$GpuDeviceIndex
    GGML_VK_ENABLE_MEMORY_PRIORITY = '1'
    LLAMA_MEMORY_PRIORITY = '5'
    LLAMA_PLE_PREFETCH = '1'
    LLAMA_QWEN4EXP_INDEXER_CACHE_TYPE = 'bf16'
    LLAMA_QSA_KEY_ONLY = if ($DisableQsaKeyOnly) { '0' } else { '1' }
    LLAMA_QSA_DENSE_BYPASS = '0'
    LLAMA_QSA_GATHER = '0'
    LLAMA_QSA_STRIDED_ADD = '0'
    LLAMA_SPEC_CKPT_ON_DEVICE = '0'
    LLAMA_HOTMOE = '0'
    LLAMA_HOSTMOE = if ($DisableHostMoe) { '0' } else { '1' }
    LLAMA_HOSTMOE_STAGE = '1'
    LLAMA_HOSTMOE_LAYERS = '4'
    LLAMA_HOSTMOE_MIN_TOK = '512'
    LLAMA_HOSTMOE_MAX_TOK = '4096'
    LLAMA_HOSTMOE_SHARED_CPU = '1'
    LLAMA_QWEN4EXP_MTP_SHARE_OUTPUT = if ($DisableMtp) { '0' } else { '1' }
    LLAMA_QWEN4EXP_MTP_CATCHUP_RESERVE = if ($DisableMtp) { '0' } else { '1' }
    LLAMA_QSA_PREFILL_GATHER = if ($DisableSparsePrefill) { '0' } else { '64' }
    GGML_VK_TOPK_HISTOGRAM_BANKS = '8'
    GGML_IQ4_NL_MMID_4ROW = '0'
}
foreach ($entry in $environment.GetEnumerator()) {
    $psi.Environment[$entry.Key] = $entry.Value
}

$process = [Diagnostics.Process]::new()
$process.StartInfo = $psi
$started = $false
try {
    $started = $process.Start()
    if (-not $started) {
        throw 'llama-server did not start.'
    }
    $baseUrl = 'http://127.0.0.1:{0}' -f $Port
    $deadline = [DateTime]::UtcNow.AddMinutes(5)
    do {
        if ($process.HasExited) {
            throw "llama-server exited during load with code $($process.ExitCode). See $logPath"
        }
        try {
            $health = Invoke-RestMethod -Method Get -Uri "$baseUrl/health" -TimeoutSec 2
            if ([string]$health.status -eq 'ok') {
                break
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    if ([DateTime]::UtcNow -ge $deadline) {
        throw "llama-server did not become healthy. See $logPath"
    }

    $nativeLog = Get-Content -LiteralPath $logPath -Raw
    $xtx = $nativeLog -match '(?:using device Vulkan0\s*\(AMD Radeon RX 7900 XTX\)|Vulkan0\s*:\s*AMD Radeon RX 7900 XTX)'
    $expectedBuffers = @{38 = 15595.52; 39 = 14633.02; 40 = 13670.52}
    $bufferMatch = [regex]::Match($nativeLog, 'Vulkan0 model buffer size\s*=\s*([\d.]+) MiB')
    $modelBuffer = $bufferMatch.Success
    if ($modelBuffer -and $expectedBuffers.ContainsKey($CpuMoeLayers)) {
        $actualBuffer = [double]::Parse($bufferMatch.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
        $modelBuffer = [Math]::Abs($actualBuffer - $expectedBuffers[$CpuMoeLayers]) -lt 0.1
    }
    if (-not $xtx -or -not $modelBuffer -or $nativeLog -match 'no usable GPU') {
        throw "GPU placement check failed for RX 7900 XTX and $CpuMoeLayers CPU expert layers. See $logPath"
    }
    Write-Host "GPU gate passed. Server ready at $baseUrl (PID $($process.Id))."
    Write-Host "Native log: $logPath"
    $process.WaitForExit()
    exit $process.ExitCode
}
finally {
    if ($started -and -not $process.HasExited) {
        $process.Kill($true)
        [void]$process.WaitForExit(15000)
    }
    $process.Dispose()
}
