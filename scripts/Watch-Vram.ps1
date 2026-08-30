[CmdletBinding()]
param(
    [ValidateRange(1, 60)]
    [int] $IntervalSeconds = 2
)

while ($true) {
    $process = Get-Process -Name 'llama-server' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $process) {
        Write-Host 'llama-server is not running.'
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    $samples = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
        Where-Object InstanceName -Like "pid_$($process.Id)_*"
    $dedicated = ($samples | Measure-Object CookedValue -Sum).Sum
    [pscustomobject]@{
        Time = Get-Date -Format 'HH:mm:ss'
        PID = $process.Id
        DedicatedVRAMGiB = [math]::Round($dedicated / 1GB, 2)
        WorkingSetGiB = [math]::Round($process.WorkingSet64 / 1GB, 2)
    } | Format-Table -AutoSize
    Start-Sleep -Seconds $IntervalSeconds
}

