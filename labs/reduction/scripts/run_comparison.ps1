param(
    [ValidateSet('complete', 'first_pass')][string]$Stage = 'complete',
    [ValidateRange(1, 100)][int]$Repeat = 10,
    [long]$N = 16777216,
    [int]$Block = 256,
    [int]$Warmup = 30,
    [int]$Iterations = 300,
    [switch]$IncludeItemsOne,
    [string]$Metrics,
    [ValidateSet('all', 'none')][string]$CacheControl = 'all',
    [ValidateSet('base', 'none')][string]$ClockControl = 'base',
    [string]$BuildDir = 'build'
)

# Compare unchanged kernels in rotating order; build explicitly before this script.
. (Join-Path $PSScriptRoot 'experiment_common.ps1')
$exe = Initialize-ReductionExperiment -BuildDir $BuildDir -SkipBuild
$cases = @(Get-ReductionCases -Exe $exe -Modes 0,1,2 -Sizes $N -Blocks $Block -ItemsPerThread 4)
if ($cases.Count -ne 3) { throw 'All three kernels must be implemented.' }
if ($IncludeItemsOne) {
    $cases += @(Get-ReductionCases -Exe $exe -Modes 1,2 -Sizes $N -Blocks $Block -ItemsPerThread 1)
}
if ($Repeat -lt 2) { Write-Warning 'A single run cannot estimate repeatability.' }
if ($Stage -eq 'first_pass' -and -not $Metrics) { throw 'Explicit metrics are required for first_pass.' }
$directory = New-ReductionResultDirectory -Kind ('comparison_' + $Stage)
$timedIterations = if ($Stage -eq 'complete') { $Iterations } else { 0 }
Save-ReductionManifest -Directory $directory -Exe $exe -Stage $Stage -Repeat $Repeat -Warmup $Warmup -Iterations $timedIterations -Cases $cases -Metrics $Metrics
[ordered]@{
    Order = 'cyclic rotation by run, starting with listed cases'; Stage = $Stage
    ReplayMode = 'kernel'; CacheControl = $CacheControl; ClockControl = $ClockControl
    Note = 'Profiler controls apply only to first_pass; ordinary timing uses natural clocks/caches.'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $directory 'measurement_policy.json') -Encoding UTF8
& nvidia-smi --query-gpu=name,driver_version,pstate,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,memory.total --format=csv |
    Set-Content -LiteralPath (Join-Path $directory 'gpu_before.csv') -Encoding UTF8
$allSamples = [System.Collections.Generic.List[object]]::new()
$order = [System.Collections.Generic.List[object]]::new()
for ($run = 1; $run -le $Repeat; ++$run) {
    for ($position = 0; $position -lt $cases.Count; ++$position) {
        $case = $cases[($position + $run - 1) % $cases.Count]
        $prefix = Join-Path $directory ('{0}_run{1:D2}' -f $case.Name, $run)
        $order.Add([pscustomobject]@{ Run = $run; Position = $position; Case = $case.Name })
        $programArgs = @($case.Mode, $case.N, $case.Block, $case.ItemsPerThread, '--warmup', $Warmup)
        if ($Stage -eq 'complete') {
            $programArgs += @('--iterations', $Iterations)
            $output = @(& $exe @programArgs 2>&1)
        } else {
            $ncuArgs = @('--replay-mode', 'kernel', '--cache-control', $CacheControl,
                '--clock-control', $ClockControl, '--profile-from-start', 'off',
                '--kernel-name-base', 'function', '--kernel-name', $case.Kernel,
                '--launch-count', '1', '--metrics', $Metrics, '--csv', '--page', 'details',
                '--print-metric-name', 'name', '--print-units', 'base', '--print-fp', '-o', $prefix)
            $output = @(& ncu @ncuArgs $exe @programArgs --profile 2>&1)
        }
        $code = $LASTEXITCODE
        $output | Set-Content -LiteralPath ($prefix + '_raw.log') -Encoding UTF8
        if ($code -ne 0) { throw "Measurement failed ($code): ${prefix}_raw.log" }
        $text = $output -join "`n"
        if ($text -notmatch 'Check \| phase=before \| status=PASS') { throw 'Missing validation PASS.' }
        if ($Stage -eq 'complete') {
            $result = @($output | Where-Object { "$_" -like 'Reduction |*' })
            if ($result.Count -ne 1 -or "$($result[0])" -notmatch 'time_ms=([0-9.eE+-]+) \| input_bandwidth_GBs=([0-9.eE+-]+) \| check=PASS$') { throw 'Invalid timing result.' }
            $time = [double]::Parse($Matches[1], $script:ReductionCulture)
            $bandwidth = [double]::Parse($Matches[2], $script:ReductionCulture)
            $allSamples.Add((New-ReductionSample $case $Stage $run 'time' 'ms' $time))
            $allSamples.Add((New-ReductionSample $case $Stage $run 'input_effective_bandwidth' 'GB/s' $bandwidth))
            Write-Host "run=$run | $($case.Name) | time_ms=$time"
        } else {
            if ($text -notmatch 'ProfileComplete \| stage=first_pass') { throw 'Missing profile completion.' }
            $csv = @($output | Where-Object { "$_" -match '^"(ID|[0-9]+)",' })
            $csv | Set-Content -LiteralPath ($prefix + '_metrics.csv') -Encoding UTF8
            $rows = @($csv | ConvertFrom-Csv)
            $requested = @($Metrics -split ',')
            if ($rows.Count -ne $requested.Count -or @($rows.ID | Sort-Object -Unique).Count -ne 1) { throw 'Unexpected metric or kernel count.' }
            foreach ($metric in $requested) {
                $row = @($rows | Where-Object { $_.'Metric Name' -eq $metric })
                if ($row.Count -ne 1) { throw "Missing/duplicate metric: $metric" }
                if ($row[0].'Kernel Name' -notmatch ('^' + [regex]::Escape($case.Kernel) + '(\(|$)')) { throw 'Wrong kernel collected.' }
                $value = [double]::Parse($row[0].'Metric Value', [System.Globalization.NumberStyles]::Float -bor [System.Globalization.NumberStyles]::AllowThousands, $script:ReductionCulture)
                $allSamples.Add((New-ReductionSample $case $Stage $run $metric $row[0].'Metric Unit' $value))
            }
            Write-Host "run=$run | $($case.Name) | $($rows.Count) metrics"
        }
    }
}
$order | Export-Csv -LiteralPath (Join-Path $directory 'execution_order.csv') -NoTypeInformation -Encoding UTF8
Save-ReductionMeasurements -Samples @($allSamples.ToArray()) -Prefix (Join-Path $directory 'comparison') -ExpectedRepeat $Repeat
& nvidia-smi --query-gpu=name,driver_version,pstate,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,memory.total --format=csv |
    Set-Content -LiteralPath (Join-Path $directory 'gpu_after.csv') -Encoding UTF8
Write-Host "COMPARISON_DIRECTORY=$directory"
