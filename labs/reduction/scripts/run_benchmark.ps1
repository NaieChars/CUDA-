param(
    [ValidateSet(0, 1, 2)][int[]]$Modes = @(0, 1, 2),
    [ValidateScript({ $_ -gt 0 })][long[]]$Sizes = @(16777216),
    [int[]]$Blocks = @(256),
    [ValidateRange(1, 64)][int[]]$ItemsPerThread = @(4),
    [ValidateRange(1, 10000)][int]$Repeat = 10,
    [ValidateRange(0, 1000000)][int]$Warmup = 10,
    [ValidateRange(1, 1000000)][int]$Iterations = 100,
    [switch]$CheckOnly,
    [string]$BuildDir = 'build',
    [switch]$SkipBuild
)

. (Join-Path $PSScriptRoot 'experiment_common.ps1')
$exe = Initialize-ReductionExperiment -BuildDir $BuildDir -SkipBuild:$SkipBuild
$cases = @(Get-ReductionCases -Exe $exe -Modes $Modes -Sizes $Sizes -Blocks $Blocks -ItemsPerThread $ItemsPerThread)
if ($cases.Count -eq 0) {
    Write-Host 'No implemented kernels selected. No performance data generated.'
    return
}
$stage = if ($CheckOnly) { 'check' } else { 'complete' }
$resultDir = New-ReductionResultDirectory -Kind $stage
Save-ReductionManifest -Directory $resultDir -Exe $exe -Stage $stage -Repeat $Repeat -Warmup $Warmup -Iterations $Iterations -Cases $cases

foreach ($case in $cases) {
    Write-Host "`n--- $($case.Name) | stage=$stage ---" -ForegroundColor Cyan
    $samples = @(foreach ($run in 1..$Repeat) {
        $programArgs = @($case.Mode, $case.N, $case.Block, $case.ItemsPerThread,
                         '--warmup', $Warmup, '--iterations', $Iterations)
        if ($CheckOnly) { $programArgs += '--check-only' }
        $output = @(& $exe @programArgs 2>&1)
        $exitCode = $LASTEXITCODE
        $prefix = Join-Path $resultDir ('{0}_run{1:D2}' -f $case.Name, $run)
        $output | Set-Content -LiteralPath "${prefix}_normal.log" -Encoding UTF8
        $text = $output -join "`n"
        if ($exitCode -ne 0) { throw "Run failed (exit $exitCode). See ${prefix}_normal.log" }
        if ($CheckOnly) {
            if ($text -notmatch 'Check \| phase=before \| status=PASS') { throw 'Missing validation result.' }
            Write-Host "Run $run/$Repeat : validation PASS"
            New-ReductionSample $case $stage $run 'validation_pass' 'bool' 1
            continue
        }
        $result = @($output | Where-Object { "$_" -like 'Reduction |*' })
        if ($result.Count -ne 1 -or "$($result[0])" -notmatch 'time_ms=([0-9.eE+-]+) \| input_bandwidth_GBs=([0-9.eE+-]+) \| check=PASS$') {
            throw "Cannot parse benchmark result. See ${prefix}_normal.log"
        }
        $time = [double]::Parse($Matches[1], $script:ReductionCulture)
        $bandwidth = [double]::Parse($Matches[2], $script:ReductionCulture)
        if ($time -le 0 -or $bandwidth -le 0) { throw 'Nonpositive benchmark result.' }
        Write-Host "Run $run/$Repeat : $($result[0])"
        New-ReductionSample $case $stage $run 'time' 'ms' $time
        New-ReductionSample $case $stage $run 'input_effective_bandwidth' 'GB/s' $bandwidth
    })
    Save-ReductionMeasurements -Samples $samples -Prefix (Join-Path $resultDir "$($case.Name)_normal") -ExpectedRepeat $Repeat
}
Write-Host "`nResults: $resultDir"
