param(
    [ValidateSet(0, 1, 2)][int[]]$Modes = @(0, 1, 2),
    [string[]]$Shapes = @('512x512x512'),
    [ValidateRange(1, 10000)][int]$Repeat = 10,
    [ValidateRange(0, 1000000)][int]$Warmup = 10,
    [ValidateRange(1, 1000000)][int]$Iterations = 100,
    [switch]$CheckOnly,
    [string]$BuildDir = 'build',
    [switch]$SkipBuild
)

. (Join-Path $PSScriptRoot 'experiment_common.ps1')
$exe = Initialize-GemmExperiment -BuildDir $BuildDir -SkipBuild:$SkipBuild
$cases = @(Get-GemmCases -Exe $exe -Modes $Modes -Shapes $Shapes)
if ($cases.Count -eq 0) {
    Write-Host 'No implemented kernels selected. No performance data generated.'
    return
}
$stage = if ($CheckOnly) { 'check' } else { 'complete' }
$timedIterations = if ($CheckOnly) { 0 } else { $Iterations }
$resultDir = New-GemmResultDirectory -Kind $stage
Save-GemmManifest -Directory $resultDir -Exe $exe -Stage $stage -Repeat $Repeat -Warmup $Warmup -Iterations $timedIterations -Cases $cases
$samples = [System.Collections.Generic.List[object]]::new()
$order = [System.Collections.Generic.List[object]]::new()
for ($run = 1; $run -le $Repeat; ++$run) {
    for ($position = 0; $position -lt $cases.Count; ++$position) {
        $case = $cases[($position + $run - 1) % $cases.Count]
        $order.Add([pscustomobject]@{ Run = $run; Position = $position; Case = $case.Name })
        $programArgs = @($case.Mode, $case.M, $case.N, $case.K, '--warmup', $Warmup, '--iterations', $Iterations)
        if ($CheckOnly) { $programArgs += '--check-only' }
        $output = @(& $exe @programArgs 2>&1)
        $exitCode = $LASTEXITCODE
        $prefix = Join-Path $resultDir ('{0}_run{1:D2}' -f $case.Name, $run)
        $output | Set-Content -LiteralPath "${prefix}_normal.log" -Encoding UTF8
        if ($exitCode -ne 0) { throw "Run failed (exit $exitCode). See ${prefix}_normal.log" }
        $text = $output -join "`n"
        if ($text -notmatch 'Check \| phase=before \| status=PASS') { throw 'Missing validation result.' }
        if ($CheckOnly) {
            $samples.Add((New-GemmSample $case $stage $run 'validation_pass' 'bool' 1))
            Write-Host "run=$run | $($case.Name) | validation PASS"
            continue
        }
        if ($text -notmatch 'Check \| phase=after \| status=PASS') { throw 'Missing final validation.' }
        $result = @($output | Where-Object { "$_" -like 'GEMM |*' })
        if ($result.Count -ne 1 -or "$($result[0])" -notmatch 'time_ms=([0-9.eE+-]+) \| gflops=([0-9.eE+-]+) \| check=PASS$') {
            throw "Cannot parse benchmark result. See ${prefix}_normal.log"
        }
        $time = [double]::Parse($Matches[1], $script:GemmCulture)
        $gflops = [double]::Parse($Matches[2], $script:GemmCulture)
        if ($time -le 0 -or $gflops -le 0) { throw 'Nonpositive benchmark result.' }
        $samples.Add((New-GemmSample $case $stage $run 'time' 'ms' $time))
        $samples.Add((New-GemmSample $case $stage $run 'throughput' 'GFLOP/s' $gflops))
        Write-Host "run=$run | $($result[0])"
    }
}
$order | Export-Csv -LiteralPath (Join-Path $resultDir 'execution_order.csv') -NoTypeInformation -Encoding UTF8
Save-GemmMeasurements -Samples @($samples.ToArray()) -Prefix (Join-Path $resultDir 'measurements') -ExpectedRepeat $Repeat
Write-Host "Results: $resultDir"
