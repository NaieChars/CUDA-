param(
    [ValidateSet(0, 1, 2)][int[]]$Modes = @(0, 1, 2),
    [string[]]$Shapes = @('512x512x512'),
    [ValidateRange(1, 10000)][int]$Repeat = 3,
    [ValidateRange(0, 1000000)][int]$Warmup = 10,
    [string]$Metrics,
    [ValidateSet('base', 'none')][string]$ClockControl = 'none',
    [ValidateSet('all', 'none')][string]$CacheControl = 'all',
    [string]$NcuPath = 'ncu',
    [string]$BuildDir = 'build',
    [switch]$SkipBuild
)

. (Join-Path $PSScriptRoot 'experiment_common.ps1')
$exe = Initialize-GemmExperiment -BuildDir $BuildDir -SkipBuild:$SkipBuild
$cases = @(Get-GemmCases -Exe $exe -Modes $Modes -Shapes $Shapes)
if ($cases.Count -eq 0) {
    Write-Host 'No implemented kernels selected. No profiler reports generated.'
    return
}
$ncu = Get-Command $NcuPath -ErrorAction SilentlyContinue
if (-not $ncu) { throw "Cannot find Nsight Compute: $NcuPath" }
if (-not $Metrics) {
    $Metrics = @(
        'gpu__time_duration.sum',
        'sm__cycles_elapsed.avg.per_second', 'dram__cycles_elapsed.avg.per_second',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__bytes_op_read.sum', 'dram__bytes_op_write.sum',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
        'launch__registers_per_thread', 'launch__shared_mem_per_block_allocated',
        'launch__occupancy_limit_blocks', 'launch__occupancy_limit_registers',
        'launch__occupancy_limit_shared_mem', 'launch__occupancy_limit_warps',
        'smsp__inst_executed.sum',
        'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
        'smsp__warps_eligible.avg.per_cycle_active',
        'smsp__issue_active.avg.pct_of_peak_sustained_elapsed',
        'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio',
        'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
        'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'
    ) -join ','
}
$requested = @($Metrics -split ',' | ForEach-Object { $_.Trim() })
if (@($requested | Select-Object -Unique).Count -ne $requested.Count -or $requested -contains '') {
    throw 'Metrics must contain unique, nonempty exact names.'
}
$Metrics = $requested -join ','
$resultDir = New-GemmResultDirectory -Kind 'kernel'
$profiler = @{ ReplayMode = 'kernel'; CacheControl = $CacheControl; ClockControl = $ClockControl; NcuPath = $ncu.Source }
Save-GemmManifest -Directory $resultDir -Exe $exe -Stage 'kernel' -Repeat $Repeat -Warmup $Warmup -Iterations 0 -Cases $cases -Metrics $Metrics -Profiler $profiler
$samples = [System.Collections.Generic.List[object]]::new()
$order = [System.Collections.Generic.List[object]]::new()
for ($run = 1; $run -le $Repeat; ++$run) {
    for ($position = 0; $position -lt $cases.Count; ++$position) {
        $case = $cases[($position + $run - 1) % $cases.Count]
        $order.Add([pscustomobject]@{ Run = $run; Position = $position; Case = $case.Name })
        $prefix = Join-Path $resultDir ('{0}_run{1:D2}' -f $case.Name, $run)
        $ncuArgs = @(
            '--replay-mode', 'kernel', '--cache-control', $CacheControl, '--clock-control', $ClockControl,
            '--profile-from-start', 'off', '--kernel-name-base', 'function', '--kernel-name', $case.Kernel,
            '--launch-count', '1', '--metrics', $Metrics,
            '--csv', '--page', 'details', '--print-metric-name', 'name',
            '--print-units', 'base', '--print-fp', '-o', $prefix,
            $exe, $case.Mode, $case.M, $case.N, $case.K, '--warmup', $Warmup, '--profile'
        )
        $output = @(& $ncu.Source @ncuArgs 2>&1)
        $exitCode = $LASTEXITCODE
        $output | Set-Content -LiteralPath "${prefix}_ncu.log" -Encoding UTF8
        if ($exitCode -ne 0) { throw "NCU failed (exit $exitCode). See ${prefix}_ncu.log" }
        $text = $output -join "`n"
        if ($text -notmatch 'Check \| phase=before \| status=PASS' -or
            $text -notmatch 'Check \| phase=after \| status=PASS' -or
            $text -notmatch 'ProfileComplete \| stage=kernel') { throw 'Missing validation or profile completion.' }
        $csv = @($output | Where-Object { "$_" -match '^"(ID|[0-9]+)",' })
        $csv | Set-Content -LiteralPath "${prefix}_metrics.csv" -Encoding UTF8
        $rows = @($csv | ConvertFrom-Csv)
        if ($rows.Count -ne $requested.Count -or @($rows.ID | Sort-Object -Unique).Count -ne 1) {
            throw "Unexpected metric count or kernel count. See ${prefix}_ncu.log"
        }
        foreach ($metric in $requested) {
            $row = @($rows | Where-Object { $_.'Metric Name' -eq $metric })
            if ($row.Count -ne 1) { throw "Missing or duplicate metric: $metric" }
            if ($row[0].'Kernel Name' -notmatch ('^' + [regex]::Escape($case.Kernel) + '(\(|$)')) {
                throw "Unexpected profiled kernel: $($row[0].'Kernel Name')"
            }
            $value = [double]::Parse($row[0].'Metric Value',
                [System.Globalization.NumberStyles]::Float -bor [System.Globalization.NumberStyles]::AllowThousands,
                $script:GemmCulture)
            $samples.Add((New-GemmSample $case 'kernel' $run $metric $row[0].'Metric Unit' $value))
        }
        Write-Host "run=$run | $($case.Name) | $($rows.Count) metrics"
    }
}
$order | Export-Csv -LiteralPath (Join-Path $resultDir 'execution_order.csv') -NoTypeInformation -Encoding UTF8
Save-GemmMeasurements -Samples @($samples.ToArray()) -Prefix (Join-Path $resultDir 'measurements') -ExpectedRepeat $Repeat
Write-Host "Reports: $resultDir"
Write-Host 'One GEMM launch is profiled. Compare profiling and ordinary timing with their cache/clock policies recorded.'
