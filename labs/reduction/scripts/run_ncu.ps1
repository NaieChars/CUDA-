param(
    [ValidateSet(0, 1, 2)][int[]]$Modes = @(0, 1, 2),
    [ValidateScript({ $_ -gt 0 })][long[]]$Sizes = @(16777216),
    [int[]]$Blocks = @(256),
    [ValidateRange(1, 64)][int[]]$ItemsPerThread = @(4),
    [ValidateRange(1, 10000)][int]$Repeat = 10,
    [ValidateRange(0, 1000000)][int]$Warmup = 10,
    [string]$Metrics,
    [string]$NcuPath = 'ncu',
    [string]$BuildDir = 'build',
    [switch]$SkipBuild
)

. (Join-Path $PSScriptRoot 'experiment_common.ps1')
$exe = Initialize-ReductionExperiment -BuildDir $BuildDir -SkipBuild:$SkipBuild
$cases = @(Get-ReductionCases -Exe $exe -Modes $Modes -Sizes $Sizes -Blocks $Blocks -ItemsPerThread $ItemsPerThread)
if ($cases.Count -eq 0) {
    Write-Host 'No implemented kernels selected. No profiler reports generated.'
    return
}
$ncu = Get-Command $NcuPath -ErrorAction SilentlyContinue
if (-not $ncu) { throw "Cannot find Nsight Compute: $NcuPath" }
if (-not $Metrics) {
    $Metrics = @(
        'gpu__time_duration.sum',
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
$resultDir = New-ReductionResultDirectory -Kind 'first_pass'
Save-ReductionManifest -Directory $resultDir -Exe $exe -Stage 'first_pass' -Repeat $Repeat -Warmup $Warmup -Iterations 0 -Cases $cases -Metrics $Metrics

foreach ($case in $cases) {
    Write-Host "`n--- $($case.Name) | stage=first_pass ---" -ForegroundColor Cyan
    $samples = @(foreach ($run in 1..$Repeat) {
        $prefix = Join-Path $resultDir ('{0}_run{1:D2}' -f $case.Name, $run)
        Write-Host "Run $run/$Repeat -> ${prefix}.ncu-rep"
        $ncuArgs = @(
            '--profile-from-start', 'off',
            '--kernel-name-base', 'function', '--kernel-name', $case.Kernel,
            '--launch-count', '1', '--metrics', $Metrics,
            '--csv', '--page', 'details', '--print-metric-name', 'name',
            '--print-units', 'base', '--print-fp', '-o', $prefix,
            $exe, $case.Mode, $case.N, $case.Block, $case.ItemsPerThread,
            '--warmup', $Warmup, '--profile'
        )
        $output = @(& $ncu.Source @ncuArgs 2>&1)
        $exitCode = $LASTEXITCODE
        $output | Set-Content -LiteralPath "${prefix}_ncu.log" -Encoding UTF8
        if ($exitCode -ne 0) { throw "NCU failed (exit $exitCode). See ${prefix}_ncu.log" }
        $text = $output -join "`n"
        if ($text -notmatch 'Check \| phase=before \| status=PASS' -or
            $text -notmatch 'ProfileComplete \| stage=first_pass') { throw 'Missing validation or profile completion marker.' }
        $csv = @($output | Where-Object { "$_" -match '^"(ID|[0-9]+)",' })
        $csv | Set-Content -LiteralPath "${prefix}_metrics.csv" -Encoding UTF8
        $rows = @($csv | ConvertFrom-Csv)
        if ($rows.Count -eq 0 -or @($rows.ID | Sort-Object -Unique).Count -ne 1) {
            throw "Expected metrics for exactly one first-pass kernel. See ${prefix}_ncu.log"
        }
        foreach ($row in $rows) {
            if ($row.'Kernel Name' -notmatch ('^' + [regex]::Escape($case.Kernel) + '(\(|$)')) {
                throw "Unexpected profiled kernel: $($row.'Kernel Name')"
            }
            $value = 0.0
            if (-not [double]::TryParse($row.'Metric Value',
                [System.Globalization.NumberStyles]::Float -bor [System.Globalization.NumberStyles]::AllowThousands,
                $script:ReductionCulture, [ref]$value)) { throw "Invalid metric: $($row.'Metric Name')" }
            New-ReductionSample $case 'first_pass' $run $row.'Metric Name' $row.'Metric Unit' $value
        }
    })
    Save-ReductionMeasurements -Samples $samples -Prefix (Join-Path $resultDir "$($case.Name)_ncu") -ExpectedRepeat $Repeat
}
Write-Host "`nReports: $resultDir"
Write-Host 'These metrics cover only the first pass. Use run_benchmark.ps1 for complete-reduction time.'
