# ============================================================
# run_ncu.ps1
# Phase 1: 普通检测 10 次，分别平均耗时和有效带宽
# Phase 2: ncu 检测 10 次，保留原始报告并分别平均各指标
# ============================================================

$ErrorActionPreference = "Stop"
$repeat = 10
$culture = [System.Globalization.CultureInfo]::InvariantCulture

# ---------- 路径（相对脚本所在目录自动推导） ----------
$scriptDir = $PSScriptRoot
$exe       = Join-Path $scriptDir "..\..\..\build\labs\day04_coalescing\Release\day04_coalescing.exe"
$resultDir = Join-Path $scriptDir ("..\result\" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))

# ---------- 要测试的组合 ----------
# 每行格式: @(row, col, block)
$cases = @(
    #@{row=1024; col=1024; block=256}
    #@{row=2048; col=2048; block=256}
    @{row=4096; col=4096; block=256}
)

# mode 对应 main.cu；Tiling 的 block 尺寸由程序固定。
$kernels = @(
    @{mode=0; name="copy"; kernel="normal_copy"; block=$null}
    @{mode=2; name="transpose_naive"; kernel="transpose_matrix"; block=$null}
    @{mode=3; name="transpose_tiling"; kernel="transpose_matrix_tiling"; block="32x32"}
    @{mode=4; name="transpose_tiling_loop"; kernel="transpose_matrix_tiling_32x8"; block="32x8"}
)
$tests = @(foreach ($case in $cases) {
    foreach ($kernel in $kernels) {
        $blockName = if ($kernel.block) { $kernel.block } else { $case.block }
        [pscustomobject]@{
            Name = "$($kernel.name)_$($case.row)_$($case.col)_b${blockName}"
            Kernel = $kernel.kernel
            Args = @($kernel.mode, 1, $case.row, $case.col, $case.block)
        }
    }
})

# ---------- ncu 指标 ----------
$metrics = @(
    # ===== 基础时间与吞吐 =====
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",

    # ===== Cache 命中率 =====
    "l1tex__t_sector_hit_rate",    # 10/3 21:06 新增Cache命中率指标
    "lts__t_sector_hit_rate",

    # ===== 全局内存合并情况 =====
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",

    # ===== Occupancy 受限原因 =====
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_warps",    # 10/3 17:10 新增
    "launch__shared_mem_config_size",   # 10/3 17:10 新增

    # ===== 调度器与 Warp Stall =====
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__inst_executed.sum",
    "smsp__issue_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",

    # ===== Shared Memory Bank Conflict =====
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
) -join ","

# ---------- 准备目录 ----------
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Executable not found: $exe. Build day04_coalescing in Release first."
}
if (-not (Get-Command ncu -ErrorAction SilentlyContinue)) {
    throw "Cannot find ncu. Add Nsight Compute to PATH."
}
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

# 同一 kernel、指标、单位独立求算术平均，不把缺失/无效值当作 0。
function Save-Measurements($samples, $prefix) {
    $samples | Export-Csv "${prefix}_samples.csv" -NoTypeInformation -Encoding UTF8
    $averages = @($samples | Group-Object Kernel, Metric, Unit | ForEach-Object {
        $group = $_.Group
        if ($group.Count -ne $repeat -or @($group.Run | Sort-Object -Unique).Count -ne $repeat) {
            throw "Expected $repeat distinct samples for $($_.Name), got $($group.Count)."
        }
        [pscustomobject]@{
            Kernel = $group[0].Kernel
            Metric = $group[0].Metric
            Unit = $group[0].Unit
            Samples = $group.Count
            Average = ($group | Measure-Object Value -Average).Average.ToString("G17", $culture)
        }
    })
    $averages | Export-Csv "${prefix}_average.csv" -NoTypeInformation -Encoding UTF8
    $averages | Format-Table Metric, Unit, Samples, Average -AutoSize | Out-String -Width 200 | Write-Host
    Write-Host "Averages: ${prefix}_average.csv"
}

# ============================================================
# Phase 1: 普通跑（无 ncu），拿真实耗时与有效带宽
# ============================================================
Write-Host ""
Write-Host "############################################################"
Write-Host "# Phase 1: Normal run (no ncu) - real time and bandwidth"
Write-Host "############################################################"

foreach ($test in $tests) {
    Write-Host "`n--- $($test.Name) ---"
    $programArgs = $test.Args
    $samples = @(foreach ($run in 1..$repeat) {
        $output = @(& $exe @programArgs)
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $($test.Name), run $run." }
        Write-Host "Run $run/$repeat : $output"
        if (($output -join "`n") -notmatch 'time=([0-9.]+) ms.*bandwidth=([0-9.]+) GB/s') {
            throw "Cannot parse benchmark output: $output"
        }
        $time = [double]::Parse($Matches[1], $culture)
        $bandwidth = [double]::Parse($Matches[2], $culture)
        [pscustomobject]@{Run=$run; Kernel=$test.Kernel; Metric="time"; Unit="ms"; Value=$time}
        [pscustomobject]@{Run=$run; Kernel=$test.Kernel; Metric="effective_bandwidth"; Unit="GB/s"; Value=$bandwidth}
    })
    Save-Measurements $samples (Join-Path $resultDir "$($test.Name)_normal")
}

# ============================================================
# Phase 2: ncu 跑，生成 .ncu-rep
# ============================================================
Write-Host ""
Write-Host "############################################################"
Write-Host "# Phase 2: ncu profiling - generating .ncu-rep reports"
Write-Host "############################################################"

foreach ($test in $tests) {
    Write-Host "`n--- Profiling $($test.Name) ---"
    $programArgs = $test.Args
    $samples = @(foreach ($run in 1..$repeat) {
        $out = Join-Path $resultDir ("{0}_run{1:D2}" -f $test.Name, $run)
        Write-Host "Run $run/$repeat -> $out.ncu-rep"
        $output = @(& ncu --metrics $metrics `
            --kernel-name-base function --kernel-name $test.Kernel `
            --launch-skip 10 --launch-count 1 `
            --csv --page details --print-metric-name name --print-units base --print-fp `
            -o $out $exe @programArgs 2>&1)
        $exitCode = $LASTEXITCODE
        $output | Set-Content "${out}_ncu.log" -Encoding UTF8
        if ($exitCode -ne 0) { throw "NCU failed (exit $exitCode). See ${out}_ncu.log." }

        # 排除程序输出及 profiler 日志，仅解析 CSV 表头和数据行。
        $csv = @($output | Where-Object { $_ -match '^"(ID|[0-9]+)",' })
        $csv | Set-Content "${out}_metrics.csv" -Encoding UTF8
        $rows = @($csv | ConvertFrom-Csv)
        if ($rows.Count -eq 0) { throw "No NCU metrics collected: $out" }
        foreach ($row in $rows) {
            $value = 0.0
            if (-not [double]::TryParse($row.'Metric Value', [System.Globalization.NumberStyles]::Float -bor [System.Globalization.NumberStyles]::AllowThousands, $culture, [ref]$value) -or
                [double]::IsNaN($value) -or [double]::IsInfinity($value)) {
                throw "Invalid NCU value for $($row.'Metric Name'): $($row.'Metric Value')"
            }
            [pscustomobject]@{
                Run = $run
                Kernel = $row.'Kernel Name'
                Metric = $row.'Metric Name'
                Unit = $row.'Metric Unit'
                Value = $value
            }
        }
    })
    Save-Measurements $samples (Join-Path $resultDir "$($test.Name)_ncu")
}

Write-Host ""
Write-Host "All done."
Write-Host "Reports: $resultDir"
Write-Host "Open with: ncu-ui <report>.ncu-rep"
