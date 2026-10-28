# Adapted from labs/reduction/scripts/experiment_common.ps1.
$ErrorActionPreference = 'Stop'
$script:GemmCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Initialize-GemmExperiment {
    param([string]$BuildDir = 'build', [switch]$SkipBuild)
    $root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
    $build = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
    # Keep CUDA runtime lookup local to this PowerShell process.
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $cudaBin = Split-Path $nvcc.Source
        $env:PATH = $cudaBin + ';' + $env:PATH
    }
    if (-not $SkipBuild) {
        & cmake -S $root -B $build | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
        & cmake --build $build --config Release --target sgemm --parallel 1 | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }
    }
    $candidates = @(
        (Join-Path $build 'labs/sgemm/Release/sgemm.exe'),
        (Join-Path $build 'labs/sgemm/sgemm.exe')
    )
    $exe = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $exe) { throw 'Cannot find sgemm.exe. Configure/build the sgemm target first.' }
    return (Resolve-Path -LiteralPath $exe).Path
}

function Get-GemmCases {
    param([string]$Exe, [int[]]$Modes, [string[]]$Shapes)
    if ($Modes.Count -eq 0 -or $Shapes.Count -eq 0) { throw 'Modes and Shapes must not be empty.' }
    $parsedShapes = @(foreach ($shape in ($Shapes | Select-Object -Unique)) {
        if ($shape -notmatch '^([1-9][0-9]*)[xX]([1-9][0-9]*)[xX]([1-9][0-9]*)$') {
            throw "Invalid shape: $shape; expected MxNxK, for example 127x193x65."
        }
        [pscustomobject]@{ M = [long]$Matches[1]; N = [long]$Matches[2]; K = [long]$Matches[3] }
    })
    $listing = @(& $Exe --list)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot query GEMM variants.' }
    $variants = @{}
    foreach ($line in $listing) {
        if ($line -match '^Variant \| mode=(\d+) \| name=(\w+) \| kernel=(\w+) \| implemented=([01]) \| block_x=(\d+) \| block_y=(\d+) \| tile_m=(\d+) \| tile_n=(\d+) \| tile_k=(\d+) \| thread_m=(\d+) \| thread_n=(\d+) \| shared_bytes=(\d+)$') {
            $variants[[int]$Matches[1]] = [pscustomobject]@{
                Mode = [int]$Matches[1]; Name = $Matches[2]; Kernel = $Matches[3]
                Implemented = $Matches[4] -eq '1'
                BlockX = [int]$Matches[5]; BlockY = [int]$Matches[6]
                TileM = [int]$Matches[7]; TileN = [int]$Matches[8]; TileK = [int]$Matches[9]
                ThreadM = [int]$Matches[10]; ThreadN = [int]$Matches[11]; SharedBytes = [long]$Matches[12]
            }
        }
    }
    foreach ($mode in ($Modes | Select-Object -Unique)) {
        if (-not $variants.ContainsKey($mode)) { throw "Unknown mode: $mode" }
        $variant = $variants[$mode]
        if (-not $variant.Implemented) {
            Write-Host "SKIP mode=$mode ($($variant.Name)): TODO in kernels.cu." -ForegroundColor Yellow
            continue
        }
        foreach ($shape in $parsedShapes) {
            [pscustomobject]@{
                Name = "$($variant.Name)_m$($shape.M)_n$($shape.N)_k$($shape.K)"
                Mode = $mode; Kernel = $variant.Kernel
                M = $shape.M; N = $shape.N; K = $shape.K
                BlockX = $variant.BlockX; BlockY = $variant.BlockY
                TileM = $variant.TileM; TileN = $variant.TileN; TileK = $variant.TileK
                ThreadM = $variant.ThreadM; ThreadN = $variant.ThreadN; SharedBytes = $variant.SharedBytes
            }
        }
    }
}

function New-GemmResultDirectory {
    param([string]$Kind)
    $suffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
    $name = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + "_${Kind}_${suffix}"
    $directory = Join-Path $PSScriptRoot "../result/$name"
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    return (Resolve-Path -LiteralPath $directory).Path
}

function Save-GemmManifest {
    param([string]$Directory, [string]$Exe, [string]$Stage,
          [int]$Repeat, [int]$Warmup, [int]$Iterations, $Cases,
          [string]$Metrics = '', [hashtable]$Profiler = @{})
    $sources = @('main.cu', 'kernels.cu', 'kernels.cuh') | ForEach-Object {
        $path = Join-Path $PSScriptRoot "../src/$_"
        [pscustomobject]@{ File = $_; SHA256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
    }
    [pscustomobject]@{
        CreatedAt = [DateTimeOffset]::Now.ToString('o'); Exe = $Exe
        ExeSHA256 = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
        Sources = @($sources); Stage = $Stage; Repeat = $Repeat
        Warmup = $Warmup; Iterations = $Iterations; Cases = @($Cases); Metrics = $Metrics
        Validation = 'full_cpu_double'; Atol = 1e-4; Rtol = 1e-4
        Order = 'cyclic rotation by run'; Profiler = $Profiler
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Directory 'manifest.json') -Encoding UTF8
}

function New-GemmSample {
    param($Case, [string]$Stage, [int]$Run, [string]$Metric, [string]$Unit, [double]$Value)
    if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) { throw "Invalid metric: $Metric" }
    [pscustomobject]@{
        Case = $Case.Name; Mode = $Case.Mode; Kernel = $Case.Kernel; Stage = $Stage
        M = $Case.M; N = $Case.N; K = $Case.K
        BlockX = $Case.BlockX; BlockY = $Case.BlockY
        TileM = $Case.TileM; TileN = $Case.TileN; TileK = $Case.TileK
        ThreadM = $Case.ThreadM; ThreadN = $Case.ThreadN
        Run = $Run; Metric = $Metric; Unit = $Unit
        Value = $Value.ToString('G17', $script:GemmCulture)
    }
}

function Save-GemmMeasurements {
    param($Samples, [string]$Prefix, [int]$ExpectedRepeat)
    if ($Samples.Count -eq 0) { throw 'Cannot summarize empty samples.' }
    $Samples | Export-Csv -LiteralPath "${Prefix}_samples.csv" -NoTypeInformation -Encoding UTF8
    $summary = @($Samples | Group-Object Case, Stage, Metric, Unit | ForEach-Object {
        $group = @($_.Group)
        $runs = @($group.Run | Sort-Object -Unique)
        if ($group.Count -ne $ExpectedRepeat -or $runs.Count -ne $ExpectedRepeat) {
            throw "Incomplete or duplicate samples for $($_.Name)."
        }
        $values = @($group | ForEach-Object { [double]::Parse($_.Value, $script:GemmCulture) } | Sort-Object)
        $average = ($values | Measure-Object -Average).Average
        $middle = [int][math]::Floor($values.Count / 2)
        $median = if ($values.Count % 2) { $values[$middle] } else { ($values[$middle-1] + $values[$middle]) / 2 }
        $stdDev = if ($values.Count -gt 1) {
            [math]::Sqrt(($values | ForEach-Object { [math]::Pow($_ - $average, 2) } | Measure-Object -Sum).Sum / ($values.Count - 1))
        } else { $null }
        $first = $group[0]
        [pscustomobject]@{
            Case = $first.Case; Mode = $first.Mode; Kernel = $first.Kernel; Stage = $first.Stage
            M = $first.M; N = $first.N; K = $first.K
            BlockX = $first.BlockX; BlockY = $first.BlockY
            TileM = $first.TileM; TileN = $first.TileN; TileK = $first.TileK
            ThreadM = $first.ThreadM; ThreadN = $first.ThreadN
            Metric = $first.Metric; Unit = $first.Unit; Samples = $group.Count
            Average = $average.ToString('G17', $script:GemmCulture)
            Median = $median.ToString('G17', $script:GemmCulture)
            StdDev = if ($null -eq $stdDev) { '' } else { $stdDev.ToString('G17', $script:GemmCulture) }
            Min = $values[0].ToString('G17', $script:GemmCulture)
            Max = $values[-1].ToString('G17', $script:GemmCulture)
        }
    })
    $summary | Export-Csv -LiteralPath "${Prefix}_average.csv" -NoTypeInformation -Encoding UTF8
    $summary | Format-Table Metric, Unit, Samples, Average, Median, StdDev, Min, Max -AutoSize |
        Out-String -Width 240 | Write-Host
    Write-Host "Summary: ${Prefix}_average.csv"
}
