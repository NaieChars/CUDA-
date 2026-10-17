# Shared build, case discovery, and statistics helpers for this experiment.
$ErrorActionPreference = 'Stop'
$script:ReductionCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Initialize-ReductionExperiment {
    param([string]$BuildDir = 'build', [switch]$SkipBuild)
    $root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
    $build = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
    if (-not $SkipBuild) {
        & cmake -S $root -B $build | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
        & cmake --build $build --config Release --target reduction --parallel | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }
    }
    $candidates = @(
        (Join-Path $build 'labs/reduction/Release/reduction.exe'),
        (Join-Path $build 'labs/reduction/reduction.exe')
    )
    $exe = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $exe) { throw 'Cannot find reduction.exe. Configure/build the reduction target first.' }
    return (Resolve-Path -LiteralPath $exe).Path
}

function Get-ReductionCases {
    param([string]$Exe, [int[]]$Modes, [long[]]$Sizes, [int[]]$Blocks, [int[]]$ItemsPerThread)
    $listing = @(& $Exe --list)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot query reduction variants.' }
    $variants = @{}
    foreach ($line in $listing) {
        if ($line -match '^Variant \| mode=(\d+) \| name=(\w+) \| kernel=(\w+) \| implemented=([01]) \| multiple_items=([01])$') {
            $variants[[int]$Matches[1]] = [pscustomobject]@{
                Mode = [int]$Matches[1]; Name = $Matches[2]; Kernel = $Matches[3]
                Implemented = $Matches[4] -eq '1'; MultipleItems = $Matches[5] -eq '1'
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
        $itemsList = if ($variant.MultipleItems) { @($ItemsPerThread | Select-Object -Unique) } else { @(1) }
        foreach ($n in ($Sizes | Select-Object -Unique)) {
            foreach ($block in ($Blocks | Select-Object -Unique)) {
                if ($block -lt 32 -or $block -gt 1024 -or ($block -band ($block - 1)) -ne 0) {
                    throw "Invalid block size: $block; use a power of two in [32, 1024]."
                }
                foreach ($items in $itemsList) {
                    [pscustomobject]@{
                        Name = "$($variant.Name)_n${n}_b${block}_i${items}"
                        Mode = $mode; Kernel = $variant.Kernel; N = $n
                        Block = $block; ItemsPerThread = $items
                    }
                }
            }
        }
    }
}

function New-ReductionResultDirectory {
    param([string]$Kind)
    $suffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
    $name = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + "_${Kind}_${suffix}"
    $directory = Join-Path $PSScriptRoot "../result/$name"
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    return (Resolve-Path -LiteralPath $directory).Path
}

function Save-ReductionManifest {
    param([string]$Directory, [string]$Exe, [string]$Stage,
          [int]$Repeat, [int]$Warmup, [int]$Iterations, $Cases, [string]$Metrics = '')
    $sources = @('main.cu', 'kernels.cu', 'kernels.cuh') | ForEach-Object {
        $path = Join-Path $PSScriptRoot "../src/$_"
        [pscustomobject]@{ File = $_; SHA256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
    }
    [pscustomobject]@{
        CreatedAt = [DateTimeOffset]::Now.ToString('o'); Exe = $Exe
        ExeSHA256 = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
        Sources = @($sources); Stage = $Stage; Repeat = $Repeat
        Warmup = $Warmup; Iterations = $Iterations; Cases = @($Cases); Metrics = $Metrics
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Directory 'manifest.json') -Encoding UTF8
}

function New-ReductionSample {
    param($Case, [string]$Stage, [int]$Run, [string]$Metric, [string]$Unit, [double]$Value)
    if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) { throw "Invalid sample: $Metric" }
    [pscustomobject]@{
        Case = $Case.Name; Mode = $Case.Mode; Kernel = $Case.Kernel
        Stage = $Stage; N = $Case.N; Block = $Case.Block; ItemsPerThread = $Case.ItemsPerThread
        Run = $Run; Metric = $Metric; Unit = $Unit
        Value = $Value.ToString('G17', $script:ReductionCulture)
    }
}

function Save-ReductionMeasurements {
    param($Samples, [string]$Prefix, [int]$ExpectedRepeat)
    $Samples | Export-Csv -LiteralPath "${Prefix}_samples.csv" -NoTypeInformation -Encoding UTF8
    $summary = @($Samples | Group-Object Case, Stage, Metric, Unit | ForEach-Object {
        $group = @($_.Group)
        $runs = @($group.Run | Sort-Object -Unique)
        if ($group.Count -ne $ExpectedRepeat -or $runs.Count -ne $ExpectedRepeat) {
            throw "Incomplete or duplicate samples for $($_.Name)."
        }
        $values = @($group | ForEach-Object { [double]::Parse($_.Value, $script:ReductionCulture) } | Sort-Object)
        $average = ($values | Measure-Object -Average).Average
        $middle = [int][math]::Floor($values.Count / 2)
        $median = if ($values.Count % 2) { $values[$middle] } else { ($values[$middle-1] + $values[$middle]) / 2 }
        $stdDev = if ($values.Count -gt 1) {
            [math]::Sqrt(($values | ForEach-Object { [math]::Pow($_ - $average, 2) } | Measure-Object -Sum).Sum / ($values.Count - 1))
        } else { $null }
        $first = $group[0]
        [pscustomobject]@{
            Case = $first.Case; Mode = $first.Mode; Kernel = $first.Kernel; Stage = $first.Stage
            N = $first.N; Block = $first.Block; ItemsPerThread = $first.ItemsPerThread
            Metric = $first.Metric; Unit = $first.Unit; Samples = $group.Count
            Average = $average.ToString('G17', $script:ReductionCulture)
            Median = $median.ToString('G17', $script:ReductionCulture)
            StdDev = if ($null -eq $stdDev) { '' } else { $stdDev.ToString('G17', $script:ReductionCulture) }
            Min = $values[0].ToString('G17', $script:ReductionCulture)
            Max = $values[-1].ToString('G17', $script:ReductionCulture)
        }
    })
    $summary | Export-Csv -LiteralPath "${Prefix}_average.csv" -NoTypeInformation -Encoding UTF8
    $summary | Format-Table Metric, Unit, Samples, Average, Median, StdDev, Min, Max -AutoSize |
        Out-String -Width 240 | Write-Host
    Write-Host "Summary: ${Prefix}_average.csv"
}
