# Rev4 param-channel probe: two levers on the 0.369 number.
#
# The parameter channel is not limited by write accuracy but by admission rate
# (budget 0.25 admits roughly a third of edits; slice o200 admits 1 row) and by
# within-window interference (W=50 sequential writes overwrite each other).
# Rev3 only tested budget 0.25 vs 0.50 -- an interior point was never tried, and
# the window length was never varied at the operating budget.
#
# Arm A: delta-budget 0.15 (fewer, cleaner writes)      W=50
# Arm B: delta-budget 0.35 (more writes, more damage)   W=50
# Arm C: delta-budget 0.25, defer-window 25            (less interference)
#
# Everything else is held identical to R6 (the frozen Rev4 headline run) so the
# three arms are comparable slice-by-slice against it.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$offsets = 0, 100, 200, 300, 400, 500, 600, 700

$arms = @(
    @{ tag = "b015"; budget = "0.15"; W = "50" },
    @{ tag = "b035"; budget = "0.35"; W = "50" },
    @{ tag = "W25b025"; budget = "0.25"; W = "25" }
)

foreach ($arm in $arms) {
    foreach ($o in $offsets) {
        $tag = $arm.tag
        $W = $arm.W
        $out = "w6run\R11_o$($o)_W$($W)_$($tag)_inject"
        $log = "$repo\w6_rev3\r11_$($tag)_o$($o).log"
        $err = "$repo\w6_rev3\r11_$($tag)_o$($o).err.log"
        if (Test-Path "$out\results_defer_W$($W).json") {
            Write-Host "[skip] $tag o$o already done"
            continue
        }
        Write-Host "[run ] $tag o$o -> $out"
        $a = @(
            "-u", "examples\w6_defer_stream.py",
            "--edits", "100", "--data-offset", "$o",
            "--defer-window", "$W", "--layers", "4,5,6,7,8",
            "--refine-passes", "1", "--delta-budget", $arm.budget, "--guard-scale", "0.5",
            "--memory-read", "--memory-read-template", "instruct",
            "--eval-paraphrase",
            "--out-dir", $out
        )
        $t0 = Get-Date
        & $py @a > $log 2> $err
        $rc = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        Write-Host "[done] $tag o$o exit=$rc ${mins}min"
    }
}
Write-Host "PARAM PROBE COMPLETE"
