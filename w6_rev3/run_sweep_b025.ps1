# Rev3 re-test at the operating point the W=50 curve actually prefers.
# The threshold sweep was flat, the budget sweep was not: budget=0.25 dominates
# budget=0.50 on both coordinates on slice o0 (e2e 0.590 vs 0.500, NLL 0.526 vs
# 0.770).  That could be one lucky slice, so the whole 8-slice set is re-run at
# 0.25 before any number is frozen.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$offsets = 0, 100, 200, 300, 400, 500, 600, 700
foreach ($o in $offsets) {
    $out = "w6run\R3b_o$($o)_W50_b025"
    $log = "..\w6_rev3\sweepb_o$($o).log"
    $err = "..\w6_rev3\sweepb_o$($o).err.log"
    if (Test-Path "$out\results_defer_W50.json") {
        Write-Host "[skip] o$o already done"
        continue
    }
    Write-Host "[run ] o$o -> $out"
    $a = @(
        "-u", "examples\w6_defer_stream.py",
        "--edits", "100", "--data-offset", "$o",
        "--defer-window", "50", "--layers", "4,5,6,7,8",
        "--refine-passes", "1", "--delta-budget", "0.25", "--guard-scale", "0.5",
        "--memory-read", "--memory-read-template", "instruct",
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] o$o exit=$($p.ExitCode)"
}
Write-Host "SWEEP B025 COMPLETE"
