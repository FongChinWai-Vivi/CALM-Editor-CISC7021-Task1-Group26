# W6 Rev3 Day-2 sweep: memory-in-loop end-to-end over the full 8 held-out slices.
# Each slice is 100 edits at W=50, offset o0..o700, identical to the Day-1 smoke
# except for the data offset, so the slices are directly poolable.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$offsets = 0, 100, 200, 300, 400, 500, 600, 700
foreach ($o in $offsets) {
    $out = "w6run\R3_o$($o)_W50_memread"
    $log = "..\w6_rev3\sweep_o$($o).log"
    $err = "..\w6_rev3\sweep_o$($o).err.log"
    if (Test-Path "$out\results_defer_W50.json") {
        Write-Host "[skip] o$o already done"
        continue
    }
    Write-Host "[run ] o$o -> $out"
    $a = @(
        "-u", "examples\w6_defer_stream.py",
        "--edits", "100", "--data-offset", "$o",
        "--defer-window", "50", "--layers", "4,5,6,7,8",
        "--refine-passes", "1", "--delta-budget", "0.5", "--guard-scale", "0.5",
        "--memory-read", "--memory-read-template", "instruct",
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] o$o exit=$($p.ExitCode)"
}
Write-Host "SWEEP COMPLETE"
