# Tang ③ — W=50 double-objective retag.
# Same slice (o0, 100 edits, W=50, memory in-loop) re-run under three gate
# write-thresholds. Each output carries the threshold it was produced under and
# both coordinates (admission / hit / NLL), so the curve is measured, not
# extrapolated from a single operating point.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$ths = @(
    @{ name = "t015"; v = "0.15" },
    @{ name = "t030"; v = "0.30" },
    @{ name = "t050"; v = "0.50" }
)

foreach ($t in $ths) {
    $out = "w6run\R3_o0_W50_retag_$($t.name)"
    $log = "..\w6_rev3\retag_$($t.name).log"
    $err = "..\w6_rev3\retag_$($t.name).err.log"
    if (Test-Path "$out\results_defer_W50.json") {
        Write-Host "[skip] $($t.name) already done"
        continue
    }
    Write-Host "[run ] $($t.name) -> $out"
    $a = @(
        "-u", "examples\w6_defer_stream.py",
        "--edits", "100", "--data-offset", "0",
        "--defer-window", "50", "--layers", "4,5,6,7,8",
        "--refine-passes", "1", "--delta-budget", "0.5", "--guard-scale", "0.5",
        "--memory-read", "--memory-read-template", "instruct",
        "--gate-write-threshold", $t.v,
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] $($t.name) exit=$($p.ExitCode)"
}
Write-Host "RETAG COMPLETE"
