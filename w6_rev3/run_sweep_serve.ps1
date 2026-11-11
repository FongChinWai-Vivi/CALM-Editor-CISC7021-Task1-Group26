# Rev3 + memory-serve ablation, same 8 slices / budget / windows as the frozen
# R3b sweep so the two sets differ in exactly one flag: --memory-serve.
# R3b (inject) forces the LM to re-emit the note it was shown; measured copy
# fidelity is ~0.51, so half of the routed memory rows died as "none" while the
# store knew the answer. hybrid keeps the injection attempt and only falls back
# to a direct serve, and every such row is labelled memory_direct so the two
# behaviours stay separable in the tables.
param(
    [string]$Serve = "hybrid"
)
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$offsets = 0, 100, 200, 300, 400, 500, 600, 700
foreach ($o in $offsets) {
    $out = "w6run\R3c_o$($o)_W50_b025_$($Serve)"
    $log = "..\w6_rev3\sweepc_$($Serve)_o$($o).log"
    $err = "..\w6_rev3\sweepc_$($Serve)_o$($o).err.log"
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
        "--memory-serve", $Serve,
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] o$o exit=$($p.ExitCode)"
}
Write-Host "SWEEP SERVE=$Serve COMPLETE"
