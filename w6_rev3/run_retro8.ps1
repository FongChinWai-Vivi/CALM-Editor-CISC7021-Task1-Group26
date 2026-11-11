# Rev4: close the coverage hole (retro-demote at end of stream) and re-measure
# all 8 held-out slices at the frozen operating point (W=50, budget=0.25).
# Identical to run_sweep_b025.ps1 except --memory-serve hybrid and
# --memory-retro-demote, so per-slice numbers stay poolable against R3b.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$offsets = 0, 100, 200, 300, 400, 500, 600, 700
foreach ($o in $offsets) {
    $out = "w6run\R4_o$($o)_W50_b025_hybrid_retro"
    $log = "..\w6_rev3\retro_o$($o).log"
    $err = "..\w6_rev3\retro_o$($o).err.log"
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
        "--memory-serve", "hybrid", "--memory-retro-demote",
        "--memory-arb-mode", "ctx",
        "--survival-every", "0",
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] o$o exit=$($p.ExitCode)"
}
Write-Host "RETRO8 COMPLETE"
