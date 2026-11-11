# Rev4b: the read path was rendered with ONE template across all of Rev3, and
# that template was never swept.  w6_rev3/w6_recall_readpath.py puts five
# templates on the same cases with the same store and key, and "fewshot" more
# than doubles paraphrase delivery on a clean LM (0.30 -> 0.635 at o0, 0.343 ->
# ~0.60 pooled), at a cost of 0.09 canonical delivery.
#
# This arm is therefore inject + retro-demote + fewshot: inject is the honest
# channel (the edited LM must actually copy the note, nothing is served
# directly), retro-demote is the coverage fix from R4, and the only change
# against R3b/R4 is the note template.  kept as inject on purpose so the gain
# cannot be attributed to serving.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$offsets = 0, 100, 200, 300, 400, 500, 600, 700
foreach ($o in $offsets) {
    $out = "w6run\R5_o$($o)_W50_b025_inject_retro_fewshot"
    $log = "..\w6_rev3\r5_o$($o).log"
    $err = "..\w6_rev3\r5_o$($o).err.log"
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
        "--memory-read", "--memory-read-template", "fewshot",
        "--memory-max-length", "224",
        "--memory-serve", "inject", "--memory-retro-demote",
        "--memory-arb-mode", "ctx",
        "--survival-every", "0", "--eval-paraphrase",
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] o$o exit=$($p.ExitCode)"
}
Write-Host "R5 COMPLETE"
