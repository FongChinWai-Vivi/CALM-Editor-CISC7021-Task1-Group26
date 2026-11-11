# Rev4c: Rev3 replication + the one thing Rev3 never measured.
#
# Rev3 quoted paraphrase recall 0.343 from a *clean* gpt2-small harness that
# shared neither the edited weights nor the stream's note handling.  R5 tested
# whether the clean-LM template win (fewshot: paraphrase 0.325 -> 0.610 pooled
# over 300 edits) transfers into the loop: it does not.  On slice 0-99 with the
# parameter channel held constant (0.430 in both arms) the memory channel falls
# from 0.160 to 0.100 and e2e from 0.590 to 0.530.  So the template is reverted
# here and the measurement gap is closed instead: the same eight slices are
# re-run with --eval-paraphrase, which reads the rephrase prompts on the EDITED
# model with the same template as the canonical measurement.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$offsets = 0, 100, 200, 300, 400, 500, 600, 700
foreach ($o in $offsets) {
    $out = "w6run\R6_o$($o)_W50_b025_inject_instruct_para"
    $log = "..\w6_rev3\r6_o$($o).log"
    $err = "..\w6_rev3\r6_o$($o).err.log"
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
        "--memory-serve", "inject",
        "--memory-arb-mode", "ctx",
        "--survival-every", "0", "--eval-paraphrase",
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] o$o exit=$($p.ExitCode)"
}
Write-Host "R6 COMPLETE"
