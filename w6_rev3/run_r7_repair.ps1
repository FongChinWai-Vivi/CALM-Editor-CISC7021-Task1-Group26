# Rev4e: one-slice pilot of the guarded repair pass (priority 1).
#
# The frozen arm writes 63 % of the stream parametrically and answers 43 % of it
# through those weights; the 37 % that the gate routes to memory plus the rows
# that were written and then lost are the dead mass this pass targets.  It runs
# on slice 0-99 only until it is known whether the guards pay for themselves.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"
$out = "w6run\R7_o0_W50_b025_inject_instruct_para_repair"
& $py -u examples\w6_defer_stream.py `
    --edits 100 --data-offset 0 `
    --defer-window 50 --layers 4,5,6,7,8 `
    --refine-passes 1 --delta-budget 0.25 --guard-scale 0.5 `
    --memory-read --memory-read-template instruct --memory-serve inject `
    --memory-arb-mode ctx --survival-every 0 `
    --repair-pass --repair-max 40 --repair-budget 0.05 --repair-total-budget 1.0 `
    --eval-paraphrase `
    --out-dir $out *> ..\w6_rev3\r7_o0.log 2> ..\w6_rev3\r7_o0.err.log
Write-Output "R7 DONE exit=$LASTEXITCODE"
