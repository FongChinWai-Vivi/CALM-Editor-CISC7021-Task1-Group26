# Rev4d: Tang W4 gate-ablation runner, re-run with the fluency coordinate.
#
# The runner gained a per-edit NLL pair (_seq_nll before/after) this round, so
# every W4 table produced before that patch has no fluency column.  Re-running
# the four arms with the patched runner is what makes §6 of the W6 report and
# the W4 table come from the same tool revision instead of an older binary.
#
# Waits for the gpt2-xl read-path job to finish first: gpt2-xl is 6.3 GB and the
# two jobs together would not fit on an 8 GB card.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"

while (Get-Process powershell -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne $PID } |
       Where-Object { (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine -like "*run_readpath2*" }) {
    Start-Sleep -Seconds 30
}
Write-Host "[gate] readpath2 finished, starting W4 arms"

Set-Location "$repo\TANG I CHOI\w4_gate_ablations"
& $py -u run_w4_gate_ablation.py --all-arms --subjects 40 --tag _rev4 `
    *>> "$repo\TANG I CHOI\w4_gate_ablations\results\rev4_run.log" `
    2>> "$repo\TANG I CHOI\w4_gate_ablations\results\rev4_run.err.log"
Write-Host "W4 REV4 DONE exit=$LASTEXITCODE"
