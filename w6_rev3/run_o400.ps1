# Detached single-slice retry: slice o400 kept dying (-1) inside the sweep driver,
# so it is re-run on its own. Launched detached so the orchestrating shell timing
# out does not kill the run mid-flight.
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
Set-Location "$repo\EasyEdit"
$argsList = @(
    "-u", "examples\w6_defer_stream.py",
    "--edits", "100", "--data-offset", "400",
    "--defer-window", "50", "--layers", "4,5,6,7,8",
    "--refine-passes", "1", "--delta-budget", "0.25", "--guard-scale", "0.5",
    "--memory-read", "--memory-read-template", "instruct",
    "--out-dir", "w6run\R3b_o400_W50_b025"
)
$p = Start-Process -FilePath "$repo\.venv\Scripts\python.exe" -ArgumentList $argsList `
    -WorkingDirectory (Get-Location).Path `
    -RedirectStandardOutput "$repo\w6_rev3\o400_run.log" `
    -RedirectStandardError "$repo\w6_rev3\o400_run.err.log" `
    -WindowStyle Hidden -PassThru
Write-Host "pid=$($p.Id)"
