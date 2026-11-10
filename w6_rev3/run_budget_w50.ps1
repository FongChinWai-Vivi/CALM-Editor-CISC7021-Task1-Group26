# Tang ③ follow-up — the parameter that actually moves admission at W=50.
# The gate-threshold retag came back flat (0.15 / 0.30 / 0.50 identical), so the
# operating curve has to be swept over delta_budget instead: that is the knob the
# writer's admission decision actually responds to in this configuration.
$ErrorActionPreference = "Stop"
$repo = "C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
$py = "$repo\.venv\Scripts\python.exe"
Set-Location "$repo\EasyEdit"

$budgets = @(
    @{ name = "b025"; v = "0.25" },
    @{ name = "b100"; v = "1.0" }
)

foreach ($b in $budgets) {
    $out = "w6run\R3_o0_W50_budget_$($b.name)"
    $log = "..\w6_rev3\budget_$($b.name).log"
    $err = "..\w6_rev3\budget_$($b.name).err.log"
    if (Test-Path "$out\results_defer_W50.json") {
        Write-Host "[skip] $($b.name) already done"
        continue
    }
    Write-Host "[run ] $($b.name) -> $out"
    $a = @(
        "-u", "examples\w6_defer_stream.py",
        "--edits", "100", "--data-offset", "0",
        "--defer-window", "50", "--layers", "4,5,6,7,8",
        "--refine-passes", "1", "--delta-budget", $b.v, "--guard-scale", "0.5",
        "--memory-read", "--memory-read-template", "instruct",
        "--out-dir", $out
    )
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru -Wait
    Write-Host "[done] $($b.name) exit=$($p.ExitCode)"
}
Write-Host "BUDGET SWEEP COMPLETE"
