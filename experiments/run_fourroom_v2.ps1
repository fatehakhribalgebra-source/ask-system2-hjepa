# System 2 v2 on FourRoom; waits for the v1 batch ("ALL DONE" in fourroom_log.txt), then runs. Resumable.
$env:PYTHONUTF8 = "1"; $env:HOME = $env:USERPROFILE
$env:HJEPA_HOME = "C:\Users\TP130289\Downloads\ask\data"
$py = "C:\Users\TP130289\AppData\Local\miniconda3\envs\hjepa\python.exe"
Set-Location "C:\Users\TP130289\Downloads\ask"
while (-not (Select-String -Path "results\fourroom_log.txt" -Pattern "ALL DONE" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 60 }
$runs = @(@("v2:fractal", ""), @("v2:none", ""), @("v2:fractal_shuffled", ""), @("v2:fractal", "--no-learn"))
foreach ($seed in 0, 1) {
  foreach ($r in $runs) {
    $tag = $r[0].Replace(":", "_") + $(if ($r[1]) { "_nolearn" } else { "" })
    if (Test-Path "results\fourroom\${tag}_seed$seed.json") { continue }
    $extra = @()
    if ($r[1]) { $extra += $r[1] }
    if ($seed -eq 0) { $extra += "--record" }
    & $py experiments\fourroom_explore_v2.py --goal-source $r[0] --episodes 3 --steps 600 --seed $seed @extra 2>&1 |
      Select-String -Pattern "^\[" | ForEach-Object { $_.Line } | Tee-Object -Append "results\fourroom_v2_log.txt"
  }
}
"ALL DONE" | Tee-Object -Append "results\fourroom_v2_log.txt"
