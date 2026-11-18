# FourRoom exploration ablation: sequential and resumable (skips finished JSONs). Replays for seed 0.
$env:PYTHONUTF8 = "1"; $env:HOME = $env:USERPROFILE
$env:HJEPA_HOME = "C:\Users\TP130289\Downloads\ask\data"
$py = "C:\Users\TP130289\AppData\Local\miniconda3\envs\hjepa\python.exe"
Set-Location "C:\Users\TP130289\Downloads\ask"
$conds = @("random_actions", "pattern:fractal", "pattern:fractal_shuffled", "random_macro", "direct:fractal",
           "pattern:noise_gauss", "pattern:lorenz", "pattern:levy", "direct:fractal_shuffled")
foreach ($seed in 0, 1) {
  foreach ($c in $conds) {
    $tag = $c.Replace(":", "_")
    if (Test-Path "results\fourroom\${tag}_seed$seed.json") { continue }
    $extra = @("--fast")
    if ($seed -eq 0) { $extra += "--record" }
    & $py experiments\fourroom_explore.py --goal-source $c --episodes 3 --steps 600 --seed $seed @extra 2>&1 |
      Select-String -Pattern "^\[" | ForEach-Object { $_.Line } | Tee-Object -Append "results\fourroom_log.txt"
  }
}
"ALL DONE" | Tee-Object -Append "results\fourroom_log.txt"
