# AntMaze exploration ablation (Free Lunch style): runs sequentially, resumable (skips finished JSONs).
$env:PYTHONUTF8 = "1"
$env:HJEPA_HOME = "C:\Users\TP130289\Downloads\ask\data"
$py = "C:\Users\TP130289\AppData\Local\miniconda3\envs\hjepa\python.exe"
Set-Location "C:\Users\TP130289\Downloads\ask"
$conds = @(
  "pattern:fractal", "pattern:fractal_shuffled", "pattern:lorenz", "pattern:levy", "pattern:noise_gauss",
  "direct:fractal", "direct:fractal_shuffled", "random_macro", "random_actions"
)
foreach ($seed in 0, 1) {
  foreach ($c in $conds) {
    $tag = $c.Replace(":", "_")
    if (Test-Path "results\antmaze\${tag}_seed$seed.json") { continue }
    $rec = if ($seed -eq 0) { "--record" } else { "" }
    & $py experiments\antmaze_explore.py --goal-source $c --episodes 3 --steps 1000 --segment 100 --fast --seed $seed $rec 2>&1 |
      Select-String -Pattern "^\[" | ForEach-Object { $_.Line } | Tee-Object -Append "results\antmaze_log.txt"
  }
}
"ALL DONE" | Tee-Object -Append "results\antmaze_log.txt"
