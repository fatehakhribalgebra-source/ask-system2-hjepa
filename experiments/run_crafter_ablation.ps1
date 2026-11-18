# Crafter exploration ablation (FourRoom-initialised H-JEPA fine-tuned on CrafterPlay). Resumable.
$env:PYTHONUTF8 = "1"; $env:HOME = $env:USERPROFILE
$env:HJEPA_HOME = "C:\Users\TP130289\Downloads\ask\data"
$py = "C:\Users\TP130289\AppData\Local\miniconda3\envs\hjepa\python.exe"
Set-Location "C:\Users\TP130289\Downloads\ask"
$conds = @("random_actions", "fl_direct:fractal", "fl_direct:fractal_shuffled", "pattern:fractal",
           "pattern:fractal_shuffled", "random_macro", "direct:fractal", "pattern:noise_gauss",
           "pattern:lorenz", "pattern:levy")
foreach ($seed in 0, 1) {
  foreach ($c in $conds) {
    $tag = $c.Replace(":", "_")
    if (Test-Path "results\crafter\${tag}_seed$seed.json") { continue }
    $extra = @()
    if ($seed -eq 0) { $extra += "--record" }
    & $py experiments\crafter_explore.py --goal-source $c --episodes 5 --steps 1000 --seed $seed @extra 2>&1 |
      Select-String -Pattern "^\[" | ForEach-Object { $_.Line } | Tee-Object -Append "results\crafter_log.txt"
  }
}
"ALL DONE" | Tee-Object -Append "results\crafter_log.txt"
