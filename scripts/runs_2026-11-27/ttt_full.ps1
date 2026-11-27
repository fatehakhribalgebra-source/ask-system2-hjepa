Set-Location 'C:\Users\TP130289\Documents\jep s2\ask-system2-hjepa'
$env:HJEPA_HOME = "$PWD\data"; $env:HOME = $env:USERPROFILE; $env:PYTHONUTF8 = '1'
$PY = "$env:LOCALAPPDATA\miniconda3\envs\hjepa\python.exe"
$tasks = 0..12 | ForEach-Object { "$_" }
foreach ($rep in 0, 1, 2) {
  "== rgm_paper_ttt rep$rep $(Get-Date -Format T)"
  & $PY experiments/eval_l3_planner.py --planner rgm_paper --rgm-obs-level 3 --rgm-paper-path data/rgm_paper_z3_k64_seed42.pkl `
    --seed 42 --tasks @tasks --deterministic --repeat $rep --ttt --ttt-lr 1e-4 --tag _ttt `
    --out "results/e0_repro/rgm_paper_ttt/rep$rep" 2>&1 | Select-Object -Last 14
}
"== done $(Get-Date -Format T)"
