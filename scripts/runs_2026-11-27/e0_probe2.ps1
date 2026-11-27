Set-Location 'C:\Users\TP130289\Documents\jep s2\ask-system2-hjepa'
$env:HJEPA_HOME = "$PWD\data"; $env:HOME = $env:USERPROFILE; $env:PYTHONUTF8 = '1'
$PY = "$env:LOCALAPPDATA\miniconda3\envs\hjepa\python.exe"
foreach ($run in @(@(0,'a'), @(0,'b'), @(0,'c'), @(1,'a'))) {
  $rep = $run[0]; $tag = $run[1]
  "== det repeat $rep run $tag $(Get-Date -Format T)"
  & $PY experiments/eval_l3_planner.py --single --planner rgm_paper --rgm-obs-level 3 --seed 42 `
    --rgm-paper-path data/rgm_paper_z3_k64_seed42.pkl --tasks 10 --deterministic --repeat $rep `
    --out "results/e0_probe/det2/rep$rep$tag" 2>&1 | Where-Object { $_ -match '^\{|deterministic implementation|Error' } | Select-Object -Unique
}
"== done $(Get-Date -Format T)"
