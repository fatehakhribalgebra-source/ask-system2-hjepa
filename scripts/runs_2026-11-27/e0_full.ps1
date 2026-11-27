Set-Location 'C:\Users\TP130289\Documents\jep s2\ask-system2-hjepa'
$env:HJEPA_HOME = "$PWD\data"; $env:HOME = $env:USERPROFILE; $env:PYTHONUTF8 = '1'
$PY = "$env:LOCALAPPDATA\miniconda3\envs\hjepa\python.exe"
$tasks = 0..12 | ForEach-Object { "$_" }
$jobs = @(
  @('rgm_paper', 0, 'rep0'), @('gd', 0, 'rep0'),
  @('rgm_paper', 1, 'rep1'), @('gd', 1, 'rep1'),
  @('rgm_paper', 2, 'rep2'), @('gd', 2, 'rep2'),
  @('gd', 0, 'rep0_again')
)
foreach ($j in $jobs) {
  $pl, $rep, $name = $j
  "== $pl $name $(Get-Date -Format T)"
  [string[]]$extra = if ($pl -eq 'rgm_paper') { '--rgm-obs-level', '3', '--rgm-paper-path', 'data/rgm_paper_z3_k64_seed42.pkl' } else { @() }
  & $PY experiments/eval_l3_planner.py --planner $pl --seed 42 --tasks @tasks --deterministic --repeat $rep `
    --out "results/e0_repro/$pl/$name" @extra 2>&1 | Select-Object -Last 14
}
"== done $(Get-Date -Format T)"
