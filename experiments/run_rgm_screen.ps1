# Screen RGM variants on the tasks the 20 Nov 2026 planner lost (results/paper_eval_rgm), one variant per process
# chain. Usage: pwsh experiments/run_rgm_screen.ps1 -Seed 42 -Variant base|stuck|ctx|all [-Tasks 3,5,...]
param(
  [int]$Seed = 42,
  [string]$Variant = 'base',
  [string]$Tasks = '3,5,6,8,22,23,27,29,35,36,39,43',
  [string]$Out = 'results/rgm_screen'
)
$ErrorActionPreference = 'Continue'
$taskList = @([regex]::Matches("$Tasks", '\d+') | ForEach-Object { $_.Value })
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = "$env:LOCALAPPDATA\miniconda3\envs\hjepa\python.exe"
$env:HJEPA_HOME = "$PWD\data"; $env:HOME = $env:USERPROFILE; $env:PYTHONUTF8 = '1'
$seqs = if ($Seed -eq 42) { 'data/latent_seqs_ant_l2_l3.npz' } else { "data/latent_seqs_ant_l2_l3_seed$Seed.npz" }
$extra = @(switch ($Variant) {
  'base'  { @() }
  'stuck' { @('--rgm-stuck', '3', '--rgm-relax') }
  'ctx'   { @('--rgm-ctx-seqs', $seqs) }
  'all'   { @('--rgm-stuck', '3', '--rgm-relax', '--rgm-ctx-seqs', $seqs) }
  'obs10' { @('--rgm-observe-every', '10') }
  'la2'   { @('--rgm-lookahead', '2') }
  'la2obs10' { @('--rgm-lookahead', '2', '--rgm-observe-every', '10') }
  'agent'  { @('--rgm-agent', '--rgm-paper-path', "data/rgm_emeans_z3_k64_seed$Seed.pkl", '--rgm-calib-seqs', $seqs,
               '--rgm-f3', "data/f3_transitions_emeans_seed$Seed.npz", '--rgm-f3-weight', '0.1', '--rgm-learn', '--rgm-lr', '1') }
  'agentp' { @('--rgm-agent', '--rgm-paper-path', "data/rgm_emeans_z3_k64_seed$Seed.pkl", '--rgm-calib-seqs', $seqs,
               '--rgm-f3', "data/f3_transitions_emeans_seed$Seed.npz", '--rgm-f3-weight', '0.1', '--rgm-learn', '--rgm-lr', '1',
               '--rgm-persist', "$Out/agentp/model_seed$Seed.pkl") }
  'live'   { @('--rgm-live', '--rgm-learn', '--rgm-lr', '1', '--rgm-persist', "$Out/live/model_seed$Seed.pkl") }
  'dream'  { @('--rgm-live', '--rgm-dream', '--rgm-learn', '--rgm-lr', '1', '--rgm-persist', "$Out/dream/model_seed$Seed.pkl") }
  'filter' { @('--rgm-full') }
  'full'   { @('--rgm-full', '--rgm-learn', '--rgm-lr', '3') }
  'fullf3' { @('--rgm-full', '--rgm-learn', '--rgm-lr', '3', '--rgm-f3', "data/f3_transitions_seed$Seed.npz", '--rgm-f3-weight', '0.1') }
  default { $Variant.Split(' ') }
})
& $PY experiments/eval_l3_planner.py --planner rgm_paper --rgm-obs-level 3 --seed $Seed `
  --rgm-paper-path "data/rgm_paper_z3_k64_seed$Seed.pkl" --tasks @taskList @extra --out "$Out/$Variant/seed$Seed"
