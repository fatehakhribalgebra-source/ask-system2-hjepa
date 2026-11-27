$ErrorActionPreference = 'Continue'
$Repo = 'C:\Users\TP130289\Documents\jep s2\ask-system2-hjepa'
$PY = "$env:LOCALAPPDATA\miniconda3\envs\hjepa\python.exe"
$HF = "$env:LOCALAPPDATA\miniconda3\envs\hjepa\Scripts\hf.exe"
$env:PYTHONUTF8 = '1'
Set-Location "$Repo\H-JEPA"
"== pip install H-JEPA $(Get-Date -Format T)"
& $PY -m pip install -e ".[train,env]" --extra-index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { "== retry (antlr4 wheel note in RESTORE.md)"; & $PY -m pip install -e ".[train,env]" --extra-index-url https://download.pytorch.org/whl/cu128 }
"== pip exit $LASTEXITCODE"
& $PY -m pip install craftax imageio-ffmpeg huggingface_hub
"== extras exit $LASTEXITCODE"
& $PY -c "import torch; print('TORCH', torch.__version__, torch.cuda.is_available())"
Set-Location $Repo
"== downloads $(Get-Date -Format T)"
& $HF download jepa-world-models/h-jepa --local-dir data/ckpts --include "ant/ant_hjepa_l3/seed42/*" --include "ant/ant_hjepa_l3/seed43/*" --include "ant/ant_hjepa_l3/seed44/*"
"== ckpts exit $LASTEXITCODE"
& $HF download jepa-world-models/h-jepa --repo-type dataset --local-dir H-JEPA/h_jepa/assets --include "eval_trajs/*"
"== eval_trajs exit $LASTEXITCODE"
& $HF download jepa-world-models/h-jepa --repo-type dataset --local-dir data/datasets --include "visual_antmaze_medium_stitch_val.h5"
"== val exit $LASTEXITCODE"
"== done $(Get-Date -Format T)"
