# ASK: System 2 over H-JEPA — how to restore this project

Everything that cannot be re-downloaded is in this repository. The ~41 GB of data and the H-JEPA clone are
left out on purpose; the steps below rebuild them.

## What is here
| Path | Contents |
|---|---|
| `system2/` | Pattern sources, reservoir, goal engines v1–v3, VAE / CEM / EFE generators, active-inference agents (`active_inference.py`, `deep_aif.py`, `discrete_aif.py`), level-3 environment (`level3_env.py`), goal bank, adapted official agents (`tschantz_l3.py`, `fountas_z3.py`), H-JEPA bridge |
| `envs/` | Free Lunch maze, Craftax wrapper (144px render, no auto-reset), Crafter action embedding |
| `experiments/` | Runners (`fourroom_explore*.py`, `crafter_explore.py`, `antmaze_explore.py`, `freelunch_direct.py`), batch scripts, stats (`summarize.py`, `contrasts.py`), report builder |
| `results/` | Per-episode / per-task JSON and replay MP4s for every run; `crafter_INVALID_autoreset/` = first Crafter batch, invalid (achievement bug); `antmaze_aif/old_mapping/` = superseded north-star run |
| `logs/` | Run logs (moved out of the repo root) |
| `docs/` | Session notes and findings (`NOTES_2026-11-19.md`) |
| `checkpoints/system2/` | Small System-2 models: action VAE, state VAE, Fountas z3 model, Tschantz level-3 ensemble |
| `report/system2_report.html` | The published report (also at https://claude.ai/artifact/EWCLvfrvwqNDfvcfJozcpR) |
| `checkpoints/crafter_ft_hjepa_l3/` | Fine-tuned Crafter H-JEPA (FourRoom init, 16k steps on CrafterPlay 144px) + normalizer + config |
| `hjepa_changes/` | Our changes to kevinghst/H-JEPA: `BASE_COMMIT.txt`, `hjepa_modified_files.patch`, `new_files/` |
| `DESIGN.md` | Working design notes |

## Restore on a new Windows machine
```bash
winget install Git.Git Anaconda.Miniconda3
git clone <this repo> ask && cd ask
git clone https://github.com/kevinghst/H-JEPA.git
cd H-JEPA && git checkout 97054ecaff83e13839195ec4b92d74c08a0127c5
git apply ../hjepa_changes/hjepa_modified_files.patch
xcopy /E /I /Y ..\hjepa_changes\new_files\h_jepa h_jepa
conda create -n hjepa python=3.11.13 -y && conda activate hjepa
pip install -e ".[train,env]" --extra-index-url https://download.pytorch.org/whl/cu128
pip install craftax imageio-ffmpeg huggingface_hub
cd ..
```
Environment variables for every session (PowerShell):
```bash
$env:HJEPA_HOME="$PWD\data"; $env:HOME=$env:USERPROFILE; $env:PYTHONUTF8="1"
```

## Rebuild the data
```bash
hf download jepa-world-models/h-jepa --local-dir data/ckpts --include "fourroom/fourroom_hjepa_l3/*" --include "ant/ant_hjepa_l3/seed42/*"
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir H-JEPA/h_jepa/assets --include "eval_trajs/*"
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir data/datasets --include "visual_antmaze_medium_stitch_val.h5"
mkdir data\ckpts\crafter\crafter_ft_hjepa_l3\seed42
copy checkpoints\crafter_ft_hjepa_l3\* data\ckpts\crafter\crafter_ft_hjepa_l3\seed42\
copy checkpoints\system2\*.pt data\
python experiments/build_macro_action_data.py      # data/macro_actions_ant_l3_h5.npz (real level-3 transitions)
python experiments/build_z3_game_assets.py         # data/z3_game_assets.pt (macro-action prototypes, eval-only xy probe)
```
Third-party active-inference code (cloned, not committed):
```bash
git clone https://github.com/alec-tschantz/rl-inference third_party/rl-inference
git -C third_party/rl-inference checkout 007dba5836b79417aaa0aa6216492745aa9e1bb3
git clone https://github.com/zfountas/deep-active-inference-mc third_party/deep-active-inference-mc
git -C third_party/deep-active-inference-mc checkout d7e76d828635a7687c1ab446645db4d9bb12584e
```
Windows note: MuJoCo renders through glfw, and every `World` created after the first in a process renders black
frames. Run one `World` per Python process (the AntMaze eval does this via `experiments/run_ant_eval_isolated.py`).
CrafterPlay (only needed to retrain the Crafter model): download shards from Harvard Dataverse
(survival `doi:10.7910/DVN/FHR2KS`, gatherer `doi:10.7910/DVN/JXNHRB`; we used parts 0000–0001 of each),
extract with `tar -xf <shard>.tar.zst -C data/crafter/raw --exclude "*.pkl"`, then
`python -I experiments/build_crafter_h5.py --per-source 40 --size 144 --name crafter_play144_train`.

## Re-run
```bash
python experiments/fourroom_explore.py --goal-source pattern:fractal --episodes 3 --steps 600 --fast --record
python experiments/fourroom_explore_v2.py --goal-source v2:fractal --episodes 3 --steps 600 --record
python experiments/crafter_explore.py --goal-source pattern:fractal --episodes 5 --steps 1000 --record
python experiments/contrasts.py; python experiments/build_report.py
python experiments/run_ant_eval_isolated.py                                   # paper AntMaze eval (9/13 on tasks 0-12)
python experiments/eval_l3_planner.py --planner tschantz --plan-len 15 --tasks 0 1 2 3 4 5 6 7 8 9 10 11 12
```
