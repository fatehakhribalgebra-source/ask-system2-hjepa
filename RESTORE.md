# ASK: System 2 over H-JEPA — how to restore this project

Everything that cannot be re-downloaded is in this repository. The ~41 GB of data and the H-JEPA clone are
left out on purpose; the steps below rebuild them.

## What is here
| Path | Contents |
|---|---|
| `system2/` | Pattern sources, reservoir, EFE goal engine v1 (`goal_engine.py`) and v2 (`goal_engine_v2.py`), H-JEPA bridge |
| `envs/` | Free Lunch maze, Craftax wrapper (144px render, no auto-reset), Crafter action embedding |
| `experiments/` | Runners (`fourroom_explore*.py`, `crafter_explore.py`, `antmaze_explore.py`, `freelunch_direct.py`), batch scripts, stats (`summarize.py`, `contrasts.py`), report builder |
| `results/` | Per-episode JSON (with trajectories) and replay MP4s for every run; `crafter_INVALID_autoreset/` = first Crafter batch, invalid (achievement bug) |
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
hf download jepa-world-models/h-jepa --local-dir data/ckpts --include "fourroom/fourroom_hjepa_l3/*"
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir H-JEPA/h_jepa/assets --include "eval_trajs/*"
mkdir data\ckpts\crafter\crafter_ft_hjepa_l3\seed42
copy checkpoints\crafter_ft_hjepa_l3\* data\ckpts\crafter\crafter_ft_hjepa_l3\seed42\
```
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
```
