#!/usr/bin/env bash
# Generator G (pattern code -> macro-action VAE) on AntMaze: structured vs unstructured codes vs plain prior.
SEED=${SEED:-0}; EP=${EP:-5}; STEPS=${STEPS:-1000}
PY=${PY:-/c/Users/TP130289/.conda/envs/hjepa/python.exe}
export HJEPA_HOME=${HJEPA_HOME:-"$(pwd)/data"} PYTHONUTF8=1 HOME=${HOME:-$USERPROFILE}
for src in fractal fractal_shuffled noise_gauss none; do
  $PY experiments/antmaze_explore_v3.py --generator vae --source $src --episodes $EP --steps $STEPS --seed $SEED --record 2>&1 \
    | grep -E --line-buffered "^\[|Traceback|Error"
done
echo "ALL DONE gen seed $SEED"
