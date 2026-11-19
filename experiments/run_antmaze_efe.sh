#!/usr/bin/env bash
# Real EFE over the H-JEPA latent (Gaussian ensemble, whitened 32-d PCA observation), VAE policy prior.
SEED=${SEED:-0}; EP=${EP:-5}; STEPS=${STEPS:-1000}
PY=${PY:-/c/Users/TP130289/.conda/envs/hjepa/python.exe}
export HJEPA_HOME=${HJEPA_HOME:-"$(pwd)/data"} PYTHONUTF8=1 HOME=${HOME:-$USERPROFILE}
run() { $PY experiments/antmaze_explore_v3.py --generator efe --episodes $EP --steps $STEPS --seed $SEED --record "$@" 2>&1 \
          | grep -E --line-buffered "^\[|Traceback|Error"; }
run --source fractal
run --source fractal_shuffled
run --source none
run --source fractal --w-amb 0 --w-risk 0 --efe-tag _epi
echo "ALL DONE efe seed $SEED"
