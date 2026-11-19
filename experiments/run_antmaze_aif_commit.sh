#!/usr/bin/env bash
# Active-inference agent, level 3 = environment; actions on H-JEPA's LayerNorm action manifold; commit to 3-step plans.
# North star T(o) from Free Lunch patterns vs controls.
SEED=${SEED:-0}; EP=${EP:-5}; STEPS=${STEPS:-1000}
PY=${PY:-/c/Users/TP130289/.conda/envs/hjepa/python.exe}
export HJEPA_HOME=${HJEPA_HOME:-"$(pwd)/data"} PYTHONUTF8=1 HOME=${HOME:-$USERPROFILE}
run() { $PY experiments/antmaze_aif.py --env real --commit --episodes $EP --steps $STEPS --seed $SEED --record "$@" 2>&1 \
          | grep -E --line-buffered "^\[|Traceback|Error"; }
for T in fractal fractal_shuffled noise_labels typical none; do run --target $T; done
run --random
echo "ALL DONE aif-commit seed $SEED"
