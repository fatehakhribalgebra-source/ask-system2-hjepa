#!/usr/bin/env bash
# Discrete active inference on a latent place map; H-JEPA walks to remembered real observations of chosen places.
SEED=${SEED:-0}; EP=${EP:-5}; STEPS=${STEPS:-1000}
PY=${PY:-/c/Users/TP130289/.conda/envs/hjepa/python.exe}
export HJEPA_HOME=${HJEPA_HOME:-"$(pwd)/data"} PYTHONUTF8=1 HOME=${HOME:-$USERPROFILE}
run() { $PY experiments/antmaze_discrete_aif.py --goal-obs --episodes $EP --steps $STEPS --seed $SEED --record "$@" 2>&1 \
          | grep -E --line-buffered "^\[|Traceback|Error"; }
( run --target fractal; run --target fractal_shuffled; run --random ) &
( run --target none; run --target fractal --w-nov 0 ) &
wait
echo "ALL DONE discrete-aif seed $SEED"
