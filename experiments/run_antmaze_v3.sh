#!/usr/bin/env bash
# System 2 v3 on AntMaze: structured vs unstructured pattern, learning vs frozen readout, plus no-pattern control.
# Each condition is its own process (Windows/glfw: one World per process).
SEED=${SEED:-0}; EP=${EP:-5}; STEPS=${STEPS:-1000}
PY=${PY:-/c/Users/TP130289/.conda/envs/hjepa/python.exe}
export HJEPA_HOME=${HJEPA_HOME:-"$(pwd)/data"} PYTHONUTF8=1 HOME=${HOME:-$USERPROFILE}
for cond in "fractal" "fractal_shuffled" "noise_gauss" "fractal --no-learn" "fractal_shuffled --no-learn" "none"; do
  $PY experiments/antmaze_explore_v3.py --source $cond --episodes $EP --steps $STEPS --seed $SEED --record 2>&1 \
    | grep -E --line-buffered "^\[|Traceback|Error"
done
echo "ALL DONE seed $SEED"
