# ASK: System 2 over H-JEPA

MSc AI project (APU). A frozen hierarchical JEPA world model (H-JEPA, kevinghst/H-JEPA) is System 1; System 2
decides what to aim for. Current best System 2: a paper-style renormalising generative model (RGM, Friston et al.
2025) that plans by inductive inference over a map counted from real experience and predicts the next latent, with
H-JEPA's lower levels as its reflex.

## Start here (new machine or new session)

1. **Restore** the environment and data: [RESTORE.md](RESTORE.md) (verified 20 Nov 2026, about 1 h; ~60 GB of data
   is downloaded or rebuilt, nothing large is in git).
2. **Read the latest session report and next steps:** [docs/NOTES_2026-11-20.md](docs/NOTES_2026-11-20.md).
3. Earlier context: [docs/NOTES_2026-11-19.md](docs/NOTES_2026-11-19.md), [DESIGN.md](DESIGN.md) (the proposal PDF
   is a loose guide only).

## Headline result (20 Nov 2026)

Visual AntMaze, paper protocol (50 tasks x 3 seeds, 350 env steps):

| Planner | Success % (mean ± SE) |
|---|---|
| Paper-style RGM on H-JEPA level 3 + H-JEPA reflex (`system2/rgm_paper.py`) | 76.0 ± 4.2 |
| 3-level H-JEPA gradient planner (paper) | 73.3 ± 3.5 |

On par. All RGM variants and lessons: `results/l3_planner_pc2/summary_A.md`; per-task results:
`results/paper_eval_rgm/`.

## Next

Push-T RGM (code ready); RGM on z2 measured in z3; RGM learning from imagined trajectories; the main direction —
explore without goals (random / Free Lunch / curiosity) to build RGM's map, then reach goals given on the fly.
Details in the notes.

## Layout

`system2/` agents and models · `experiments/` runners, builds, eval · `results/` per-task JSON, summaries, replays ·
`checkpoints/` small trained models (RGM in `checkpoints/rgm/`) · `hjepa_changes/` our patch to H-JEPA ·
`docs/` session notes · `report/` earlier published report.
