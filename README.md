# ASK: System 2 over H-JEPA

MSc AI project (APU). A frozen hierarchical JEPA world model (H-JEPA, kevinghst/H-JEPA) is System 1; System 2
decides what to aim for. Current best System 2: a paper-style renormalising generative model (RGM, Friston et al.
2025) that plans by inductive inference over a map counted from real experience and predicts the next latent, with
H-JEPA's lower levels as its reflex.

## Where this is going: an animal

The end point is not a planner for one benchmark. It is an **animal**: an agent that

- **learns from its own experience**. It plays, and its map of places and moves is counted from what really happened.
  Nothing is handed to it from an offline dataset.
- **dreams**. Between bouts it rehearses in its world model and tests what it imagined against reality.
- **is driven by curiosity**. With no goal, it goes where it has the most to learn.
- **has preferences**. A goal is a preferred observation, and it weighs reaching it against risk and time.
- **builds abstractions**. Places, moves and routes are its "letters, words and sentences", each level slower and more
  general than the one below.
- **plans** over those abstractions, and acts through a reflex it does not have to think about.
- **can act anywhere**. Mazes, pushing objects, games, and abstract spaces such as mathematics, where a "place" is a
  state of a proof and a "move" is a step of reasoning. Nothing in the agent may assume a body, a floor or a fixed
  action set.

Every experiment is a step toward that agent. Each one is judged by whether it removes an assumption the animal
could not make, not only by a benchmark score.

**Roadmap** (as of 25 Nov 2026):
1. AntMaze, the map from play alone: curious / random / Free Lunch explorers, the risk planner, data-decided states
   (branch `rgm-play-map`, [docs/NOTES_2026-11-25.md](docs/NOTES_2026-11-25.md)).
2. **One continuous life** (`experiments/life_run.py`). No episodes and no resets: the ant plays, a goal is thrown
   every 350 steps (about 20 goals), and between goals it plays again. We look at its behaviour over the whole life.
3. Push-T and the LEXA benchmark. Generalise the agent beyond navigation, improve the world model with the
   test-time-training / geometry / counterfactual papers, train it, and test it across different games.
4. Then other worlds, including abstract ones.

## Start here (new machine or new session)

1. **Restore** the environment and data: [RESTORE.md](RESTORE.md) (verified 20 Nov 2026, about 1 h; ~60 GB of data
   is downloaded or rebuilt, nothing large is in git).
2. **Read the latest session report and next steps:** [docs/NOTES_2026-11-25.md](docs/NOTES_2026-11-25.md)
   (branch `rgm-play-map`: play maps, risk planner, Free Lunch explorers, the continuous life run, what ran overnight
   and what to do next); before that [docs/NOTES_2026-11-24.md](docs/NOTES_2026-11-24.md) (branch `rgm-curious`: curious RGM -- map built by play + imagination, Beta hop model and one EFE planner instead
   of the 1/16 threshold); before that [docs/NOTES_2026-11-23.md](docs/NOTES_2026-11-23.md) (branch `rgm-beat-76`)
   and [docs/NOTES_2026-11-20.md](docs/NOTES_2026-11-20.md).
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

See the roadmap above and "Tomorrow" in [docs/NOTES_2026-11-25.md](docs/NOTES_2026-11-25.md): collect the overnight
play-map results, run one continuous life (`experiments/life_run.py`), then Push-T and LEXA.

## Layout

`system2/` agents and models · `experiments/` runners, builds, eval · `results/` per-task JSON, summaries, replays ·
`checkpoints/` small trained models (RGM in `checkpoints/rgm/`) · `hjepa_changes/` our patch to H-JEPA ·
`docs/` session notes · `report/` earlier published report.
