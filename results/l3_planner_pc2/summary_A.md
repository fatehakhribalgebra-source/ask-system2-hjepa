# Paper-style RGM on H-JEPA level 3 — 37/39 (95%) over three runs

`system2/rgm_paper.py`, `experiments/build_rgm_paper.py`, planner `rgm_paper` in `experiments/eval_l3_planner.py`.
The RGM of "From pixels to planning" (Atari section) with the game frame replaced by H-JEPA's level-3 latent:
level-1 states = 64 k-means states of the PCA-whitened z3; level-2 / level-3 items = runs of 2 / 4 consecutive states
with counted transitions; inductive inference at the top level from items containing a goal state; NO action list --
RGM predicts the next latent (the real exemplar of the next distinct state on its plan) and H-JEPA's levels 2 and 1
act as the reflex that makes it happen. The prediction is passed down as [now, target, target] so H-JEPA's
last-subgoal anchoring cannot replace it with the goal image. Commitment by temporal depth: the plan is held for one
top-level item (4 latent steps) while reality matches it. Offline (held-out): plans found 125/125, plan ends 0.59 maze
units from the goal's xy.

| Run | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | solved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper-style RGM, z3, run 1 | 293 | 201 | 159 | 294 | 316 | 152 | 291 | 304 | 187 | 306 | 76 | 185 | 134 | 13/13 |
| paper-style RGM, z3, run 2 | 290 | 328 | 130 | 175 | 180 | 170 | 253 | 243 | 232 | 299 | 124 | 238 | 172 | 13/13 |
| paper-style RGM, z3, run 3 | 242 | 297 | 337 | 282 | x | 200 | 270 | 251 | x | 172 | 145 | 281 | 186 | 11/13 |
| paper-style RGM, z3 + continual learning (lr 10) | 312 | x | 274 | x | 235 | 132 | 243 | x | x | 266 | 157 | 174 | 116 | 9/13 |
| paper-style RGM, z2 (reflex = level 1) + learning | x | x | x | 156 | x | 348 | 304 | x | x | 234 | x | x | 159 | 5/13 |
| H-JEPA gradient planner (one run, same PC) | x | 250 | 285 | x | 231 | 324 | 318 | 292 | 248 | x | x | 329 | 275 | 9/13 |

Mean steps when solved: 223, 218, 242 (H-JEPA 284). Continual learning at lr 10 hurts: the offline counts are already
a good map of where the ant can go, and strong in-episode counts distort it. The z2 version's reflex (level 1
reaching a z2 exemplar 10 env steps ahead) often fails, and the agent re-plans to the same unreachable target.
Still to do for a fair claim: repeat H-JEPA's own planner on this PC, and run all 50 tasks.

# RGM as H-JEPA's level-3 action planner (design A) — first evaluation

Paper AntMaze goal-image eval, tasks 0–12, seed 42, 350-step budget, one process per task, this PC (RTX 5070 Ti).
Only `level_solvers[3]` changes; levels 2/1 are H-JEPA's. Steps to success, x = failed.

| Level-3 planner | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 | solved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H-JEPA gradient planner (gd) | x | 250 | 285 | x | 231 | 324 | 318 | 292 | 248 | x | x | 329 | 275 | 9/13 |
| RGM, replan every call (closed) | 346 | 308 | 308 | x | 309 | 134 | x | 325 | x | 312 | 327 | 321 | 181 | 10/13 |
| RGM, commit to route, replan on drift (commit) | x | 343 | 314 | 297 | 288 | x | x | 340 | 296 | x | 155 | 301 | 295 | 9/13 |
| RGM, EFE commit-or-replan, long routes (adaptive) | x | x | x | x | 329 | x | x | 244 | x | x | 330 | x | x | 3/13 |

| RGM, two-realities feedback, frozen (feedback) | x | x | x | x | x | 282 | 330 | x | x | x | 142 | 208 | 189 | 5/13 |
| RGM, two-realities feedback + online learning | 300 | x | 303 | 285 | 213 | 270 | x | 333 | x | x | x | 173 | 304 | 8/13 |

Feedback mode (the two-realities design): inner loop checks each planned route in F3 (H-JEPA level-3 predictor),
blocks moves where F3 disagrees with RGM (surprise above the 90th percentile of normal real-transition surprise,
1.63 nats) and replans, up to 3 tries; the committed route's expected states are the F3-checked ones. Outer loop
compares the real state after every level-3 step (20 env steps) with that expectation and replans on surprise.
Frozen: fastest when it succeeds (mean 230 steps) but replans on 83 of 157 real checks (median surprise 2.18 nats):
F3's imagined states exaggerate movement, so reality looks surprising against them. Learning (soft Dirichlet counts
from each real level-3 transition): 8/13, mean 273 steps.

| RGM, replan after every real action (F3-checked), frozen | 134 | x | x | 337 | 282 | 213 | 318 | x | 294 | x | x | 268 | 327 | 8/13 |
| RGM, replan after every real action, learning | 127 | 303 | x | x | x | 204 | x | 279 | x | x | 117 | 229 | 183 | 7/13 |
| RGM bridge, learning lr 1, decide every call | 202 | 270 | 209 | x | x | 189 | 298 | 266 | 288 | 350 | 142 | x | 329 | 10/13 |
| RGM bridge, learning lr 1, hold one action | 216 | x | x | x | x | x | 262 | x | 147 | 304 | 123 | 172 | 256 | 7/13 |
| RGM bridge, frozen, decide every call | 131 | x | 143 | 292 | x | x | 303 | x | 209 | x | 140 | x | x | 6/13 |
| **RGM bridge, learning lr 10, decide every call** | x | 200 | 213 | x | x | 154 | 215 | 328 | 124 | 241 | 203 | 308 | 160 | **10/13** |
| RGM bridge, learning lr 30, decide every call | 156 | 319 | x | 306 | 280 | 161 | 269 | x | 334 | 343 | x | x | 195 | 9/13 |

Bridge mode: closed loop (re-infer the real state every call) + learning of RGM's counts from each real level-3
transition + product of experts over all 12 actions: ln P_RGM(progress) + w ln P_F3(progress), with F3 imagining one
step from the real z3 and w = F3's precision relative to RGM's, calibrated offline (F3 -0.51 vs RGM -0.72 mean
log-likelihood of the real next state) and updated from reality. Progress = states one induction layer closer to the
goal. Mean steps when solved: lr 10 = 215 (fastest robust variant). Failure analysis: level 3 is only consulted for
the first 270 env steps (28 calls); every failure is an ant still 2-3 layers from the goal by then, typically
repeating one move that keeps failing. Across all 11 runs above, every task was solved 3-9 times: all solvable,
none reliable.

Replays (fresh runs, `replays/`, tasks 1, 6, 10; camera | maze map with trail and goal):

| planner | task 1 | task 6 | task 10 |
|---|---|---|---|
| H-JEPA gradient planner | 290 | 301 | 295 |
| Tschantz, 15-step plan | x | 141 | 322 |
| RGM closed | x | 183 | 160 |
| RGM commit | 306 | x | 315 |
| RGM adaptive | x | 341 | 301 |
| RGM feedback | 312 | x | 105 |
| RGM feedback + learning | x | 266 | x |

Many replay outcomes contradict the table run of the same planner on the same task (e.g. RGM closed fails task 1,
gd solves task 10, adaptive solves tasks 6 and 10). Single runs cannot rank these planners; repeated runs are needed.

Adaptive mode: at each call, continue the committed route or switch to a fresh one (extended until RGM predicts
P(goal) >= 0.5), whichever has lower G = -ln P(goal), with a 0.5-nat switching cost. It is clearly worse. Diagnosis
(task 1): the ant oscillates between states 10 and 28 for the whole episode at distance 2 from the goal, while every
fresh route still predicts P(goal) ~ 0.53. RGM keeps trusting transitions that reality keeps refusing; the rule looks
only forward and never compares its predictions with what actually happened. This motivates a feedback mode
(surprise of the real outcome against the plan's prediction triggers replanning) and online learning of RGM's counts
from real transitions.

Mean steps when solved: gd 284, closed 287, commit 292. On the 6 tasks all three solve: gd 277, closed 292, commit 314.
Planner wall clock per task: gd 76 s, RGM ~47 s.

RGM: K = 64 discrete states (k-means on the whitened z3), 12 macro-action prototypes as paths, two levels
(level-2 event = state + two-action word), planning by `spm_induction`-style backward induction.
Offline (held-out episodes): a path found for 100/100 start/goal pairs; closed loop inside F3 reaches the goal state
86/100.

**Caveat — run-to-run noise.** The same gd planner solved task 0 in 321 steps in an earlier smoke run on this PC and
failed it here; RGM-commit likewise (238 then fail). Runs with identical seeds are not deterministic on this GPU,
so 9 vs 10 of 13 is within noise. Repeats are needed before any claim.
