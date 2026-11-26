# ASK Animal — design v1 (26 Nov 2026)

A hierarchical agent that learns its own map from experience it gathers by curiosity, starts moving with a Free Lunch
action vocabulary before it has learned anything, and adapts quickly when the world or the body changes.

This design replaces nothing that works. It keeps H-JEPA as System 1 and the paper-style RGM as the planner. It changes
**where the evidence goes, who decides, and when the agent learns.** Every rule below is tied to a result we already
have (section 1) or to one of the five papers (section 2). Section 7 lists the experiments in order, each with a
pass/fail test.

---

## 0. The design in one paragraph

Five layers, each slower and more abstract than the one below:

| Layer | What it is | Learns how fast | Built from |
|---|---|---|---|
| **L0 body** | the H-JEPA reflex, levels 1–2 | every replan (seconds) | H-JEPA predictor, updated online (AdaJEPA / JEPA-TTT) |
| **L1 moves** | the action vocabulary: what the agent can do from here | every hop (one level-3 step) | Free Lunch patterns at birth, then outcome-moves |
| **L2 places** | the map: places and the moves between them | each awake bout + each sleep | real z3 only, sliding window, consolidated in sleep |
| **L3 routes** | sequences of places: "words and sentences" | in sleep | RGM upper levels, counted over the place stream |
| **S2 self** | decides: explore, exploit or hand over; detects a new world | every decision | expected free energy, with trust learned from outcomes |

There are two memories, kept strictly apart: **seen** (what happened) and **dreamed** (what the world model says could
happen). Plans run only on *seen*. Dreams only propose experiments, and a dream becomes *seen* only when reality
confirms it. **Sleep** rebuilds the place map cleanly from the real stream. This is the 24 Nov pipeline (explore → build
K=64 `PaperRGM` → plan with H-JEPA fallback), done as a recurring phase of one life instead of an offline step.

---

## 1. What our own results force on the design

| # | Result | Where | Rule it forces |
|---|---|---|---|
| R1 | Imagination counted as experience makes planning worse, three times over: sleep-dreaming (24 Nov), F3 imagined counts (risk screen 11/13 → 4/13, 26 Nov), and the life run (446 states, 138 real, 0–2 of 13 hops made per bout) | NOTES 24, 26 | **Plan only on seen evidence.** Dreams are hypotheses, never counts. |
| R2 | The play map scored well on 24 Nov mainly because H-JEPA planned when the RGM had no path (RGM decided ~17–23%) | NOTES 25, 26 | **The RGM must know when it does not know** and hand over to H-JEPA. A planner that always finds a route (link_goal + "still go") is worse. |
| R3 | Radius-grown live states make ~30% of hops, whatever the map size. k-means states in whitened z3 (PaperRGM) work | NOTES 24 (stream maps) | **Places must be one reflex step apart.** Build them by clustering the real stream at that scale, not by growing radius balls online. |
| R4 | The sliding window gives 20× the transitions, and it matters (13/13 with it, 6/13 without, same play) | NOTES 24 | Keep it. JEPA-TTT's dense replay is the same idea for the reflex. |
| R5 | Curious play stays local. In the life it never left ~3 units of birth, and the episodic play maps are fragmented (seed 42: 18/64 strongly connected) | NOTES 25, 26 | **Exploration must travel to the frontier,** not only test exits near home. |
| R6 | The risk planner (reach probability, −ln V) helps on connected maps (8 → 11 and 7 → 11 on the screens) | NOTES 25, 26 | Keep it as the L2 planner. |
| R7 | The ant got stuck from step ~9800 and never recovered (no resets) | NOTES 26 | **L0 needs a stuck reflex** and must log body state (height, orientation). |
| R8 | The same map, command and checkpoint gave 13/13, 17/24 and 13/24 on different machines | NOTES 26 | **Compare only on one machine, with repeats.** Fix nondeterminism before any claim. |
| R9 | Calibrated trust in F3 barely moved; local experience was the signal that grew (CalibRGM) | NOTES 24 | Trust in imagination is learned **per region from tests**, and starts low. |

---

## 2. What the five papers give us

| Paper | Key result | What we take |
|---|---|---|
| **AdaJEPA** (Wang, Bounou, LeCun, Ren 2026) | One gradient step per replan, last encoder + predictor layers, 5 recent transitions. Unseen mazes 53.3 → 78.7% (GD, first predictor block + last encoder stage). With little data, PushObj 28.1 → 60.8% | **L0 fast loop:** adapt the reflex inside MPC, cheaply (+0.01–0.03 s per replan). |
| **JEPA-TTT** (Zhang et al. 2026) | Predictor-only updates that **persist across episodes**, encoder frozen, dense replay (a window at every offset). −83% latent error and +153% planning after 500 episodes | **L0 slow loop:** the reflex keeps what it learned for the whole life. The frozen encoder keeps L2's places stable. |
| **OpTTA** (Mete, Kalathil, Kumar 2026) | Prediction-only adaptation fits only the outcomes of the agent's own actions, so it can stay wrong about the actions that reach the goal (closed-loop identifiability). Adding the planner's goal cost, with an optimism weight α_k ∝ √(k/K_max), gives +13.5 pp vs +6.9 for AdaJEPA | **Bounded, anchored optimism:** optimism helps only while anchored to observed transitions, and it ramps up over a task. The same principle governs dreams at L2 (section 5). |
| **Content–Geometry** (Ren et al. 2026) | Stop-gradient JEPAs give jagged latents. SVReg (a variance floor on random slices, no stop-gradient) gives smooth maps. Slow SVReg keeps slow content. First-person Medium hard goals: SG 21.2% → Slow SVReg 92.9%. Jaggedness predicts planning | **Places are only as good as the latent's geometry.** Measure jaggedness on our z3. If H-JEPA's z3 is jagged, a small SVReg adapter on top of it is the cheapest fix. |
| **Counterfactual WMs** (Balestriero, LeCun, Varaku 2026) | ẑ(a′) = z′ + μ(z,a′) − μ(z,a): one real outcome plus the model's difference between two actions. O(h²) error, counterfactual R² 38–46% → 86–96%. **Fails at contacts, folds, and command-scaling disturbances** | **Grounded imagination for L1:** one tried move tells us about the untried moves from the same place. Use it in open space only; a large residual means contact (a wall). |

Related work:
- **Free Lunch** (Ertle, Levin, Scheutz, ICDL 2025): fixed structured patterns (a Halley fractal, art) guide exploration better than unstructured ones.
- **CSCG** (George et al. 2021): one observation, several "clones" by context. This is how a map handles aliasing and remaps in a new room.
- **LEXA** (Mendonca et al. 2021): an explorer and an achiever trained in one world model, reaching goals zero-shot.
- **Learning-progress curiosity** (ICLR 2026): reward the improvement of the model, not its error, so noise is ignored.
- **Latent-action models** (LAC-WM, SCAR 2026): an action space learned from observation pairs, shared across bodies.
- **WARP-JEPA** (2026): amortised planners for frozen H-JEPA, 9.6–18.8× faster.

---

## 2b. H-JEPA v2: the world model changes too (retrain, reusing the release)

Today (`H-JEPA/h_jepa/config/train/ant_hjepa_l3.yaml`, `loss.py`, `models/jepa.py`):
- Every level is trained end to end with **SIGReg** (weight 0.09, no stop-gradient), after a BatchNorm-MLP projector.
- Each level predicts with a plain transformer (`ARPredictor`).
- Level 2 and level 3 "actions" are a **learned 4-d LayerNorm code** of the level-below latent chunk (`action_encoder`,
  `output_dim: 4`). So the upper action is already an outcome code, but a continuous one that the RGM cannot count.

| # | Change | From | Why for us | Reuse |
|---|---|---|---|---|
| C1 | **Slow SVReg** in place of SIGReg-BN at levels 2–3 (variance floor on random slices, ~1/8 of dims as slow channels) | Content–Geometry | Places are k-means clusters of z3: they need smooth geometry (jaggedness predicts planning; first-person hard maze SIGReg-BN 42.8% → Slow SVReg 92.9%). Slow channels keep slow content (Push-T objects) | Warm-start every weight from the release checkpoint; only the loss changes |
| C2 | **Gated residual predictor**: ẑ(t+1) = z(t) + c ⊙ f(z, u), with c₀ = 0.1 | Content–Geometry | Slow factors become cheap to predict. "Predict the change" is also the form the counterfactual rule uses | Wrap the existing `ARPredictor`; the gate is the only new parameter |
| C3 | **Discrete moves at upper levels**: FSQ quantisation of the 4-d action code (e.g. 3⁴ = 81 codes), initialised from a Free Lunch structured basis (fractal-ordered directions on the manifold) | Free Lunch + "actions as outcomes" | The RGM can count transitions **per move** (a real B tensor), the L1 vocabulary is the codebook, and the Free Lunch prior is built into the model | Same action encoder plus a quantiser; codes are fine-tuned |
| C4 | **Time-to-goal head** at level 3: predicts how many level-3 steps separate two z3, trained from the same trajectories | Time-to-Goal (2026), H-JEPA's own data | A calibrated metric for the risk planner, the hand-over threshold θ, and "one reflex step" when choosing K | A small MLP on frozen z3 pairs |
| C5 | **Fast/slow parameter split for life**: first predictor block + gate c = fast (adapted online, AdaJEPA/JEPA-TTT, OpTTA in goal mode); everything else slow (changes only in sleep) | AdaJEPA, JEPA-TTT, OpTTA | Fast adaptation without moving the encoder under the map | A parameter-group flag, no new weights |
| C6 | **Sleep fine-tuning on the life's own stream**: dense windows of the life, mixed 50/50 with replay of the offline set | JEPA-TTT dense replay | The world model improves where the animal lives, without forgetting the rest | The existing trainer with a mixed loader |
| C7 | (Later) **Shared upper levels across worlds**: per-world level-1 encoder and latent-action encoder, shared levels 2–3 | LAC-WM / SCAR latent actions | Moves and places transfer between worlds; nothing assumes a body | Level 1 per world from its own release checkpoint |

**Training plan (reuse first):**
- **T1:** freeze level 1 (ViT-tiny); fine-tune levels 2–3 with C1 + C2 + C3 + C4 on the AntMaze training set. This needs
  `visual_antmaze_medium_explore_stitch_train` from the H-JEPA dataset release; the size is still to be checked.
- **T2:** unfreeze level 1 only if T1 improves geometry but not hop success.
- **T3:** Push-T, the same recipe from its own release checkpoint.

**Pass if:**
- jaggedness J of z3 along recorded routes falls, and the held-out xy probe R² stays the same or rises;
- place-hop success on the K=64 map rises (offline-map baseline, same machine);
- the paper eval (50 tasks × 3 seeds) is at least the release model's, so we lose nothing.

---

## 3. The layers

### L0 — body (H-JEPA levels 1–2, the reflex)

- **What it does:** reaches the level-3 target it is handed (a real exemplar latent, never a dream latent; R1).
- **Fast adaptation (AdaJEPA):** after each executed chunk, take one gradient step on the latent prediction loss over
  the last ~5 transitions. Update the level-1/2 predictors' first block, plus the last encoder stage only if visuals
  shift. Use the training learning rate.
- **Persistent adaptation (JEPA-TTT):** keep the adapted predictor and a dense-replay buffer for the whole life.
  Freeze the encoders, so L2's places do not drift under the map.
- **Optimism (OpTTA):** add α_k·L_plan only in goal mode, ramped by √(k/K_max). Never during play, because play has no
  goal cost.
- **Stuck reflex (R7):** if displacement over 40 env steps is below ε while actions are non-zero, run a short bout of
  random level-1 actions. Then mark the current L2 edge as failed. Log torso height and orientation to tell stuck from
  flipped.
- **Done when:** the per-step latent error on the life's own transitions falls over the life, and the stuck time falls.

### L1 — moves (the action vocabulary; where Free Lunch lives)

An upper-level action is **"go to that place"**, a displacement in the level below (NOTES 25: actions as outcomes). So
the vocabulary is not one fixed manifold. It is, per place, the set of displacements that have worked from there.

1. **Birth: the Free Lunch vocabulary.** Before any experience there are no outcomes to choose from. The agent gets a
   *fixed, structured* code on level 3's action manifold. The code has M directions; a Halley-fractal label stream
   decides which direction is held and for how long, which gives long runs (persistent headings) and occasional turns.
   This is Free Lunch moved from raw actions to the action embedding. It costs nothing to learn, and its structure is
   what makes the body travel (R5). Its control is the pixel-shuffled stream: same label histogram, no structure.
2. **Growth: outcome moves.** Each executed hop s → t adds the observed displacement (z3(t) − z3(s), plus the
   macro-action used) to s's vocabulary. Similar displacements are merged: a Dirichlet process over displacements per
   place, shared across places through a common prior. That prior is the "schema" that transfers to new places.
3. **Counterfactual moves (CWM).** For a tried move (z, a, z′), every untried move a′ from the same place gets
   ẑ(a′) = z′ + F3(z, a′) − F3(z, a). These go into the **dreamed** memory, with depth 0 ("one step from a real
   outcome"), which starts with more trust than a free F3 dream. They are skipped when ‖z′ − F3(z, a)‖ is large
   (contact or wall: the case where the paper says the method fails).
4. **Later, other worlds:** replace level 3's own action encoder with a latent-action model (inverse dynamics on z3
   pairs, as in LAC-WM/SCAR). Then "move" means the same thing in a maze, in Push-T and in a game, and nothing assumes
   a body.

### L2 — places (the map)

- **Evidence:** the real z3 stream only, through the sliding window (R4). Each env step t counts
  place(t − 20) → place(t).
- **Two memories (R1, R9):**
  - *Seen:* Dirichlet transition counts plus a Beta success/failure per tried edge.
  - *Dreamed:* edges proposed by F3 rollouts and by counterfactual moves. Each carries a trust learned per region from
    the first real test of similar dreams (the existing `dream_prior` kernel). A dreamed edge enters *seen* only after
    a real traversal.
- **Places while awake:** the current consolidated K-place model assigns each z3 to a place. A z3 too far from every
  place, beyond the 95th percentile of within-place distance, goes into a *new-place buffer*. It is never a new state
  on the spot, which is the R3 failure.
- **Places in sleep:** refit the places by k-means on all real z3 so far, in PaperRGM's whitened space. Choose K so
  that neighbouring places are one reflex step apart (`build_rgm_stream.py --k auto` logic). Recount the transitions
  over the sliding window, and merge redundant places by Bayesian model reduction. This step is the 13/13 pipeline,
  run inside the life.
- **Planner:** the risk planner (R6) on *seen* only: V = discounted reach probability, score −ln V.
- **Handing over (R2):** if V(goal) from here < θ, or the goal is not near any seen place, System 2 hands level 3 to
  H-JEPA's own gradient planner. That trip is still recorded as experience. θ is learned: it is the V below which the
  RGM's own success rate falls under H-JEPA's success rate on the same kind of goal.

### L3 — routes

- Upper RGM levels counted over the place stream in sleep: routes = frequent place sequences (PaperRGM's 3 levels of
  items).
- They are used for long goals: a route is one "move" at L3, and its internal hops go to L2. This is also where
  compression shows ("letters, words, sentences").

### S2 — self (who decides)

Every decision chooses among four modes by expected free energy:
**exploit** (a route to the goal on *seen*), **hand over** (H-JEPA plans), **explore here** (test an untried or dreamed
edge), **explore there** (travel to the frontier on *seen*, then explore).

- **Risk:** −ln V, as in RiskRGM.
- **Epistemic value:** expected information gain on Beta edges, *plus* **learning progress**. Learning progress is the
  drop in L0's prediction error at a place over the last visits (ICLR 2026 LPM). Places where the model cannot get
  better, such as noise or stuck corners, lose their pull. This also discounts the "curiosity satisfied by dreams near
  home" failure of the life.
- **Frontier travel (Go-Explore inside the map; R5):** a place's frontier value = how much there is to learn there
  (untested exits, high learning progress, few confirmed exits). Pick the best frontier place by value minus travel
  risk, go there on *seen* edges, then use Free Lunch moves or outcome moves to expand.
- **New-world detection (fast adaptation to new situations):** keep a library of maps, one per context, as CSCG keeps
  clones. If the recent z3 stream's likelihood under the current map drops below every stored map's (a sustained
  surprise spike), open a new map that inherits the move vocabulary's shared prior. Otherwise switch to the best stored
  map. Moves and the reflex transfer; places do not have to.
- **Sleep schedule:** sleep when the new-place buffer is full (default ≥ 5% of awake steps), at each goal throw, and
  after a context switch.

---

## 4. One life, step by step

```
born, empty seen memory, Free Lunch vocabulary, frozen H-JEPA (+ TTT buffers)
loop
  awake (each level-3 tick = 20 env steps):
    L0: follow the target; record z3 every env step (sliding window); AdaJEPA step after each chunk; stuck reflex
    S2: mode = argmin EFE over {exploit, hand over, explore here, explore there}
    L1: chosen move -> target latent (real exemplar, or fresh F3 step for an explore move)
    L2: score the hop made/failed; failed dreamed edge -> trust down in that region
    dreams (cheap): F3 + counterfactual proposals -> dreamed memory only
  sleep (when due):
    refit places (k-means at reflex scale) on all real z3; recount windows; model reduction
    recount routes (L3); refit dream trust; JEPA-TTT minibatches for the reflex
  goal thrown -> S2 exploit / hand over; play again after reaching it
```

---

## 5. Why this should beat both the 24 Nov pipeline and the 26 Nov life

| | 24 Nov episodic pipeline | 26 Nov life run | This design |
|---|---|---|---|
| Coverage comes from | resets to 50 task starts | local curiosity (fails) | frontier travel + Free Lunch headings |
| Map evidence | real only | real + dreams mixed | real only; dreams kept apart |
| Places | k-means K=64, offline | radius-grown live | k-means at reflex scale, refit in sleep |
| When the RGM is unsure | hands to H-JEPA (77%) | never hands over | hands over below a learned V threshold |
| Body adapts | no | no | AdaJEPA + JEPA-TTT, stuck reflex |
| New world | n/a | n/a | map library + surprise switch |

The common thread is OpTTA's lesson applied at every level: **optimism is useful only when anchored to what was
observed.** Dreams propose and reality decides; optimism in the reflex ramps up and stays tied to L_pred.

---

## 6. What could make this wrong (and how we would see it)

- **H-JEPA's z3 is jagged** (Content–Geometry). Then no clustering gives one-reflex-step places. *Check first:*
  jaggedness J and place hop success on the offline map. *Fix:* an SVReg adapter on z3, trained on the life's stream.
- **Free Lunch headings hit walls** and waste steps. *Check:* coverage per step vs random and vs the shuffled pattern.
  If structure does not beat shuffle, drop it to a fallback.
- **Online TTT drifts** the reflex (no stop-gradient issue, but forgetting). *Check:* the offline eval before and after
  a life. *Fix:* a smaller learning rate or predictor-first-block only, as AdaJEPA's ablation suggests.
- **Counterfactual moves through walls.** AntMaze is full of contacts. *Check:* the residual gate's false-accept rate on
  tried moves.
- **Evaluation noise hides everything (R8).** That is why step E0 comes first.

---

## 7. Experiments, in order (each removes one assumption)

| Step | What | Pass if (same machine, ≥ 3 repeats) |
|---|---|---|
| **E0b Measure z3 geometry** | Jaggedness J along recorded routes, xy-probe R², place-hop success of the release z3 | Gives the baseline that T1 (H-JEPA v2, section 2b) must beat |
| **T1 H-JEPA v2 fine-tune** | Slow SVReg + gated residual + FSQ moves + time-to-goal head on levels 2–3, warm-started | Section 2b pass criteria |
| **E0 Reproducibility** | Set deterministic torch, pin CUDA and EGL/glfw behaviour, one process per run. Rerun the offline map and the curious B50 map on tasks 0–12, ×3 | The run-to-run spread is known; per-task flips are < 2/13, or else we report means of repeats only |
| **E1 Life v2 (minimal)** | Life with no dreams, places refit in sleep at each goal throw (PaperRGM fit on the life's own stream), risk planner on *seen*, H-JEPA handover below θ | ≥ the episodic curious pipeline on the same 20 goals, same machine |
| **E2 Frontier travel** | Add "explore there" (Go-Explore in the map) vs local curiosity only | Life coverage (cells) and largest strongly connected set grow; goal success does not drop |
| **E3 Free Lunch vocabulary** | Fractal headings vs shuffled vs random vs F3-random at birth | Fractal > shuffled on coverage and on map connectivity at 7000 steps |
| **E4 Dreams as hypotheses** | Dreamed memory proposes explore targets only, vs no dreams | Coverage per step improves; plan success unchanged or better (never the R1 drop) |
| **E5 Reflex adaptation** | AdaJEPA step + JEPA-TTT buffer; perturbed ant (mass ×0.5, damping ×5) | Hop success and goals improve over the life under perturbation; no drop on default dynamics |
| **E6 Counterfactual moves** | CWM proposals with the residual gate | Higher first-try success of proposed hops than free F3 dreams |
| **E7 New worlds** | Push-T (code ready) and LEXA RoboYoga/RoboKitchen; map library + context switch | A second world is learned without losing the first (success on world 1 after world 2) |

Free Lunch as action embedding is first tested in **E3**. That fits the plan to try it after the life run.

---

## 8. Open questions for the user

1. **Free Lunch as action embedding.** This design reads it as "a fixed structured code on the action manifold that
   gives the newborn persistent, diverse movement." If your idea is different (for example, the embedding *itself* is
   built from the pattern, or patterns map to outcome displacements), L1.1 changes, but the rest stays.
2. **LEXA:** it needs a JEPA trained on its environments (none exists in the H-JEPA release). Train one, or use the
   LeWM release?
3. **Sleep cost:** a K-means refit on ~10⁵ latents takes seconds. Is sleeping at every goal throw acceptable, or
   should sleep be rarer?
