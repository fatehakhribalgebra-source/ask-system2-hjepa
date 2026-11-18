# System 2 over H-JEPA — working design (v0)

Living design notes. This replaces the spec PDF as the working reference: the general idea is kept,
the details are re-derived against the real H-JEPA code.

## The idea in one paragraph

A frozen hierarchical JEPA world model (H-JEPA) gives a latent space at several timescales and a
planner that can reach a latent goal. On top of it sits a goal generator with no reward: a fixed
"Platonic" pattern source (fractal, chaotic attractor, cellular automaton, art…) drives a fixed
reservoir, whose state is read out into candidate goals in the top-level latent. Candidates are
scored by an expected-free-energy-like objective (reachability + information gain + optional
preference), the best one is handed to H-JEPA's top-down planner as its goal, and the agent acts.
Question under test (Free Lunch, Ertle/Levin/Scheutz 2025): does structured pattern input produce
better exploration than unstructured input, once you lift it from raw actions into a world-model latent?

## Components

### System 1 — H-JEPA (kevinghst/H-JEPA, frozen)
- Levels ℓ=1..L. Encoders E(ℓ), action encoders A(ℓ), predictors F(ℓ). Level 1 = LeWM (ViT-Tiny).
- **Every level is a planner, not a policy**: actions at each level are optimised by gradient descent
  so the rollout reaches the goal latent (level L) or the subgoals from the level above (ℓ<L).
- Integration point: the top level normally gets g(L) = E(L)(...E(1)(o_goal)). We replace g(L)
  with the System-2 goal z*. No goal image is needed.

### Pattern sources ("Platonic seeds")
Streams p_t ∈ R^d, read sequentially; read position reset per episode (as in Free Lunch).
- Halley fractal of x(x^a − b), a∈[2,6], b∈[1,12], centred at 0, 4 colour labels (Free Lunch Alg. 2)
- Lorenz attractor, Lévy flight, Rule 30, Game of Life
- NGA open-access artworks, colour-quantised to 4 labels (Free Lunch used 60; exact list unknown)
- **Controls**: pixel-shuffled versions of each pattern (same label histogram, no structure),
  uniform noise, Gaussian noise. These controls are the core of the experiment.

### Reservoir (fixed)
r_t = (1−λ) r_{t−1} + λ tanh(W_res r_{t−1} + W_p p_t + W_z z_t), with sparse W_res and ρ ≈ 0.9–0.99.
The context input W_z is weaker than W_p. No training of W_res, W_p, W_z.

### Candidate goals
K candidates per decision, each from a **different seed stream offset** p_t^(k) (not just noise
on one reservoir state): z^(k) = z_t + clip(W_out r_t^(k)). Bound the step norm so goals stay in the
reachable shell of the top-level latent (calibrated from data: typical ‖z_{t+H} − z_t‖).

### Scoring (EFE-flavoured, deterministic latents)
G(k) = β·E_reach(k) − γ·I_epi(k) − ν·N(k) + α·D_pref(k)
- **E_reach**: run the top-level planner for a few GD iterations toward z^(k), and measure the residual
  ‖ẑ_H − z^(k)‖². This uses the real F(L), not a separate head.
- **I_epi**: an ensemble of M top-level predictors, trained online on the agent's own top-level
  transitions (bootstrapped batches). Use the disagreement of their rollouts under the planned macro-actions.
- **N** (novelty): kNN distance of z^(k) to an episodic memory of visited top-level latents. This is cheap
  and robust, and it is the standard practical stand-in for information gain in exploration.
- **D_pref**: distance to a preference prior μ_c. It is **off** for pure exploration. It is the hook
  for VLM / goal images / generative-model priors later.

### Readout learning
The spec's "minimise G of the winner w.r.t. W_out" degenerates (it can blow up the −γ term). Instead:
- **Hindsight regression**: after executing toward z*, record (r_t^(k*), z_{t+H} − z_t) and fit W_out by
  online ridge/RLS so that the reservoir learns to propose **reachable** displacements, conditioned
  on pattern state. Weight samples by realised novelty, which acts as a preference for productive patterns.
- Variant to compare: a fixed random W_out (pure "free lunch", no learning at all).

## Environments

1. **Free-Lunch maze (reproduction + bridge)**: Kruskal mazes, 11×11, path scale 2, 4 actions
   (N,E,S,W). Rendered top-down. For H-JEPA, discrete actions are embedded as continuous
   vectors (N=(0,1), E=(1,0), S=(0,−1), W=(−1,0)); the planner optimises in R², and execution snaps
   to the nearest prototype. Also a continuous point-mass version where the 2D vector is the actual command.
   Data: collect our own (random walk + noisy shortest-path), as H-JEPA does for FourRoom.
   Metric: % of maze explored in N steps (Free Lunch metric).
2. **Visual AntMaze (OGBench, medium)**: pretrained H-JEPA L3 from HF. Exploration only.
   Metric: number of distinct maze cells visited and coverage-vs-steps curve, from the true xy.
3. **Crafter**: 17 discrete actions. The embedding has movement actions as 2D direction vectors, and
   the others are learned or fixed vectors. The planner works in the continuous space and snaps to the nearest action.
   Data: CrafterPlay (CrafterDojo, CC0) and/or self-collected random + exploratory rollouts.
   Metric: Crafter achievement success rates / Crafter score, with no reward used.

## Baselines and ablations
- Random-action policy; Free-Lunch direct pattern→action mapping (no world model)
- Random latent goals (Gaussian in Z(L)); novelty-only goal selection (no patterns)
- Pattern vs shuffled pattern vs noise (the key comparison); each pattern family
- Learned vs fixed W_out; with vs without each G term; L=1/2/3 hierarchy depth

## Order of work
1. Run pretrained H-JEPA AntMaze eval to confirm the stack works on this PC (RTX 5070 Ti, Windows).
2. Implement pattern sources + reservoir + scorer as a standalone package (`ask/system2/`).
3. Hook System 2 into H-JEPA's planner as a goal provider; AntMaze exploration runs.
4. Free-Lunch maze env + data + train small H-JEPA; reproduce Free Lunch numbers, then with System 2.
5. Crafter: data, action embedding, train H-JEPA, System 2 runs.
