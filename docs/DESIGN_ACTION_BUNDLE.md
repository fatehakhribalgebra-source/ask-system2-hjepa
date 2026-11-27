# Actions as a fibre bundle — design v2 (27 Nov 2026, branch `animal-v1`)

**The free lunch is composition.** An upper-level action is not a separate thing to learn or a vocabulary to
invent. It *is* a composition of lower-level actions. The upper action space is a manifold, and each of its points
is the image of a fibre of lower-level chunks. Each level gets its actions for free from what the level below can
already do.

**v2 change: the manifold is a Lie group, not a sphere.** H-JEPA's upper actions live on a 2-sphere only because its
action encoder ends in `LayerNorm(4)`. That sphere cannot express composition. v2 puts actions in a group where
"compose" is the group product: the **free nilpotent Lie group**, whose elements are truncated **path signatures**.
Composition is then exact and free (Chen's identity), with no learned encoder deciding what a composition means.

This replaces design C3 (FSQ codes) and L1.1 (pattern-driven birth vocabulary) in [DESIGN_ANIMAL.md](DESIGN_ANIMAL.md).
Design only: no code yet.

---

## 1. What H-JEPA does today (read from the code at `BASE_COMMIT`)

| Fact | Where |
|---|---|
| Level ℓ's macro-action is `action_encoder(chunk)`: a 1-layer transformer over a chunk of **lower-level action embeddings**, with a CLS head and then `LayerNorm(4)` | `models/encoders/seq_encoder.py: SequenceEncoder`; called in `hierarchical_solver.py:_encode_actions_up` |
| AntMaze: stride 2 at levels 2 and 3, so one level-ℓ action composes **L = 2** level-(ℓ−1) actions | `config/train/ant_hjepa_l3.yaml` |
| Level 1 actions are blocks of ant torques (8-d × frameskip); levels 2 and 3 take 4-d codes | same config |
| Each level keeps a queue of 4,000 **real** composed actions | `JEPA.latent_action_queue` (in the checkpoint) |

`LayerNorm(4)` maps any input to x̂ with Σx̂ᵢ = 0 and ‖x̂‖² = 4: a 2-sphere, made an ellipsoid by γ, β. This is why every
real level-3 action has norm 2.033 ± 0.006 (19 Nov).

## 2. Why the sphere is the wrong manifold

| Needed for composed actions | Sphere (LayerNorm(4)) |
|---|---|
| **Composition is an operation of the space**: compose(a₁, a₂) is again an action, and compose(compose(a₁,a₂),a₃) = compose(a₁,compose(a₂,a₃)) | No. Composition is whatever a learned transformer outputs; nothing makes it associative or consistent across levels |
| **Identity**: "do nothing" (stand still) | No distinguished point |
| **Inverse**: undo a move (maze moves can be undone; `rgm_curious.link_goal` already assumes it) | No |
| **Size**: a short step vs a long run; norm carries information | Removed: every action has the same norm |
| **Order matters but is measurable**: left-then-forward ≠ forward-then-left, and by how much | Not represented explicitly |
| **Dimension from the structure**, not from a config value | Fixed at 2 by `output_dim: 4` |
| **Same kind of space at every level**, so the tower is uniform | Only by accident (each level has its own learned sphere) |

## 3. Alternatives considered

| Manifold | Composition | Identity / inverse | Size and order | Training to compose | Verdict |
|---|---|---|---|---|---|
| Sphere S² (today) | learned, unconstrained | no / no | no / no | yes (the encoder) | replace |
| Vector space of outcomes (Δz) | a₁ + a₂ (approximate: Δz is not additive in a curved latent) | yes / yes | size yes, order no | none | too weak: loses order |
| Hyperbolic space | none natural | — | — | yes | for hierarchies of *states* (trees), not composition of actions |
| Learned matrix Lie group (exp of learned generators; homomorphism-autoencoder style) | product g₁g₂, exact | yes / yes | yes / yes | yes, and needs symmetry assumptions (SE(2) suits a maze, not Push-T) | strong, but the group is guessed per world |
| **Free nilpotent Lie group (path signatures)** | **Chen: S(p₁ ⋆ p₂) = S(p₁) ⊗ S(p₂), exact** | **yes / yes (reverse the path)** | **yes / yes (Lévy area)** | **none** | **chosen** |

The free nilpotent group is the *universal* group of composed paths: every other group of actions (SE(2), a learned
matrix group) is a quotient of it. So it assumes nothing about the world or the body, which the animal needs.
A learned group can still be added later as a quotient on top (section 7).

## 4. The chosen manifold: truncated signatures

Take the lower-level actions as a path in a small base space ℝᵈ (section 5 says which path). Its **step-N log-
signature** is a point of the free nilpotent Lie algebra 𝔤_N(ℝᵈ):

- step 1, **ℝᵈ**: the net increment (where the chunk goes overall);
- step 2, **Λ²ℝᵈ**: the Lévy areas, the signed area between each pair of channels (the order: which turn came first);
- step 3 and up: finer order structure.

Dimension of step 2: d + d(d−1)/2. For d = 4 that is 10; step 3 adds 20 more.

**Composition at the top is closed form.** One level-ℓ action composes L lower actions. With their log-signatures
a₁, a₂, the composed action is the group product, given by the Baker–Campbell–Hausdorff series, which terminates
at step N:

```
step 2:  log(e^{a₁} e^{a₂}) = a₁ + a₂ + ½[a₁, a₂]
```

There is no learned encoder in this step. The upper action is exactly the composition of the lower ones.

**Each level reads the same group at a coarser step.** Level ℓ keeps the log-signature truncated at step N_ℓ, with
N_ℓ growing with ℓ (for example N₂ = 2, N₃ = 3). A longer composed path carries more order structure, and the level
above keeps more of it. The manifold above *contains* the manifold below as its low-step part. For the product to be
exact, the lower actions must be stored to step N_ℓ (the step of the level that composes them), even if level ℓ−1
itself plans and clusters only on its first N_{ℓ−1} steps.

### 4.1 The bundle, now with a group

```
             π_ℓ = group product (BCH, truncated at N_ℓ)
 E_ℓ = G_{ℓ−1}^L  ─────────────────────────────────────►  G_ℓ = G_{N_ℓ}(ℝᵈ)
 (L lower actions)                                          (one upper action)
```

- **Fibre** F_g = π⁻¹(g): every factorisation of g into L lower actions, i.e. every way of doing g below.
- Fibres are **tree-like equivalence classes** of paths: a chunk with an out-and-back detour has the same signature as
  the chunk without it. This is the right notion of "the same upper move".
- **Vertical moves** (along the fibre: same g, a different factorisation) and **horizontal moves** (change g) are
  defined by the group itself, not by a learned Jacobian. This keeps the stuck reflex (R7): on a failed hop, try
  another factorisation of the same g before choosing a new g.
- **Usable region** U_ℓ: the part of G_ℓ reached by compositions the lower level can really perform. At birth it is
  the image of composing the lower repertoire, and it grows with it. The upper level never learns it separately.

## 5. Which path the signature is taken of

Two choices, and v2 uses both:

1. **Intent path** (what was commanded): lower action embeddings, projected to d channels.
   - Level 2: torque blocks → d channels by PCA of real torques (or a few fixed primitives).
   - Level 3: the level-2 log-signatures themselves (composition by BCH, section 4).
2. **Outcome path** (what happened): the lower-level latent trajectory over the chunk, projected to d channels (PCA
   of real z_{ℓ−1}, the PaperRGM whitening). This is "actions as outcomes" (25 Nov notes) made exact: an upper action
   *is* the signature of the path it caused below.

The two meet in the fibre-consistency check: commanded compositions with the same intent signature should give
outcome signatures that agree, and where they do not (walls, contact) the residual says so (Counterfactual WM's
failure case, now measured per hop).

Time is added as an extra channel of the path. Otherwise signatures ignore speed (they are invariant to
reparametrisation), and here duration matters.

## 6. How it connects to H-JEPA

The release model expects its own LayerNorm codes, so there are two stages:

- **v2a, no retraining (System 2 only).** Fit a small map from log-signatures to H-JEPA's codes, h: 𝔤_N → S²,
  on real chunks (the queue plus our recorded episodes). System 2 plans and counts in the group; h hands the frozen
  F3 the code it understands. h is only an adapter, used where the group action is in U_ℓ.
- **v2b, H-JEPA v2 fine-tune (with T1 in DESIGN_ANIMAL §2b).** Replace `action_encoder_ℓ` with the BCH composition of
  lower log-signatures (no parameters) and retrain only `action_embed_ℓ` and the predictor to take log-signatures.
  The upper level's actions then *are* compositions inside the model too.

## 7. Moves, planning and learning on the group

- **Countable moves.** Cluster the executed upper actions in the group (log coordinates, whitened) the same way
  PaperRGM clusters z3. Each cluster is a move; the RGM counts B[s, k, s′] per move. This is what C3 wanted, without
  FSQ.
- **Inverses for free.** The reverse of move k is −a_k (step-1 part flips, Lévy areas flip). An untried reverse hop
  t → s gets a prior from the tried s → t.
- **Counterfactuals.** A horizontal change uses ẑ′ = z′ + F(z, g′) − F(z, g). A vertical change (same g) should leave
  z′ unchanged; a large residual flags contact.
- **Learned quotient (later).** If a world has a real symmetry (SE(2) for a maze), learn a homomorphism from the free
  group onto that smaller group. Moves that differ only in ways the world ignores then merge.

## 8. What has to be true (checks, in order)

| Step | Check | Pass if |
|---|---|---|
| **B0 Base path** | Choose d for intent and outcome paths (PCA on real torques and real z2); compute step-2/3 log-signatures of real chunks | ≥ 90% of chunk variance kept at small d (≤ 6) |
| **B1 Composition matches the model** | For real level-3 chunks: BCH(a₁, a₂) from the level-2 parts vs the signature of the whole lower path; and h(g) vs the release code | The BCH error is ~0 (it is exact up to truncation); h predicts the release code well on held-out chunks (R² ≥ 0.9) |
| **B2 Fibre consistency** | Execute two factorisations of the same g from the same state (one process per world); compare real z3 | Within-fibre spread of z3′ ≪ between-move spread |
| **B3 Group vs sphere** | Explorer that composes in the group vs the current 4 ± axis codes vs random sphere points; same budget, one machine, ≥ 3 repeats | More successful hops, more coverage, a better-connected play map |
| **B4 Moves and inverses** | Moves = clusters in the group; reverse-move priors on | Hop prediction and planning ≥ the move-free map on the same experience |
| **B5 Vertical escape** | On a failed hop, try another factorisation before a new g | Stuck time and failed-hop rate fall in the life run |

## 9. Where it goes in the code (when we build it)

- New `system2/action_group.py`: log-signature of a path (step N, with a time channel), BCH product, inverse,
  `factorisations(g)` (sample the fibre), the adapter h, and move clustering.
- `experiments/eval_l3_planner.py`: remove the `dir_protos` / `f3_dir` block; level-3 proposals are group products of
  level-2 actions, sent to F3 through h.
- `system2/rgm_curious.py`: `_pattern_step` and the `fractal` / `fractal_shuffled` / `noise_labels` explorers go; a
  `compose` explorer takes their place; each hop records (move, factorisation).
- H-JEPA v2 (v2b): `action_encoder` → parameter-free BCH; `action_embed` and predictor retrained on log-signatures.
- Dependency: signatures are short closed-form code for step ≤ 3 (or `signatory` / `iisignature`); BCH to step 3 is a
  few lines.

## 10. Open questions

1. **Truncation per level:** N₂ = 2 and N₃ = 3, or the same step everywhere?
2. **Base channels d:** as small as 4, or keep more channels at a higher dimension cost?
3. **Bottom of the tower:** raw torques as the base path, or a few fixed primitive gaits?
4. **v2a first** (an adapter on the frozen model, quick to test), or go straight to v2b with the H-JEPA v2 fine-tune?
