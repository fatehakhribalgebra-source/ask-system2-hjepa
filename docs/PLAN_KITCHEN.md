# LEXA RoboKitchen build (target 4 Dec 2026)

Same System 2 as AntMaze (RGM places + routes over a JEPA latent, a reflex below), on a second kind of world:
manipulation. The Content–Geometry paper shows the latent "places" also track arm configuration and object state
(Reacher: shoulder turn = one smooth loop; Cube: two sheets, free vs grasped, that meet at grasp/release). Routes over
places (approach → grasp → move → release) are where a planner on raw latent distance struggles, which is the case
for System 2.

## Pieces

| Piece | Choice | Why |
|---|---|---|
| World | Gymnasium-Robotics `FrankaKitchen-v1` (modern MuJoCo, Windows OK) | LEXA's RoboKitchen objects: kettle, microwave, burners, light, cabinets |
| Data | Minari `D4RL/kitchen` mixed-v2 (621 ep, 156,560 steps; partial-v2 is the same episodes) + complete-v2 (19 ep) | Full qpos/qvel in every obs → exact re-render |
| Frames | 256 px, rendered from stored states (`experiments/kitchen/render_dataset.py`) | Replay of recorded actions stays within 0.01 rad over 50 steps |
| Encoder | **Frozen V-JEPA 2.1** ViT-L/16 (image mode, `(B,C,1,H,W)`), pre-encoded once | Strong pretrained JEPA; no 2.1 action-conditioned predictor is released |
| Head | Small trained head on pooled tokens, **Slow SVReg** (no SG, slow channels, gated residual predictor) | The paper's "two-stage" (frozen SG trunk + SVReg head): Cube 86.8%, PushT 86.7% vs Slow SVReg 89.1 / 87.3 |
| Predictor | Action-conditioned, gated residual, levels of 1 / 4 / 16 env steps | Same hierarchy idea as H-JEPA; env step = 0.08 s |
| System 2 | RGM over the top level: k-means places, counted transitions, routes back from the goal place | Unchanged recipe from AntMaze |
| Reflex | CEM/gd on the predictor toward the next place's real exemplar latent; gd stays at the bottom for now | Torques need a continuous solver; RGM at more levels during generalisation week |

## Measures

- **Content** (before planning): ridge readouts on held-out episodes of arm joints, end-effector, each object's qpos;
  contact/grasp AUC. Raw V-JEPA tokens vs our head.
- **Geometry**: jaggedness J on held-out episodes, cost-to-goal ρ (latent L2 vs true state distance).
- **Planning**: LEXA-style goal images (single objects and combinations) from env reset; success = env's own task
  thresholds. Report success and paired steps to goal for (a) planner on latent distance only, (b) RGM + reflex.
  LEXA's numbers are context only (online learning vs our offline data).

## Days

| Date | Step | Done when |
|---|---|---|
| 27 Nov | Env, data, rendering | Frames for all 160k steps on disk ✔ (running) |
| 28 Nov | Encode with V-JEPA 2.1; content readouts on raw features | We know if frozen V-JEPA sees the arm, kettle, doors, switch |
| 29–30 Nov | Train head + predictor (Slow SVReg); geometry + content checks | J near 1, readouts kept, ρ above raw features |
| 1–2 Dec | RGM map, reflex, goal eval harness | One goal solved end to end |
| 3–4 Dec | Full eval, paired with the latent-distance planner | Table for the paper |

## Risks

- **Sim-to-video gap:** V-JEPA 2.1 was trained on real video; small objects (light switch, knobs) may not show in
  pooled features. Day-1 readouts decide. Fallback: train our own encoder on the rendered frames with Slow SVReg.
- **Data coverage:** demonstrations cover the 4–7 task objects but not much else; places off the demos are missing.
  That is the "play will travel at scale" question for later.
- **Speed:** 384 px is V-JEPA's native size; we use 256 (RoPE interpolation) to keep encoding fast.
