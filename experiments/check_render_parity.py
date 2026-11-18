"""Compare our (Windows/GLFW) MuJoCo renders with the authors' stored eval pixels at identical states."""
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
from system2.hjepa_bridge import build_policy  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from experiments.antmaze_explore import CKPT, loco_of  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

p = torch.load(ROOT / 'H-JEPA/h_jepa/assets/eval_trajs/ant/expert_grid_d3_n50.pt', weights_only=False)
policy, solver, cfg, model = build_policy(str(CKPT).format(seed=42), 'ant_l3')
cfg.world.max_episode_steps = 50; cfg.world.num_envs = 1
world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(64, 64), verbose=0)
world.set_policy(policy); world.reset(seed=0)
loco = loco_of(world)
diffs, outs = [], []
for ep in range(5):
    d = p['data'][ep]
    for t in (0, len(d['pixels']) // 2):
        loco.env.set_state(d['qpos'][t].numpy(), d['qvel'][t].numpy())
        ours = loco.env.get_ob().astype(np.float32)
        theirs = d['pixels'][t].permute(1, 2, 0).numpy().astype(np.float32)
        diffs.append(np.abs(ours - theirs).mean())
        outs.append(np.concatenate([theirs, ours, np.abs(ours - theirs) * 4], 1).clip(0, 255).astype(np.uint8))
print('mean |ours - theirs| per pixel (0-255):', np.round(diffs, 2))
import imageio
imageio.imwrite(ROOT / 'results' / 'render_parity.png', np.concatenate(outs[:6], 0))
world.close()
