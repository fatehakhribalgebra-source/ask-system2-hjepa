"""How much did each arm learn? Offline counts in the model vs updates accepted online (and carried, for agentp).

  python experiments/learning_amount.py results/paper_eval_pc3
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system2.rgm_agent import RGMAgent  # noqa: E402

root = sys.argv[1]
for s in (42, 43, 44):
    base = RGMAgent.from_paper(pickle.load(open(f'data/rgm_emeans_z3_k64_seed{s}.pkl', 'rb')))
    offline = base.N_real.sum()
    for v in ('agent', 'agentp'):
        files = sorted(glob.glob(os.path.join(root, v, f'seed{s}', 'rgm_paper_task*.json')))
        st = [json.load(open(f)).get('rgm_stats') or {} for f in files]
        kept = sum(x.get('gate_kept', 0) for x in st); rej = sum(x.get('gate_rejected', 0) for x in st)
        moves = sum(x.get('moves_ok', 0) + x.get('moves_failed', 0) for x in st)
        line = (f'seed {s} {v:6s}: {len(files):2d} tasks | offline level-1 counts {offline:,.0f} | online updates kept '
                f'{kept} (rejected {rej}), {kept / max(len(files), 1):.1f} per task | move outcomes recorded {moves}')
        if v == 'agentp':
            p = os.path.join(root, v, f'model_seed{s}.pkl')
            if os.path.exists(p):
                m = pickle.load(open(p, 'rb'))
                line += (f' | carried model: +{m.N_real.sum() - offline:.0f} counts '
                         f'({100 * (m.N_real.sum() - offline) / offline:.2f}% of offline), '
                         f'{int(m.att.sum())} move attempts, {len(m.blocked)} moves refused')
        else:
            line += ' | nothing carried: every task restarts from the offline model'
        print(line)
