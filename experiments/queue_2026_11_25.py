"""Jobs for 25 Nov 2026 (run with experiments/run_queue.py).

1. Play map on the full paper protocol: curious explore dumps for seeds 43/44 (seed 42's exist), K=64 sliding-window
   maps per seed, 50 tasks x 3 seeds; the offline map re-run on the same 50 x 3 on this machine as the paired baseline.
2. Risk planner (counts -> hop probabilities, -ln V) on the same maps: screen on tasks 0-12 (seed 42).
4. Free Lunch explorer (fractal vs its pixel shuffle) -> play maps -> tasks 0-12; data-decided K; one-step F3
   imagined counts from play under the risk planner.
"""
import sys

PY = sys.executable
EVAL = 'experiments/eval_l3_planner.py'
ALL = [str(t) for t in range(50)]
SCREEN = [str(t) for t in range(13)]


def ev(map_path, seed, tasks, out, *extra):
    return [PY, EVAL, '--planner', 'rgm_paper', '--rgm-obs-level', '3', '--seed', str(seed),
            '--rgm-paper-path', map_path, '--tasks', *tasks, '--out', out, *extra]


def explore(seed, arm, explorer):
    return [PY, 'experiments/run_curious.py', '--arm', arm, '--explorer', explorer, '--stream', '--dump-stream',
            '--budgets', '50', '--skip-eval', '--seed', str(seed)]


def build(seed, arm, out, k='64'):
    return [PY, 'experiments/build_rgm_stream.py', '--streams', f'results/curious/seed{seed}/{arm}/explore',
            '--episodes', '50', '--k', k, '--out', out]


JOBS = []
# ---- 1. play map, full protocol --------------------------------------------------------------------------------
for s in (43, 44):
    JOBS.append(dict(name=f'explore_curious_s{s}', cmd=explore(s, 'dump_curious', 'curious')))
JOBS.append(dict(name='build_play_s42', cmd=build(42, 'dump_curious', 'data/rgm_stream_curious_B50_seed42.pkl')))
for s in (43, 44):
    JOBS.append(dict(name=f'build_play_s{s}', after=[f'explore_curious_s{s}'],
                     cmd=build(s, 'dump_curious', f'data/rgm_stream_curious_B50_seed{s}.pkl')))
# risk screens first (they decide whether the risk planner gets the full protocol too)
JOBS.append(dict(name='risk_play_s42_screen', after=['build_play_s42'],
                 cmd=ev('data/rgm_stream_curious_B50_seed42.pkl', 42, SCREEN, 'results/risk_screen/play_s42',
                        '--rgm-risk')))
JOBS.append(dict(name='risk_base_s42_screen',
                 cmd=ev('data/rgm_paper_z3_k64_seed42.pkl', 42, SCREEN, 'results/risk_screen/base_s42', '--rgm-risk')))
for s in (42, 43, 44):
    JOBS.append(dict(name=f'play_s{s}', after=[f'build_play_s{s}'],
                     cmd=ev(f'data/rgm_stream_curious_B50_seed{s}.pkl', s, ALL, f'results/paper_eval_play/play_seed{s}')))
for s in (42, 43, 44):
    JOBS.append(dict(name=f'base_s{s}',
                     cmd=ev(f'data/rgm_paper_z3_k64_seed{s}.pkl', s, ALL, f'results/paper_eval_play/base_seed{s}')))
# ---- 4. Free Lunch explorer, data-decided K, imagined counts ----------------------------------------------------
for exp in ('fractal', 'fractal_shuffled'):
    JOBS.append(dict(name=f'explore_{exp}_s42', cmd=explore(42, f'dump_{exp}', exp)))
    JOBS.append(dict(name=f'build_{exp}_s42', after=[f'explore_{exp}_s42'],
                     cmd=build(42, f'dump_{exp}', f'data/rgm_stream_{exp}_B50_seed42.pkl')))
    JOBS.append(dict(name=f'{exp}_s42_screen', after=[f'build_{exp}_s42'],
                     cmd=ev(f'data/rgm_stream_{exp}_B50_seed42.pkl', 42, SCREEN, f'results/stream_k64/{exp}_B50')))
JOBS.append(dict(name='build_autok_s42', cmd=build(42, 'dump_curious', 'data/rgm_stream_curious_B50_autok_seed42.pkl',
                                                    k='auto')))
JOBS.append(dict(name='autok_s42_screen', after=['build_autok_s42'],
                 cmd=ev('data/rgm_stream_curious_B50_autok_seed42.pkl', 42, SCREEN, 'results/stream_k64/curious_B50_autok')))
JOBS.append(dict(name='imag_play_s42', after=['build_play_s42'],
                 cmd=[PY, 'experiments/build_f3_transitions.py', '--seed', '42', '--streams',
                      'results/curious/seed42/dump_curious/explore', '--model', 'data/rgm_stream_curious_B50_seed42.pkl',
                      '--out', 'data/f3_play_curious_B50_seed42.npz']))
JOBS.append(dict(name='risk_imag_play_s42_screen', after=['imag_play_s42', 'risk_play_s42_screen'],
                 cmd=ev('data/rgm_stream_curious_B50_seed42.pkl', 42, SCREEN, 'results/risk_screen/play_imag_s42',
                        '--rgm-risk', '--rgm-imag-counts', 'data/f3_play_curious_B50_seed42.npz')))
