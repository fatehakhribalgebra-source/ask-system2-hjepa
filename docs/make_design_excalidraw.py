"""Write docs/ASK_Animal_design.excalidraw: the ASK Animal design (docs/DESIGN_ANIMAL.md) as a hand-drawn board.

Open the output at https://excalidraw.com (menu -> Open, or drag the file onto the page).
  python docs/make_design_excalidraw.py
"""
import json
import pathlib
import random

OUT = pathlib.Path(__file__).with_name('ASK_Animal_design.excalidraw')
rng = random.Random(7)
els = []
n = 0

BLUE, GREEN, YELLOW, WHITE, PINK = '#a5d8ff', '#b2f2bb', '#ffec99', '#ffffff', '#ffc9c9'
INK, RED, GREY = '#1e1e1e', '#e03131', '#868e96'


def _id():
    global n
    n += 1
    return f'el{n}'


def _base(t, x, y, w, h, **kw):
    e = dict(id=_id(), type=t, x=x, y=y, width=w, height=h, angle=0, strokeColor=kw.get('stroke', INK),
             backgroundColor=kw.get('bg', 'transparent'), fillStyle=kw.get('fill', 'solid'),
             strokeWidth=kw.get('sw', 1), strokeStyle=kw.get('style', 'solid'), roughness=kw.get('rough', 1),
             opacity=100, groupIds=[], frameId=None, roundness=kw.get('round', {'type': 3}), seed=rng.randint(1, 2**31),
             version=1, versionNonce=rng.randint(1, 2**31), isDeleted=False, boundElements=[], updated=1,
             link=None, locked=False)
    els.append(e)
    return e


def text(x, y, s, size=16, color=INK, align='left', container=None, w=None, h=None):
    lines = s.split('\n')
    tw = w if w is not None else max(len(l) for l in lines) * size * 0.56
    th = h if h is not None else len(lines) * size * 1.25
    e = _base('text', x, y, tw, th, stroke=color, round=None)
    e.update(text=s, originalText=s, fontSize=size, fontFamily=1, textAlign=align,
             verticalAlign='middle' if container else 'top', containerId=container, lineHeight=1.25,
             autoResize=True, baseline=int(size))
    return e


def box(x, y, w, h, s, bg=BLUE, size=16, stroke=INK, style='solid', sw=1, fill='solid'):
    r = _base('rectangle', x, y, w, h, bg=bg, stroke=stroke, style=style, sw=sw, fill=fill)
    lines = s.split('\n')
    th = len(lines) * size * 1.25
    tw = min(w - 16, max(len(l) for l in lines) * size * 0.56)
    t = text(x + (w - tw) / 2, y + (h - th) / 2, s, size=size, align='center', container=r['id'], w=tw, h=th)
    r['boundElements'].append({'id': t['id'], 'type': 'text'})
    return r


def section(x, y, w, h, title, color=GREY):
    _base('rectangle', x, y, w, h, stroke=color, style='dashed', sw=2, round={'type': 3})
    text(x + 8, y - 34, title, size=24, color=INK)


def arrow(pts, both=False, color=INK, dashed=False, label=None):
    x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    xs = [p[0] for p in rel]; ys = [p[1] for p in rel]
    a = _base('arrow', x0, y0, max(xs) - min(xs), max(ys) - min(ys), stroke=color, sw=2,
              style='dashed' if dashed else 'solid', round={'type': 2})
    a.update(points=rel, lastCommittedPoint=None, startBinding=None, endBinding=None,
             startArrowhead='arrow' if both else None, endArrowhead='arrow', elbowed=False)
    if label:
        mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
        text(mx - len(label) * 4.5, my - 26, label, size=16, color=color)


# ---------------------------------------------------------------- title
text(0, -230, 'ASK Animal: design v1', size=48)
text(0, -150, 'Plan on what was seen; dreams only propose experiments', size=28, color=RED)
text(1480, -150, 'Key:  blue = part of the agent   green = seen (real)\n'
                 '      white = dreamed            yellow = hand over to H-JEPA', size=16, color=INK)

# ---------------------------------------------------------------- A. the layers
section(0, 0, 1400, 1236, 'One life: four layers over the world')
LX, BX, BW, BH = 24, [248, 534, 820, 1106], 262, 180
rows = [
    (40, '**L3 Routes**\nwords and sentences\nslowest, most abstract', BLUE, [
        ('Routes = frequent place\nsequences, counted in sleep\n(RGM upper levels)', BLUE),
        ('A route is one L3 move;\nits hops go down to L2', BLUE),
        ('Long goals: plan over\nroutes, then over places', BLUE),
        ('Compression:\nletters -> words -> sentences', BLUE)]),
    (284, '**L2 Places**\nthe map, learned\nfrom experience', BLUE, [
        ('Evidence: real z3 only\nsliding window:\nplace(t-20) -> place(t)\n20x more transitions', GREEN),
        ('Places = k-means in\nwhitened z3; K so neighbours\nare one reflex step apart;\nrefit in every sleep', BLUE),
        ('Risk planner on seen edges\nV = discounted reach prob.\nscore = -ln V', BLUE),
        ('Unsure? Hand over:\nif V(goal) < theta, H-JEPA\nplans; theta learned where\nthe RGM loses to H-JEPA', YELLOW)]),
    (528, '**L1 Moves**\nwhat I can do\nfrom here', BLUE, [
        ('Birth: Free Lunch moves\na fractal label stream holds\none direction for a while\n-> persistent headings', BLUE),
        ('Growth: outcome moves\na move = a displacement\nthat worked from this place\n(per-place DP, shared prior)', BLUE),
        ("Counterfactual moves (CWM)\nz(a') = z' + F3(z,a') - F3(z,a)\nskipped at contacts, walls\n-> go to Dreamed only", WHITE),
        ('Later: latent actions\nthe same move in a maze,\nPush-T or a game\n(LAC-WM, SCAR)', BLUE)]),
    (772, '**L0 Body**\nH-JEPA v2 reflex\n(levels 1-2)', BLUE, [
        ('Reach the target latent\nreal exemplars only,\nnever a dreamed latent', BLUE),
        ('Every replan (AdaJEPA):\n1 gradient step on the first\npredictor block, last 5 steps', BLUE),
        ('Whole life (JEPA-TTT):\npersistent predictor, dense\nreplay, encoder frozen;\ngoal mode: optimism (OpTTA)', BLUE),
        ('Stuck reflex: no motion\nfor 40 steps -> random burst,\nedge marked failed; log\ntorso height, orientation', BLUE)]),
    (1016, '**World**\npixels in,\nactions out', WHITE, [
        ('AntMaze\n(now)', WHITE), ('Push-T\n(code ready)', WHITE),
        ('LEXA\nRoboYoga, RoboKitchen', WHITE), ('Abstract worlds\n(later: games, proofs)', WHITE)]),
]
for y, lab, lbg, blocks in rows:
    box(LX, y, 200, BH, lab.replace('**', ''), bg=lbg, size=20, sw=2)
    for x, (s, bg) in zip(BX, blocks):
        box(x, y, BW, BH, s, bg=bg, size=16, style='dashed' if bg == WHITE and y == 1016 else 'solid')
for y in (284, 528, 772, 1016):
    arrow([(124, y), (124, y - 64)], both=True)

# ---------------------------------------------------------------- B. self
section(1480, 0, 820, 440, 'S2 Self: who decides')
box(1504, 40, 374, 150, 'Exploit\nroute to the goal on seen edges', BLUE)
box(1902, 40, 374, 150, "Hand over\nH-JEPA's gradient planner plans;\nits trip is still recorded", YELLOW)
box(1504, 214, 374, 150, 'Explore here\ntest an untried or dreamed exit\n(Beta information gain)', BLUE)
box(1902, 214, 374, 150, 'Explore there (Go-Explore)\ntravel to the frontier on seen\nedges, then explore with moves', BLUE)
text(1504, 380, 'Choose by expected free energy: risk -ln V + eta * (information gain + learning progress).\n'
                'A sustained surprise opens a new map in the library (CSCG-style clones).', size=14)

# ---------------------------------------------------------------- C. memories
section(1480, 520, 820, 300, 'Two memories, kept apart')
box(1504, 560, 300, 220, 'Seen\nreal sliding-window counts;\nBeta made/failed per edge\n-> the ONLY input to planning',
    GREEN, sw=2)
box(1976, 560, 300, 220, 'Dreamed\nF3 rollouts + counterfactual\nmoves; trust learned per region\n-> only proposes experiments',
    WHITE, style='dashed')
arrow([(1976, 670), (1808, 670)], dashed=True, label='real hop confirms')
# seen -> risk planner (L2), routed through the gap under the L2 row
arrow([(1504, 700), (1440, 700), (1440, 496), (951, 496), (951, 468)], color='#2f9e44', label=None)
text(1180, 500, 'plans use Seen only', size=14, color='#2f9e44')

# ---------------------------------------------------------------- D. sleep loop
section(1480, 900, 820, 360, 'Sleep: at each goal throw, or when new places pile up')
s1 = box(1504, 940, 340, 106, '1 Refit places: k-means on all\nreal z3, K at reflex scale;\nmerge by Bayesian model reduction', BLUE, size=15)
s2 = box(1936, 940, 340, 106, '2 Recount sliding-window\ntransitions per move,\nthen routes (L3)', BLUE, size=15)
s3 = box(1936, 1130, 340, 106, "3 Fine-tune H-JEPA v2 on the\nlife's dense windows\n+ 50% replay of the offline set", BLUE, size=15)
s4 = box(1504, 1130, 340, 106, '4 Re-learn trust: how often\nsimilar dreams came true,\nper region and rollout depth', BLUE, size=15)
arrow([(1848, 993), (1932, 993)])
arrow([(2106, 1050), (2106, 1126)])
arrow([(1932, 1183), (1848, 1183)])
arrow([(1674, 1126), (1674, 1050)])
# sleep refits the places (L2)
arrow([(1504, 993), (1460, 993), (1460, 480), (665, 480), (665, 468)], color='#1971c2')
text(700, 448, 'sleep refits places', size=14, color='#1971c2')

# ---------------------------------------------------------------- E. H-JEPA v2
section(0, 1340, 1400, 300, 'H-JEPA v2: retrain, warm-started from the release checkpoints')
cs = [
    'C1 Slow SVReg\nreplaces SIGReg at L2-L3\nsmooth z3 -> good places;\nslow channels keep objects',
    'C2 Gated residual\nz_next = z + c*f(z,u)\npredict the change;\nslow factors are cheap',
    'C3 Discrete moves\nFSQ on the 4-d action\ncode (81 codes), seeded\nby a Free Lunch basis\n-> RGM counts per move',
    'C4 Time-to-goal head\nlevel-3 steps between\ntwo z3 -> risk, theta, K',
    'C5 Fast / slow split\nfast: first block + gate,\nadapted during life;\nslow: changed in sleep',
    'C6 Sleep fine-tune\nlife\'s dense windows\n+ 50% offline replay\n(no forgetting)',
]
for i, s in enumerate(cs):
    box(24 + i * 229, 1380, 205, 220, s, BLUE, size=15)

# ---------------------------------------------------------------- F. experiments
section(1480, 1340, 820, 470, 'Experiments, in order (same machine, >= 3 repeats)')
exps = [('Step', 'What', 'Pass if'),
        ('E0', 'deterministic runs, one machine', 'run-to-run spread is known'),
        ('E0b', 'z3 jaggedness, probe R2, hops', 'baseline for T1'),
        ('T1', 'C1-C4 on L2-L3, warm-start', 'J down, hops up, eval >= release'),
        ('E1', 'no dreams, sleep refits, hand-over', '>= episodic pipeline, same goals'),
        ('E2', "add 'explore there'", 'more cells, bigger connected map'),
        ('E3', 'Free Lunch vs shuffled vs random', 'fractal > shuffled on coverage'),
        ('E4', 'dreams only pick experiments', 'coverage up, planning not worse'),
        ('E5', 'AdaJEPA + JEPA-TTT, heavy ant', 'goals improve over the life'),
        ('E6', 'CWM moves + residual gate', 'better first-try hop success'),
        ('E7', 'Push-T, LEXA, map library', 'world 1 kept after world 2')]
for i, (a, b, c) in enumerate(exps):
    y = 1376 + i * 38
    _base('rectangle', 1504, y, 772, 38, bg=YELLOW if i == 0 else 'transparent', stroke=GREY, round=None)
    for x, s in ((1514, a), (1590, b), (1930, c)):
        text(x, y + 9, s, size=15)

# ---------------------------------------------------------------- G. lessons
section(0, 1720, 1400, 340, 'Why: what our own runs showed')
lessons = [('Result', 'Rule it forces'),
           ('Dreams counted as experience: 11/13 -> 4/13; life 3/20', 'Plan only on seen; dreams propose'),
           ('Play map worked when H-JEPA took over (77% of steps)', 'The RGM hands over when unsure'),
           ('Radius-grown live states: ~30% of hops made', 'Places one reflex step apart, refit in sleep'),
           ('Sliding window: 13/13 vs 6/13 on the same play', 'Keep it (= JEPA-TTT dense replay)'),
           ('Curious play stayed within ~3 units of birth', 'Travel to the frontier first'),
           ('Ant stuck from step ~9800, never recovered', 'Stuck reflex in the body'),
           ('Same map: 13/13, 17/24, 13/24 on other machines', 'Compare on one machine, with repeats')]
for i, (a, b) in enumerate(lessons):
    y = 1756 + i * 36
    _base('rectangle', 24, y, 1352, 36, bg=GREEN if i == 0 else 'transparent', stroke=GREY, round=None)
    text(36, y + 9, a, size=15)
    text(700, y + 9, b, size=15)

# ---------------------------------------------------------------- H. papers
section(1480, 1890, 820, 400, 'Five papers: where each one plugs in')
papers = [('AdaJEPA', '1 gradient step per replan -> L0 fast loop\n(unseen mazes 53 -> 79%)'),
          ('JEPA-TTT', 'persistent predictor + dense replay -> L0 for life\n(-83% latent error, +153% planning)'),
          ('OpTTA', 'anchored optimism, ramped over a task\n-> goal mode in L0; the rule for dreams'),
          ('Content-Geometry', 'Slow SVReg: smooth maps, kept content\n-> H-JEPA v2 C1, C2; places'),
          ('Counterfactual WM', "z' + F3(z,a') - F3(z,a), fails at contacts\n-> L1 counterfactual moves")]
for i, (p, s) in enumerate(papers):
    y = 1926 + i * 70
    box(1504, y, 200, 56, p, PINK, size=16)
    text(1724, y + 6, s, size=15)

# ---------------------------------------------------------------- open questions (stickies)
text(0, 2110, 'Open questions', size=24)
for i, q in enumerate(['Free Lunch = a fixed\nstructured code on the\naction manifold?',
                       'LEXA: train a JEPA\nor use the LeWM release?',
                       'Sleep at every goal\nthrow, or rarer?']):
    box(i * 300, 2150, 270, 110, q, YELLOW, size=16)

doc = {'type': 'excalidraw', 'version': 2, 'source': 'https://excalidraw.com',
       'elements': els, 'appState': {'viewBackgroundColor': '#ffffff', 'gridSize': None}, 'files': {}}
OUT.write_text(json.dumps(doc, indent=1), encoding='utf-8')
print(f'wrote {OUT} ({len(els)} elements)')
