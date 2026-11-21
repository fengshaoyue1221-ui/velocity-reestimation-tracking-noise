"""
Parse MTR eval logs produced by run_noise_sweep.sh and generate:
  results/tab_main.tex      main table: clean vs noisy, per class, clean-trained vs noise-aug
  results/tab_ablation.tex  per-noise-type ablation (clean-trained model)
  results/fig_degradation.pdf  mAP / minADE / MissRate vs. noise LEVEL
  results/numbers.tex       \\newcommand macros for numbers quoted in the text

Usage (from MTR root):
  python collect_results.py --root output/waymo/mtr+20p_noise_eval --tags clean aug \
         --names "MTR (clean-trained)" "MTR (noise-aug)" --out paper/results
"""
import argparse, glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LEVELS = ['0.0', '0.5', '1.0', '2.0', '3.0']
ABL = [('only_pos', 'Position jitter'), ('only_head', 'Heading jitter'), ('only_vel', 'Velocity error'),
       ('only_drop', 'Frame dropout'), ('only_trunc', 'History truncation'), ('only_idswap', 'ID switch')]
CLASSES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
LINE = re.compile(r'^\s*([A-Za-z]+(?: - [A-Z]+)?)\s*:\s*([-\d.eE]+)\s*$')


def parse_log(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = re.sub(r'^.*?INFO\s+', '', line.rstrip())  # strip logger prefix if any
            m = LINE.match(line)
            if m and m.group(1).split(' - ')[0] in ('minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP'):
                out[m.group(1)] = float(m.group(2))
    return out


def load(root, tag, eval_tag):
    logs = sorted(glob.glob(os.path.join(root, tag, 'eval', 'epoch_*', eval_tag, 'log_eval_*.txt')))
    for p in reversed(logs):  # latest log that actually contains metrics
        r = parse_log(p)
        if 'mAP' in r:
            return r
    return None


def fmt(v, nd=3):
    return '--' if v is None else f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--main_level', default='1.0')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {t: {lv: load(a.root, t, f'level_{lv}') for lv in LEVELS} for t in a.tags}
    A = {t: {k: load(a.root, t, k) for k, _ in ABL} for t in a.tags}
    g = lambda r, k: None if r is None else r.get(k)

    # ---------- main table ----------
    L = a.main_level
    rows = []
    for t, n in zip(a.tags, a.names):
        for cond, lv in [('Clean', '0.0'), (f'Noisy ($\\lambda$={L})', L)]:
            r = R[t][lv]
            cells = [fmt(g(r, 'mAP')), fmt(g(r, 'minADE')), fmt(g(r, 'MissRate'))]
            cells += [fmt(g(r, f'mAP - {c}')) for c in CLASSES]
            rows.append(f'{n} & {cond} & ' + ' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    rows = rows[:-1]
    tab = r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{All classes} & \multicolumn{3}{c}{mAP per class $\uparrow$} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Model & Eval input & mAP$\uparrow$ & minADE$\downarrow$ & MR$\downarrow$ & Veh. & Ped. & Cyc. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Motion forecasting on the WOMD validation set with clean and perturbed histories.
Noise is applied to observed history only; ground-truth futures are clean.}
\label{tab:main}
\end{table}
"""
    open(os.path.join(a.out, 'tab_main.tex'), 'w').write(tab)

    # ---------- ablation table (first tag = clean-trained) ----------
    t0 = a.tags[0]
    base = R[t0]['0.0']
    rows = []
    for k, name in ABL:
        r = A[t0][k]
        d = None if (r is None or base is None) else (r['mAP'] - base['mAP'])
        rows.append(f"{name} & {fmt(g(r,'mAP'))} & {fmt(d)} & {fmt(g(r,'minADE'))} & "
                    + ' & '.join(fmt(g(r, f'mAP - {c}')) for c in CLASSES) + r' \\')
    tab = r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Noise type (alone) & mAP$\uparrow$ & $\Delta$mAP & minADE$\downarrow$ & Veh. & Ped. & Cyc. \\
\midrule
Clean & """ + f"{fmt(g(base,'mAP'))} & -- & {fmt(g(base,'minADE'))} & " + ' & '.join(fmt(g(base, f'mAP - {c}')) for c in CLASSES) + r""" \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Effect of each perturbation type in isolation on the clean-trained model.}
\label{tab:ablation}
\end{table}
"""
    open(os.path.join(a.out, 'tab_ablation.tex'), 'w').write(tab)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, m, lab in zip(axes, ['mAP', 'minADE', 'MissRate'], ['mAP $\\uparrow$', 'minADE (m) $\\downarrow$', 'Miss rate $\\downarrow$']):
        for t, n in zip(a.tags, a.names):
            xs = [float(lv) for lv in LEVELS if R[t][lv] is not None]
            ys = [R[t][lv][m] for lv in LEVELS if R[t][lv] is not None]
            ax.plot(xs, ys, marker='o', ms=3, label=n)
        ax.set_xlabel('noise level $\\lambda$'); ax.set_title(lab, fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'fig_degradation.pdf'))

    # ---------- macros for numbers in text ----------
    def pct_drop(t, lv, k='mAP'):
        c, n = R[t]['0.0'], R[t][lv]
        return None if (c is None or n is None) else 100 * (c[k] - n[k]) / c[k]
    mac = []
    for i, t in enumerate(a.tags):
        tagname = 'ABCDEFG'[i]
        for lv in LEVELS[1:]:
            v = pct_drop(t, lv)
            word = {'0.5': 'half', '1.0': 'one', '2.0': 'two', '3.0': 'three'}[lv]
            mac.append(f"\\newcommand{{\\drop{tagname}{word}}}{{{fmt(v,1)}\\%}}")
        for c in CLASSES:
            v = pct_drop(t, L, f'mAP - {c}')
            mac.append(f"\\newcommand{{\\drop{tagname}{c.lower()}}}{{{fmt(v,1)}\\%}}")
        v = pct_drop(t, L)
        mac.append(f"\\newcommand{{\\drop{tagname}main}}{{{fmt(v,1)}\\%}}")
    open(os.path.join(a.out, 'numbers.tex'), 'w').write('\n'.join(mac) + '\n')
    print('wrote', os.listdir(a.out))


if __name__ == '__main__':
    main()
