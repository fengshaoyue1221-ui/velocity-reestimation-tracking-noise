"""
Builds LaTeX tables for the supplementary analyses from finished evaluation logs and
the JSON files written by cv_baseline.py / distance_bins.py.

  tab_horizon.tex   relative mAP / minADE degradation at 3 s, 5 s, 8 s (clean-trained & noise-aug)
  tab_distance.tex  relative minADE increase by distance-to-ego bin
  tab_cv.tex        constant-velocity reference vs MTR under the same noise levels
  tab_classscale.tex per-class mAP drop with default vs. uniform class factors

Usage (from MTR root):
  python ../analysis/extra_tables.py --eval_base output/waymo/a2000_clean --extra ../paper/results/extra
"""
import argparse, glob, json, os, re

HZ = {'5': '3\\,s', '9': '5\\,s', '15': '8\\,s'}
CLS = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
LINE_H = re.compile(r'^(minADE|minFDE|MissRate|mAP) - TYPE_(VEHICLE|PEDESTRIAN|CYCLIST)_(5|9|15)\s*:\s*([-\d.eE]+)', re.M)
LINE_C = re.compile(r'^(minADE|minFDE|MissRate|mAP) - (VEHICLE|PEDESTRIAN|CYCLIST)\s*:\s*([-\d.eE]+)\s*$', re.M)
LINE_A = re.compile(r'^(minADE|minFDE|MissRate|mAP)\s*:\s*([-\d.eE]+)\s*$', re.M)


def log_text(eval_base, tag, eval_tag):
    fs = sorted(glob.glob(os.path.join(eval_base, tag, 'eval', 'epoch_*', eval_tag, 'log_eval_*.txt')))
    for f in reversed(fs):
        t = open(f).read()
        if LINE_A.search(t):
            return t
    return None


def horizons(t):
    out = {}
    for m, c, h, v in LINE_H.findall(t):
        out.setdefault(m, {}).setdefault(h, []).append(float(v))
    return {m: {h: sum(v) / len(v) for h, v in d.items()} for m, d in out.items()}


def per_class(t):
    return {(m, c): float(v) for m, c, v in LINE_C.findall(t)}


def overall(t):
    return {m: float(v) for m, v in LINE_A.findall(t)}


def rel(a, b):  # % change from a (clean) to b (noisy)
    return None if a in (None, 0) or b is None else 100 * (b - a) / abs(a)


def f(x, nd=1):
    return '--' if x is None else f'{x:+.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_base', required=True)
    ap.add_argument('--extra', required=True)
    ap.add_argument('--level', default='1.0')
    a = ap.parse_args()
    os.makedirs(a.extra, exist_ok=True)
    tags = [('clean', 'MTR (clean-trained)'), ('aug', 'MTR (noise-aug)')]

    # ---- horizon table ----
    rows = []
    for tag, name in tags:
        c, n = log_text(a.eval_base, tag, 'level_0.0'), log_text(a.eval_base, tag, f'level_{a.level}')
        if not (c and n):
            continue
        hc, hn = horizons(c), horizons(n)
        cells = []
        for m in ['mAP', 'minADE']:
            for h in ['5', '9', '15']:
                cells.append(f(rel(hc.get(m, {}).get(h), hn.get(m, {}).get(h))))
        rows.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    open(os.path.join(a.extra, 'tab_horizon.tex'), 'w').write(r"""\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}\toprule
 & \multicolumn{3}{c}{$\Delta$mAP (\%)} & \multicolumn{3}{c}{$\Delta$minADE (\%)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & 3\,s & 5\,s & 8\,s & 3\,s & 5\,s & 8\,s \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}
\caption{Relative change under noise ($\lambda{=}""" + a.level + r"""$) vs.\ clean input, per prediction horizon (averaged over classes).}
\label{tab:horizon}\end{table}
""")

    # ---- distance table ----
    rows = []
    for tag, name in tags:
        p = os.path.join(a.extra, f'dist_{tag}.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        labels = d['bins_m']; r = d['results'].get('ALL', {})
        rows.append(f'{name} & ' + ' & '.join(f(r.get(b, {}).get('minADE_rel_increase_%')) for b in labels) + r' \\')
        rows.append(r'\quad \#agents & ' + ' & '.join(str(r.get(b, {}).get('n', '--')) for b in labels) + r' \\')
    if rows:
        labels = json.load(open(os.path.join(a.extra, 'dist_clean.json')))['bins_m'] if os.path.exists(os.path.join(a.extra, 'dist_clean.json')) else ['0-10', '10-20', '20-30', '30-50', '>50']
        open(os.path.join(a.extra, 'tab_distance.tex'), 'w').write(r"""\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" + 'c' * len(labels) + r"""}\toprule
Distance to ego (m) & """ + ' & '.join(l.replace('>', '$>$') for l in labels) + r""" \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}
\caption{Relative minADE increase (\%) under noise ($\lambda{=}""" + a.level + r"""$) by distance of the target agent to the ego vehicle (plain displacement errors, 6 modes).}
\label{tab:distance}\end{table}
""")

    # ---- CV table ----
    p = os.path.join(a.extra, 'cv_baseline.json')
    if os.path.exists(p):
        cv = json.load(open(p))
        rows = []
        for lv in sorted(cv, key=float):
            mtr = log_text(a.eval_base, 'clean', f'level_{float(lv):.1f}')
            mo = overall(mtr) if mtr else {}
            rows.append(f'{float(lv):.1f} & {cv[lv].get("minADE", float("nan")):.3f} & {cv[lv].get("MissRate", float("nan")):.3f} & '
                        f'{mo.get("minADE", float("nan")):.3f} & {mo.get("MissRate", float("nan")):.3f}' + r' \\')
        open(os.path.join(a.extra, 'tab_cv.tex'), 'w').write(r"""\begin{table}[t]\centering\small
\begin{tabular}{ccccc}\toprule
 & \multicolumn{2}{c}{Constant velocity} & \multicolumn{2}{c}{MTR (clean-trained)}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
$\lambda$ & minADE$\downarrow$ & MR$\downarrow$ & minADE$\downarrow$ & MR$\downarrow$ \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}
\caption{A constant-velocity extrapolation of the corrupted current state vs.\ MTR under identical corruptions.}
\label{tab:cv}\end{table}
""".replace('nan', '--'))

    # ---- class-scale table ----
    rows = []
    for tag, name in tags:
        c = log_text(a.eval_base, tag, 'level_0.0')
        d = log_text(a.eval_base, tag, f'level_{a.level}')
        u = log_text(a.eval_base, tag, 'uniform_class')
        if not (c and d and u):
            continue
        pc, pd, pu = per_class(c), per_class(d), per_class(u)
        rows.append(f'{name}, default factors & ' + ' & '.join(f(rel(pc.get(("mAP", k)), pd.get(("mAP", k)))) for k in CLS) + r' \\')
        rows.append(f'{name}, uniform factors & ' + ' & '.join(f(rel(pc.get(("mAP", k)), pu.get(("mAP", k)))) for k in CLS) + r' \\')
    if rows:
        open(os.path.join(a.extra, 'tab_classscale.tex'), 'w').write(r"""\begin{table}[t]\centering\small
\begin{tabular}{lccc}\toprule
 & \multicolumn{3}{c}{$\Delta$mAP (\%)}\\\cmidrule(lr){2-4}
Setting & Veh. & Ped. & Cyc. \\\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}
\caption{Per-class mAP change at $\lambda{=}""" + a.level + r"""$ with the default class factors ($\kappa_c$ = 1.0/1.5/1.3) and with uniform factors ($\kappa_c{=}1$).}
\label{tab:classscale}\end{table}
""")
    print('wrote', sorted(os.listdir(a.extra)))


if __name__ == '__main__':
    main()
