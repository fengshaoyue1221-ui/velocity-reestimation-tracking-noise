"""
Rebuild the result table and the three statistical figures of the paper from the raw evaluation logs.

    python make_paper_figures.py            # run from code/plotting/

Outputs:
    ../../results/all_results.csv      one row per (model, evaluation setting): mAP, minADE, minFDE, MR, per-class mAP
    output/fig_degradation.pdf    accuracy vs. noise severity (Fig. 2)
    output/fig_mechanism.pdf      paired decomposition (Fig. 3)
    output/fig_horizon_sens.pdf   horizon and temporal-correlation analysis (Fig. 4)
"""
import csv, glob, json, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # archive root
LOGS = os.path.join(ROOT, 'results', 'eval_logs')
SUM = os.path.join(ROOT, 'results')
FIG = os.path.join(HERE, 'output')
MODELS = [('clean', 'clean-trained', '#1f77b4', 'o', ''), ('aug', 'noise-aug', '#ff7f0e', 's', '////'),
          ('mix', 'mixed', '#2ca02c', '^', '....'), ('novel', 'no velocity', '#9467bd', 'D', 'xx')]


def parse(path):
    t = open(path).read()
    d = {m: float(v) for m, v in re.findall(r'^(minADE|minFDE|MissRate|mAP)\s*:\s*([-\d.]+)\s*$', t, re.M)}
    for m, c, v in re.findall(r'^(mAP) - (VEHICLE|PEDESTRIAN|CYCLIST)\s*:\s*([-\d.]+)\s*$', t, re.M):
        d[f'{m}-{c[:3]}'] = float(v)
    hz = {}
    for c, h, v in re.findall(r'^mAP - TYPE_(VEHICLE|PEDESTRIAN|CYCLIST)_(5|9|15)\s*:\s*([-\d.eE]+)', t, re.M):
        hz.setdefault(h, []).append(float(v))
    d['horizon'] = {h: sum(v) / len(v) for h, v in hz.items()}
    return d


def load():
    R = {}
    for f in glob.glob(os.path.join(LOGS, '*', '*', '*', 'log_eval_*.txt')):
        tag, et = f.split(os.sep)[-3], f.split(os.sep)[-2]
        d = parse(f)
        if 'mAP' in d:
            R[(tag, et)] = d
    return R


def write_csv(R):
    cols = ['mAP', 'minADE', 'minFDE', 'MissRate', 'mAP-VEH', 'mAP-PED', 'mAP-CYC']
    with open(os.path.join(SUM, 'all_results.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'setting'] + cols)
        for (tag, et) in sorted(R):
            w.writerow([tag, et] + [R[(tag, et)].get(c, '') for c in cols])


def rel(R, tag, et, m='mAP'):
    b = R[(tag, 'level_0.0')][m]
    return 100 * (R[(tag, et)][m] - b) / b


def fig_degradation(R):
    lv = ['0.0', '0.25', '0.5', '1.0', '2.0', '3.0']; x = [float(l) for l in lv]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95))
    for ax, (m, title) in zip(axes, [('mAP', r'mAP $\uparrow$'), ('minADE', r'minADE (m) $\downarrow$'), ('MissRate', r'Miss rate $\downarrow$')]):
        ax.axvspan(0.35, 0.4, color='0.85', lw=0); ax.axvspan(0.6, 0.8, color='0.92', lw=0)
        for tag, name, c, mk, _ in MODELS:
            ax.plot(x, [R[(tag, f'level_{l}')][m] for l in lv], marker=mk, ms=2.8, lw=1.2, color=c, label='MTR (%s)' % name)
        ax.set_title(title); ax.set_xlabel(r'noise severity $\lambda$'); ax.set_xticks([0, 0.5, 1, 2, 3]); ax.grid(alpha=0.3)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3, w_pad=1.2); fig.savefig(os.path.join(FIG, 'fig_degradation.pdf')); plt.close(fig)


def fig_mechanism(R):
    sets = [('mech_pos_offset', 'Offset\nonly'), ('mech_vel_derived', 'Velocity\nonly'), ('only_pos', 'Both'),
            ('level_1.0', 'All\nerrors'), ('mech_all_cleanvel', 'All, clean\nvelocity')]
    x = np.arange(len(sets)); w = 0.27
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    for k, (tag, name, c, _, hatch) in enumerate(MODELS[:3]):
        vals = [rel(R, tag, et) for et, _ in sets]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=name.capitalize(), color=c, hatch=hatch, edgecolor='black', linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v - 1.5, f'{v:.0f}', ha='center', va='top', fontsize=6.5)
    ax.axvline(2.5, color='0.6', lw=0.6, ls='--'); ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([s[1] for s in sets], fontsize=7.5)
    ax.set_ylabel(r'$\Delta$mAP (%) at $\lambda=1$'); ax.set_ylim(-82, 0)
    ax.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, 'fig_mechanism.pdf')); plt.close(fig)


def fig_horizon_sens(R):
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.35, 2.1)); hs = []
    for tag, name, c, mk, _ in MODELS:
        h0, h1 = R[(tag, 'level_0.0')]['horizon'], R[(tag, 'level_1.0')]['horizon']
        y = [100 * (h1[k] - h0[k]) / h0[k] for k in ['5', '9', '15']]
        line, = a.plot([3, 5, 8], y, marker=mk, ms=2.8, lw=1.2, color=c, label=name); hs.append(line)
    a.set_xticks([3, 5, 8]); a.set_xticklabels(['3 s', '5 s', '8 s']); a.set_xlabel('prediction horizon')
    a.set_ylabel(r'$\Delta$mAP (%) at $\lambda=1$'); a.set_title('(a) Horizon', pad=3); a.set_ylim(-90, 5); a.grid(alpha=0.3)
    rho = [0.0, 0.5, 0.8, 0.95]; amp = [np.sqrt(2 * (1 - r)) for r in rho]
    for tag, _, c, mk, _ in MODELS[:2]:
        y = [rel(R, tag, 'sens_rho0.0'), rel(R, tag, 'sens_rho0.5'), rel(R, tag, 'level_1.0'), rel(R, tag, 'sens_rho0.95')]
        b.plot(amp, y, marker=mk, ms=2.8, lw=1.2, color=c)
    b.set_xlim(0.15, 1.6); b.set_xlabel(r'$\sqrt{2(1-\rho)}$'); b.set_title(r'(b) Error correlation $\rho$', pad=3)
    b.set_ylim(-90, 5); b.set_yticklabels([]); b.grid(alpha=0.3)
    for ax in (a, b):
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.legend(handles=hs, loc='upper center', ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(pad=0.25, w_pad=0.6, rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(FIG, 'fig_horizon_sens.pdf')); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'hatch.linewidth': 0.6})
    R = load()
    write_csv(R)
    fig_degradation(R); fig_mechanism(R); fig_horizon_sens(R)
    print(f"{len(R)} result rows -> results/all_results.csv; figures written to code/plotting/output/")
