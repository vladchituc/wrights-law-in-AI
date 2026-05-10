"""
Wright's Law in AI — METR-17-only robustness check
Same 17 SOTA models, but cumulative compute computed using ONLY those 17 models'
training-FLOP estimates (rather than Epoch's full frontier-model corpus).

This is the methodologically stricter "single-dataset" version — useful as a
robustness check on the main analysis. The main analysis uses Epoch's full
corpus because Wright's Law is industry-level cumulative learning, not just
the subset METR evaluated. But both methods produce sublinear power laws
and Sahal's identity nearly closes in both.
"""
import os
import yaml, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.dates import date2num
from sklearn.linear_model import LinearRegression

np.random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
FIG_DIR = os.path.join(HERE, 'figures')

# Fonts (Montserrat preferred; fall back to default)
for f in ['/tmp/Montserrat-Regular.ttf','/tmp/Montserrat-Bold.ttf','/tmp/Montserrat-Medium.ttf']:
    if os.path.exists(f): fm.fontManager.addfont(f)
font_family = 'Montserrat' if any('Montserrat' in n.name for n in fm.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({
    'axes.spines.right':False,'axes.spines.top':False,
    'font.family':font_family,'font.weight':'regular','axes.titleweight':'regular',
    'figure.facecolor':'white','axes.facecolor':'white',
})

LOG_TICKS = np.array([1/60,2/60,4/60,8/60,15/60,30/60,1,2,4,8,15,30,60,2*60,4*60,8*60,16*60], dtype=float)
def fmt_time(s):
    s = round(s); h = s/3600
    if h >= 1: return f"{int(h)}h" if h == int(h) else f"{h:.1f}h"
    if h >= 1/60: return f"{int(h*60)}m"
    return f"{int(s)}s"
def fmt_doubling(v):
    if v < 1000: return f"{int(v)}×"
    if v < 1_000_000: return f"{int(v/1000)}K×"
    return f"{int(v/1_000_000)}M×"

# === Load METR 17 SOTA ===
with open(os.path.join(DATA_DIR, 'metr_benchmark.yaml')) as f: d = yaml.safe_load(f)
metr = pd.DataFrame([{
    'id': k, 'date': pd.Timestamp(v['release_date']),
    'y': v['metrics']['p50_horizon_length']['estimate'],
    'sota': v['metrics'].get('is_sota', False),
} for k, v in d['results'].items()]).sort_values('date').reset_index(drop=True)
metr = metr[metr.sota].sort_values('date').reset_index(drop=True)

# === Hand-mapped training FLOP for each of the 17 METR SOTA models ===
# Sources: Epoch corpus where matched; otherwise canonical public estimates and
# imputed values consistent with neighboring releases. Values include RL post-training
# (where reasoning-class models have non-trivial RL fraction).
METR_FLOP = {
    'gpt2':                                  1.5e21,
    'davinci_002':                           3.1e23,
    'gpt_3_5_turbo_instruct':                2.6e23,
    'gpt_4':                                 2.1e25,
    'gpt_4_1106_inspect':                    2.1e25,
    'gpt_4o_inspect':                        2.0e25,
    'claude_3_5_sonnet_20240620_inspect':    2.7e25,
    'o1_preview':                            2.5e25,
    'claude_3_5_sonnet_20241022_inspect':    3.0e25,
    'o1_inspect':                            3.5e25,
    'claude_3_7_sonnet_inspect':             7.5e25,
    'o3_inspect':                            8.0e25,
    'gpt_5_2025_08_07_inspect':              7.0e25,
    'gemini_3_pro':                          1.0e26,
    'claude_opus_4_5_inspect':               8.0e25,
    'gpt_5_2':                               1.1e26,
    'claude_opus_4_6_inspect':               1.0e26,
}
metr['own_flop'] = metr['id'].map(METR_FLOP)
metr['flop_cum'] = metr['own_flop'].cumsum()

ANCHOR_CUM = 1.5e21  # anchor at GPT-2's own training FLOP so it sits at exactly 1× on the relative axis

# === Fits ===
def fit_lin(x, y):
    X = np.asarray(x, float).reshape(-1,1); ly = np.log(np.clip(np.asarray(y, float), 1e-3, np.inf))
    return LinearRegression().fit(X, ly)
def fit_pow(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    a, lnA = np.polyfit(lx, ly, 1); A = np.exp(lnA)
    pred = lnA + a*lx
    r2 = 1 - np.sum((ly-pred)**2)/np.sum((ly-ly.mean())**2)
    return A, a, r2

metr_t = np.array([date2num(d) for d in metr['date']])
regA = fit_lin(metr_t, metr['y'].values); slopeA = regA.coef_[0]; dblA = np.log(2)/slopeA
ly_a = np.log(metr['y'].values); ly_p = regA.predict(metr_t.reshape(-1,1))
r2A = 1 - np.sum((ly_a-ly_p)**2)/np.sum((ly_a-ly_a.mean())**2)
regB = fit_lin(metr_t, metr['flop_cum'].values); slopeB = regB.coef_[0]; dblB = np.log(2)/slopeB
ly_a = np.log(metr['flop_cum'].values); ly_p = regB.predict(metr_t.reshape(-1,1))
r2B = 1 - np.sum((ly_a-ly_p)**2)/np.sum((ly_a-ly_a.mean())**2)
A_w, a_w, r2_w = fit_pow(metr['flop_cum'].values, metr['y'].values)
A_rel = A_w * (ANCHOR_CUM ** a_w)

# === Style ===
BLUE='#e91e63'; LABEL='#444'; WMARK='#bbb'; TEXT_DARK='#222'  # pink for METR-17 to distinguish
TITLE_FS=22; AXIS_FS=20; TICK_FS=17; LABEL_FS=17; ANNOT_FS=17
SUPTITLE_FS=28; WMARK_FS=16
DEX = 6.0

HEADLINE_LABELS = {
    'gpt2':'GPT-2','davinci_002':'GPT 3','gpt_3_5_turbo_instruct':'GPT 3.5',
    'gpt_4':'GPT-4','gpt_4o_inspect':'GPT 4o','o1_inspect':'o1',
    'claude_opus_4_5_inspect':'Claude Opus 4.5',
}

def label_with_bg(ax, x, y, text, off):
    ax.annotate(text, xy=(x,y), xytext=off, textcoords='offset points',
                fontsize=LABEL_FS, color=TEXT_DARK, fontweight='medium',
                va='center', zorder=8,
                bbox=dict(facecolor='white', edgecolor='#dddddd',
                          boxstyle='round,pad=0.25', linewidth=0.8, alpha=0.95))

def metr_panel(ax, df, ycol, reg, title, watermark_right, legend_label,
               ylim, ykind, anchor=None, label_pos=None):
    xlim = (pd.Timestamp('2018-09-01'), pd.Timestamp('2026-09-01'))
    xs_num = np.linspace(date2num(xlim[0]), date2num(xlim[1]), 250)
    xs_dt = mdates.num2date(xs_num)
    pred = np.exp(reg.predict(xs_num.reshape(-1,1)))
    if anchor: pred = pred/anchor
    dmin, dmax = df['date'].min(), df['date'].max()
    inside = (xs_num>=date2num(dmin)) & (xs_num<=date2num(dmax))
    ax.plot([d for d, m in zip(xs_dt, inside) if m], pred[inside], color=BLUE, lw=2.6, zorder=4)
    ax.plot([d for d, m in zip(xs_dt, inside) if not m], pred[~inside], color=BLUE, lw=2.6, ls='--', alpha=0.85, zorder=4)
    yvals = df[ycol].values / anchor if anchor else df[ycol].values
    ax.scatter(df['date'], yvals, s=140, color=BLUE, edgecolor='white', linewidth=1.2, zorder=6)
    for key, txt in HEADLINE_LABELS.items():
        sub = df[df['id']==key]
        if len(sub):
            r = sub.iloc[0]
            yv = r[ycol]/anchor if anchor else r[ycol]
            off = (label_pos or {}).get(key, (12,-3))
            label_with_bg(ax, r['date'], yv, txt, off)
    ax.set_yscale('log')
    if ykind == 'time':
        t = LOG_TICKS[(LOG_TICKS>=ylim[0]) & (LOG_TICKS<=ylim[1])]
        ax.set_yticks(t); ax.set_yticklabels([fmt_time(x*60) for x in t])
    elif ykind == 'doublings':
        all_pows = [2**k for k in range(0,25)]
        t = [v for v in all_pows if v>=ylim[0] and v<=ylim[1]]
        labels = [fmt_doubling(v) if i%2==0 else '' for i, v in enumerate(t)]
        ax.set_yticks(t); ax.set_yticklabels(labels)
    ax.set_ylim(*ylim); ax.set_xlim(xlim[0], xlim[1])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#999'); ax.spines['bottom'].set_color('#999')
    ax.tick_params(colors=TEXT_DARK, labelsize=TICK_FS)
    for tl in ax.get_xticklabels()+ax.get_yticklabels(): tl.set_fontweight('medium')
    ax.set_xlabel('Model release date', fontsize=AXIS_FS, labelpad=12, color=TEXT_DARK, fontweight='medium')
    ax.text(0.0,1.06, title, transform=ax.transAxes, fontsize=TITLE_FS, fontweight='bold', color='#222', ha='left', va='bottom')
    ax.text(0.97,0.04, legend_label, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=ANNOT_FS, color=TEXT_DARK, fontweight='medium',
            bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.6', linewidth=0.8))
    ax.text(0.0,-0.18,'CC-BY', transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='left', va='top', fontweight='medium')
    ax.text(1.0,-0.18, watermark_right, transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='right', va='top', fontweight='medium')

labelsA = {'gpt2':(12,-3),'davinci_002':(14,-3),'gpt_3_5_turbo_instruct':(14,-3),
           'gpt_4':(-58,-15),'gpt_4o_inspect':(-78,-15),'o1_inspect':(14,3),
           'claude_opus_4_5_inspect':(-180,-3)}
labelsB = {'gpt2':(12,-3),'davinci_002':(14,-3),'gpt_3_5_turbo_instruct':(14,3),
           'gpt_4':(-58,-15),'gpt_4o_inspect':(-78,15),'o1_inspect':(14,3),
           'claude_opus_4_5_inspect':(-180,-3)}
labelsC = {'gpt2':(15,-3),'davinci_002':(15,-3),'gpt_3_5_turbo_instruct':(15,-3),
           'gpt_4':(-72,12),'gpt_4o_inspect':(-78,-15),'o1_inspect':(15,3),
           'claude_opus_4_5_inspect':(-180,-3)}

fig, ax3 = plt.subplots(1, 3, figsize=(28, 9.5))
plt.subplots_adjust(top=0.78, bottom=0.13, left=0.04, right=0.985, wspace=0.22)

ylim_h = (1/200, 1/200 * 10**DEX)
metr_panel(ax3[0], metr, 'y', regA,
    'A. Task horizons double every 6 months',
    'metr.org', f'Doubling time: {dblA:.0f} days\nR² = {r2A:.2f}',
    ylim=ylim_h, ykind='time', label_pos=labelsA)

ylim_c = (1, 10**DEX)
metr_panel(ax3[1], metr, 'flop_cum', regB,
    f'B. Cumulative compute (METR-17 only) doubles every {dblB/30.4:.0f} months',
    'metr.org', f'Doubling time: {dblB:.0f} days\nR² = {r2B:.2f}',
    ylim=ylim_c, ykind='doublings', anchor=ANCHOR_CUM, label_pos=labelsB)

metr['rel_cum'] = metr['flop_cum'] / ANCHOR_CUM
ax = ax3[2]
xs = np.geomspace(1, 10**DEX, 250)
ax.plot(xs, A_rel*xs**a_w, color=BLUE, lw=2.6, zorder=5)
ax.scatter(metr['rel_cum'], metr['y'], s=140, color=BLUE, edgecolor='white', lw=1.2, zorder=6)
for key, txt in HEADLINE_LABELS.items():
    sub = metr[metr['id']==key]
    if len(sub):
        r = sub.iloc[0]
        off = labelsC.get(key, (15,-3))
        label_with_bg(ax, r['rel_cum'], r['y'], txt, off)
ax.set_xscale('log'); ax.set_yscale('log')
all_pows = [2**k for k in range(0,25)]
xt = [v for v in all_pows if v>=1 and v<=10**DEX]
xt_labels = [fmt_doubling(v) if i%2==0 else '' for i, v in enumerate(xt)]
ax.set_xticks(xt); ax.set_xticklabels(xt_labels)
yt = LOG_TICKS[(LOG_TICKS>=1/60) & (LOG_TICKS<=16*60)]
ax.set_yticks(yt); ax.set_yticklabels([fmt_time(t*60) for t in yt])
ax.set_xlim(1, 10**DEX); ax.set_ylim(1/60, 16*60)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.spines['left'].set_color('#999'); ax.spines['bottom'].set_color('#999')
ax.tick_params(colors=TEXT_DARK, labelsize=TICK_FS)
for tl in ax.get_xticklabels()+ax.get_yticklabels(): tl.set_fontweight('medium')
ax.set_xlabel('Relative cumulative compute (METR-17 only)', fontsize=AXIS_FS, labelpad=12, color=TEXT_DARK, fontweight='medium')
ax.text(0.0,1.06, f'C. Doubling task horizon needs {2**(1/a_w):.1f}× more compute',
        transform=ax.transAxes, fontsize=TITLE_FS, fontweight='bold', color='#222', ha='left', va='bottom')
ax.text(0.97,0.04, f'horizon $\\propto$ cumFLOP$^{{{a_w:.2f}}}$\nR² = {r2_w:.2f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=ANNOT_FS, color=TEXT_DARK, fontweight='medium',
        bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.6', linewidth=0.8))
ax.text(0.0,-0.18,'CC-BY', transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='left', va='top', fontweight='medium')
ax.text(1.0,-0.18,'metr.org', transform=ax.transAxes, fontsize=WMARK_FS, color=WMARK, ha='right', va='top', fontweight='medium')

fig.canvas.draw()
bold_t = fig.text(0.04, 0.965, "Wright's Law in AI (METR-17 robustness check):",
                  fontsize=SUPTITLE_FS, fontweight='bold', color='#222',
                  ha='left', va='top', family=font_family)
renderer = fig.canvas.get_renderer()
bbox = bold_t.get_window_extent(renderer=renderer)
fig_w_pix = fig.get_figwidth() * fig.dpi
x_after = (bbox.x1 / fig_w_pix) + 0.005
fig.text(x_after, 0.965,
         "same 17 SOTA models, cumulative compute from those 17 only",
         fontsize=SUPTITLE_FS, fontweight='regular', color='#333',
         ha='left', va='top', family=font_family)

out_path = os.path.join(FIG_DIR, 'wrights_law_in_AI_metr17.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'saved: {out_path}')

print(f"\nFINAL FIT NUMBERS (n=17 METR SOTA, METR-only cumFLOP)")
print(f"Panel A: horizon doubling = {dblA:.0f} d ({dblA/30.4:.1f} mo), R²={r2A:.3f}")
print(f"Panel B: cumFLOP doubling = {dblB:.0f} d ({dblB/30.4:.1f} mo), R²={r2B:.3f}")
print(f"Panel C: horizon = {A_w:.2e} * cumFLOP^{a_w:.3f}, R²={r2_w:.3f}")
print(f"  → 2× horizon needs {2**(1/a_w):.2f}× cumulative compute")
print(f"  → predicted horizon-doubling = {dblB/a_w:.1f} d vs observed {dblA:.0f} d (diff {dblB/a_w-dblA:+.1f})")
